//! EU Volt Interview Bot — voice interview server (Rustvani + Dhara).
//!
//! Rust port of the Python Pipecat Gemini Live interviewer.
//! Uses Dhara conversation flow with one node per interview question:
//!   intro → q1 → q2 → … → q10 → farewell
//!
//! Pipeline:
//!   WebSocketTransport.input()
//!     → RaviProcessor
//!     → SarvamStt
//!     → LLMUserAggregator
//!     → OpenAILLM (with Dhara transition hook)
//!     → LLMAssistantAggregator
//!     → DeepgramTts
//!     → WebSocketTransport.output()
//!
//! Environment variables:
//!   PORT             — listen port (default: 10000)
//!   SARVAM_API_KEY   — required (STT)
//!   OPENAI_API_KEY   — required (LLM)
//!   DEEPGRAM_API_KEY — required (TTS)

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use axum::{
    Router,
    extract::{State, WebSocketUpgrade, ws::WebSocket},
    response::IntoResponse,
    routing::get,
};
use serde_json::{json, Value};
use tower_http::cors::CorsLayer;

use rustvani::{
    system_clock, SileroVadNative, VadParams,
    PipelineParams, PipelineTask,
};
use rustvani::adapters::schemas::{FunctionSchema, ToolsSchema};
use rustvani::context::LLMContext;
use rustvani::dhara::{ContextStrategy, DharaManager, NodeConfig, TransitionResult};
use rustvani::observer::{BaseObserver, FrameProcessed, FramePushed};
use rustvani::processors::{
    llm_assistant_aggregator::LLMAssistantAggregator,
    llm_user_aggregator::LLMUserAggregator,
};
use rustvani::ravi::{
    RaviObserverParams,
    processor::{RaviParams, RaviProcessor},
};
use rustvani::services::{
    OpenAILLMConfig, OpenAILLMHandler,
    SarvamSttConfig, SarvamSttHandler,
    DeepgramTtsConfig, DeepgramTtsHandler,
};
use rustvani::frames::{Frame, FrameDirection};
use rustvani::ravi::models as ravi_models;
use rustvani::services::llm::function_registry::FunctionRegistry;
use rustvani::transport::websocket::{WebSocketParams, WebSocketTransport};
use rustvani::transport::TransportParams;
use rustvani::turn::SmartTurnConfig;

// ---------------------------------------------------------------------------
// Deferred push sender
// ---------------------------------------------------------------------------

type PushSender = tokio::sync::mpsc::Sender<(Frame, FrameDirection)>;
type DeferredPush = Arc<std::sync::OnceLock<PushSender>>;

// ---------------------------------------------------------------------------
// Connection ID counter
// ---------------------------------------------------------------------------

static CONN_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_conn_id() -> u64 {
    CONN_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Shared app state
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AppState {
    sarvam_api_key:   String,
    openai_api_key:   String,
    deepgram_api_key: String,
}

// ---------------------------------------------------------------------------
// Interview state — one per connection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct InterviewState {
    candidate_name:  Option<String>,
    questions_asked: u32,
}

// ---------------------------------------------------------------------------
// Ravi client push helper
// ---------------------------------------------------------------------------

async fn push_ravi_msg(push: &DeferredPush, data: Value) {
    if let Some(tx) = push.get() {
        let payload = ravi_models::msg_server_message(data);
        let frame = Frame::ravi_server_message(payload);
        if let Err(e) = tx.send((frame, FrameDirection::Downstream)).await {
            log::error!("push_ravi_msg: send failed: {}", e);
        }
    } else {
        log::warn!("push_ravi_msg: deferred sender not yet initialized");
    }
}

// ---------------------------------------------------------------------------
// Interview questions — the 10 priority topics
// ---------------------------------------------------------------------------

struct Question {
    id:     u32,
    topic:  &'static str,
    prompt: &'static str,
}

const QUESTIONS: &[Question] = &[
    Question {
        id: 1,
        topic: "Relevant experience",
        prompt: "Ask the candidate: What part of your previous experience do you think \
                 will be most valuable in this Production Technician role? \
                 Listen carefully and ask a brief follow-up if their answer is vague.",
    },
    Question {
        id: 2,
        topic: "HMI troubleshooting",
        prompt: "Ask the candidate: You encounter an error message on the HMI that halts \
                 production. How do you approach diagnosing and fixing it? \
                 Probe for a systematic thought process.",
    },
    Question {
        id: 3,
        topic: "Chemical handling",
        prompt: "Ask the candidate: Do you have any experience handling chemicals? \
                 What chemicals have you worked with, and what safety protocols did you follow? \
                 Note any red flags around safety awareness.",
    },
    Question {
        id: 4,
        topic: "Quality deviation",
        prompt: "Ask the candidate: If you notice a recurring deviation in product quality, \
                 what steps would you take? \
                 Look for a structured quality-mindset.",
    },
    Question {
        id: 5,
        topic: "Conflict resolution under pressure",
        prompt: "Ask the candidate: Imagine you and another technician disagree on how to \
                 troubleshoot an HMI error. Production is halted and time is critical. \
                 How would you approach the situation? \
                 Evaluate teamwork and communication under stress.",
    },
    Question {
        id: 6,
        topic: "Chemical spill response",
        prompt: "Ask the candidate: What do you do if there is a chemical spill in your workplace? \
                 Check for knowledge of emergency procedures and PPE.",
    },
    Question {
        id: 7,
        topic: "Reducing downtime",
        prompt: "Ask the candidate: If you were tasked with reducing downtime by 10 percent, \
                 what methods or tools would you use? \
                 Look for process-improvement thinking.",
    },
    Question {
        id: 8,
        topic: "Stress management",
        prompt: "Ask the candidate: How do you handle stress when production deadlines are tight? \
                 Keep the tone empathetic — this is a personal question.",
    },
    Question {
        id: 9,
        topic: "Salary expectations",
        prompt: "Ask the candidate: What are your salary expectations for this role? \
                 Be warm and non-judgmental regardless of the number. \
                 Note if it is wildly out of range as a red flag.",
    },
    Question {
        id: 10,
        topic: "Five-year vision",
        prompt: "Ask the candidate: Where do you see yourself in five years — as a technician, \
                 specialist, or in leadership? \
                 This is the final question. After the candidate responds, \
                 thank them warmly and then call end_conversation with your full scoring.",
    },
];

// ---------------------------------------------------------------------------
// Shared system prompt (injected into every question node)
// ---------------------------------------------------------------------------

const BASE_SYSTEM_PROMPT: &str = "\
You are William, a friendly and professional interviewer for EU Volt, a sustainable \
energy storage company specializing in advanced battery production for vehicles. \
EU Volt has 1000 skilled professionals and is committed to Courage, Integrity, \
Collaboration, and Innovation. You are interviewing for a Production Technician \
position in Zurich.

Keep all responses brief — they are converted to speech. One or two sentences per turn. \
Do not use special characters. Be conversational, encouraging, and warm. \
Acknowledge the candidate's answers before moving on. If an answer is unclear or \
too short, ask a natural follow-up — but do not interrogate. \
When you are satisfied with the answer (or after one follow-up), call next_question \
to move on.";

const SCORING_INSTRUCTIONS: &str = "\
SCORING CRITERIA (use when calling end_conversation):

1. Technical Knowledge (30 points): HMI/PLC experience, production line understanding, \
   equipment maintenance, battery/chemical production familiarity.
2. Problem-Solving (25 points): Systematic diagnostics, logical thinking under pressure, \
   process improvement mindset, quality issue resolution.
3. Safety Awareness (20 points): Chemical handling protocols, emergency response, PPE usage, \
   risk assessment.
4. Soft Skills (15 points): Teamwork, stress management, communication, training ability.
5. Cultural Fit (10 points): Alignment with EU Volt values, realistic career goals, \
   sustainability commitment, reasonable salary expectations.

RED FLAGS: No chemical safety knowledge, no relevant technical experience, \
poor teamwork, unrealistic salary demands, no interest in sustainability.

90-100 Exceptional | 75-89 Good | 60-74 Average | 45-59 Below average | 0-44 Poor fit";

// ---------------------------------------------------------------------------
// Tool schemas
// ---------------------------------------------------------------------------

fn begin_interview_schema() -> FunctionSchema {
    FunctionSchema::new(
        "begin_interview",
        "Candidate has introduced themselves. Call this to start the first question."
    )
    .with_parameters(json!({
        "type": "object",
        "properties": {
            "candidate_name": {
                "type": "string",
                "description": "The candidate's name as they introduced themselves"
            }
        },
        "required": []
    }))
}

fn next_question_schema() -> FunctionSchema {
    FunctionSchema::new(
        "next_question",
        "Move to the next interview question. Call this when you are satisfied \
         with the candidate's answer to the current question."
    )
    .with_parameters(json!({
        "type": "object",
        "properties": {
            "notes": {
                "type": "string",
                "description": "Brief internal notes on the candidate's answer to this question"
            }
        },
        "required": []
    }))
}

fn end_conversation_schema() -> FunctionSchema {
    FunctionSchema::new(
        "end_conversation",
        "End the interview with a full score breakdown. Call after the final question, \
         or immediately if the candidate asks to stop."
    )
    .with_parameters(json!({
        "type": "object",
        "properties": {
            "technical_knowledge": {
                "type": "integer",
                "description": "Score for technical knowledge (0-30)"
            },
            "problem_solving": {
                "type": "integer",
                "description": "Score for problem-solving (0-25)"
            },
            "safety_awareness": {
                "type": "integer",
                "description": "Score for safety awareness (0-20)"
            },
            "soft_skills": {
                "type": "integer",
                "description": "Score for soft skills (0-15)"
            },
            "cultural_fit": {
                "type": "integer",
                "description": "Score for cultural fit (0-10)"
            },
            "total_score": {
                "type": "integer",
                "description": "Total score (0-100)"
            },
            "red_flags": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Any red flags identified"
            },
            "summary": {
                "type": "string",
                "description": "Interview summary — key topics, strengths, weaknesses, highlights (max 10 sentences)"
            }
        },
        "required": [
            "technical_knowledge", "problem_solving", "safety_awareness",
            "soft_skills", "cultural_fit", "total_score", "red_flags", "summary"
        ]
    }))
}

// ---------------------------------------------------------------------------
// Node config factories
// ---------------------------------------------------------------------------

fn intro_node() -> NodeConfig {
    NodeConfig::new("intro")
        .with_system_prompt(BASE_SYSTEM_PROMPT)
        .with_task_message(
            "Greet the candidate warmly: Hi, great to see you and thanks for joining \
             us today. My name is William and I am coordinating this interview on behalf \
             of EU Volt. We are expanding our team and looking for skilled production \
             technicians. This will take around 10 minutes — a few questions about your \
             background, then some technical scenarios. Feel free to ask if anything is \
             unclear. Before we dive in, could you briefly introduce yourself? \
             Once they introduce themselves, call begin_interview."
        )
        .with_tools(ToolsSchema::new(vec![begin_interview_schema()]))
        .with_respond_immediately(true)
}

/// Build a question node. Every question node carries:
///   - next_question tool (transitions to the next node)
///   - end_conversation tool (early exit if candidate requests)
///
/// The last question (q10) omits next_question — only end_conversation.
fn question_node(q: &Question, is_last: bool) -> NodeConfig {
    let system = format!(
        "{}\n\nYou are on question {} of 10. Topic: {}.\n\n{}",
        BASE_SYSTEM_PROMPT, q.id, q.topic, SCORING_INSTRUCTIONS
    );

    let tools = if is_last {
        ToolsSchema::new(vec![end_conversation_schema()])
    } else {
        ToolsSchema::new(vec![next_question_schema(), end_conversation_schema()])
    };

    NodeConfig::new(&format!("q{}", q.id))
        .with_system_prompt(&system)
        .with_task_message(q.prompt)
        .with_tools(tools)
        .with_context_strategy(ContextStrategy::Append)
        .with_respond_immediately(true)
}

fn farewell_node() -> NodeConfig {
    NodeConfig::new("farewell")
        .with_task_message(
            "The interview is complete and scored. Thank the candidate briefly for their time, \
             let them know the EU Volt recruitment team will be in touch, and say goodbye warmly. \
             Two or three sentences."
        )
        .with_tools(ToolsSchema::new(vec![]))
        .with_context_strategy(ContextStrategy::Append)
        .with_respond_immediately(true)
}

// ---------------------------------------------------------------------------
// Dhara handler factories
// ---------------------------------------------------------------------------

fn make_begin_interview_handler(
    interview: Arc<Mutex<InterviewState>>,
) -> rustvani::dhara::DharaHandlerFn {
    Arc::new(move |args: String| {
        let interview = interview.clone();
        Box::pin(async move {
            let parsed: Value = serde_json::from_str(&args).unwrap_or_default();
            let name = parsed["candidate_name"].as_str().map(String::from);

            {
                let mut state = interview.lock().unwrap();
                state.candidate_name = name.clone();
            }

            log::info!(
                "Interview started — candidate: {}",
                name.as_deref().unwrap_or("unnamed")
            );

            TransitionResult::transition(
                json!({
                    "status": "interview_started",
                    "candidate_name": name,
                }).to_string(),
                "q1",
            )
        })
    })
}

/// Factory for the `next_question` handler.
/// `current_q` is 1-based, `next_node` is the target (e.g. "q2").
fn make_next_question_handler(
    interview: Arc<Mutex<InterviewState>>,
    current_q: u32,
    next_node: &'static str,
    push: DeferredPush,
) -> rustvani::dhara::DharaHandlerFn {
    Arc::new(move |args: String| {
        let interview = interview.clone();
        let push = push.clone();
        Box::pin(async move {
            let parsed: Value = serde_json::from_str(&args).unwrap_or_default();
            let notes = parsed["notes"].as_str().unwrap_or("").to_string();

            {
                let mut state = interview.lock().unwrap();
                state.questions_asked = current_q;
            }

            log::info!("Q{} complete → {}. Notes: {}", current_q, next_node, notes);

            // Push progress to client UI
            push_ravi_msg(&push, json!({
                "type": "interview_progress",
                "question_completed": current_q,
                "total_questions": 10,
                "notes": notes,
            })).await;

            TransitionResult::transition(
                json!({
                    "status": "moving_on",
                    "completed_question": current_q,
                    "instruction": "Ask the next question naturally. \
                                    Briefly acknowledge what they said before transitioning.",
                }).to_string(),
                next_node,
            )
        })
    })
}

fn make_end_conversation_handler(
    interview: Arc<Mutex<InterviewState>>,
    push: DeferredPush,
) -> rustvani::dhara::DharaHandlerFn {
    Arc::new(move |args: String| {
        let interview = interview.clone();
        let push = push.clone();
        Box::pin(async move {
            let parsed: Value = serde_json::from_str(&args).unwrap_or_default();

            let tech    = parsed["technical_knowledge"].as_u64().unwrap_or(0).min(30) as u32;
            let problem = parsed["problem_solving"].as_u64().unwrap_or(0).min(25) as u32;
            let safety  = parsed["safety_awareness"].as_u64().unwrap_or(0).min(20) as u32;
            let soft    = parsed["soft_skills"].as_u64().unwrap_or(0).min(15) as u32;
            let culture = parsed["cultural_fit"].as_u64().unwrap_or(0).min(10) as u32;
            let total   = parsed["total_score"].as_u64().unwrap_or(0).min(100) as u32;

            let red_flags: Vec<String> = parsed["red_flags"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();

            let summary = parsed["summary"].as_str().unwrap_or("").to_string();
            let candidate_name = interview.lock().unwrap().candidate_name.clone();

            // ---- Console output ----
            println!("\n{}", "=".repeat(60));
            println!("INTERVIEW COMPLETED");
            println!("{}", "=".repeat(60));
            if let Some(ref name) = candidate_name {
                println!("\nCandidate: {}", name);
            }
            println!("\n  SCORE BREAKDOWN:\n");
            println!("  Technical Knowledge:      {}/30", tech);
            println!("  Problem-Solving:          {}/25", problem);
            println!("  Safety Awareness:         {}/20", safety);
            println!("  Soft Skills:              {}/15", soft);
            println!("  Cultural Fit:             {}/10", culture);
            println!("  {}", "-".repeat(40));
            println!("  TOTAL SCORE:              {}/100\n", total);

            if !red_flags.is_empty() {
                println!("  RED FLAGS:");
                for flag in &red_flags {
                    println!("    - {}", flag);
                }
                println!();
            }

            println!("  SUMMARY:\n{}\n", summary);
            println!("{}", "=".repeat(60));

            // ---- Push to client ----
            push_ravi_msg(&push, json!({
                "type": "interview_completed",
                "candidate_name": candidate_name,
                "score_breakdown": {
                    "technical_knowledge": tech,
                    "problem_solving": problem,
                    "safety_awareness": safety,
                    "soft_skills": soft,
                    "cultural_fit": culture,
                    "total_score": total,
                },
                "red_flags": red_flags,
                "summary": summary,
            })).await;

            TransitionResult::transition(
                json!({
                    "status": "interview_scored",
                    "total_score": total,
                    "instruction": "Thank the candidate warmly and say goodbye.",
                }).to_string(),
                "farewell",
            )
        })
    })
}

// ---------------------------------------------------------------------------
// NullObserver
// ---------------------------------------------------------------------------

struct NullObserver;

#[async_trait]
impl BaseObserver for NullObserver {
    async fn on_process_frame(&self, _: FrameProcessed) {}
    async fn on_push_frame(&self, _: FramePushed) {}
}

// ---------------------------------------------------------------------------
// ConnectionFlow
// ---------------------------------------------------------------------------

struct ConnectionFlow {
    context:         Arc<Mutex<LLMContext>>,
    registry:        Arc<Mutex<FunctionRegistry>>,
    transition_hook: rustvani::services::llm::openai::TransitionHook,
    push_tx:         DeferredPush,
}

/// Node names for questions 1–10.
const Q_NODES: &[&str] = &[
    "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10",
];

fn build_flow() -> ConnectionFlow {
    let interview = Arc::new(Mutex::new(InterviewState::default()));
    let context   = Arc::new(Mutex::new(LLMContext::new(None)));
    let registry  = Arc::new(Mutex::new(FunctionRegistry::new()));
    let push_tx: DeferredPush = Arc::new(std::sync::OnceLock::new());

    // Seed context so the bot speaks first on connect.
    {
        let mut ctx = context.lock().unwrap();
        ctx.add_user_message("Start the interview now. Greet the candidate and introduce yourself.");
    }

    let mut dhara = DharaManager::new(context.clone(), registry.clone());

    // ---- intro node ----
    dhara.register_node("intro", intro_node(), vec![
        ("begin_interview", make_begin_interview_handler(interview.clone())),
    ]);

    // ---- question nodes q1–q10 ----
    for (i, q) in QUESTIONS.iter().enumerate() {
        let is_last = i == QUESTIONS.len() - 1;
        let node_config = question_node(q, is_last);

        if is_last {
            // q10 — only end_conversation, no next_question
            dhara.register_node(Q_NODES[i], node_config, vec![
                ("end_conversation", make_end_conversation_handler(
                    interview.clone(), push_tx.clone(),
                )),
            ]);
        } else {
            // q1–q9 — next_question → q(n+1), plus early-exit end_conversation
            let next_node: &'static str = Q_NODES[i + 1];
            dhara.register_node(Q_NODES[i], node_config, vec![
                ("next_question", make_next_question_handler(
                    interview.clone(),
                    q.id,
                    next_node,
                    push_tx.clone(),
                )),
                ("end_conversation", make_end_conversation_handler(
                    interview.clone(), push_tx.clone(),
                )),
            ]);
        }
    }

    // ---- farewell node ----
    dhara.register_node_no_tools("farewell", farewell_node());

    dhara.set_initial_node("intro");

    let transition_hook = dhara.create_transition_hook();

    ConnectionFlow { context, registry, transition_hook, push_tx }
}

// ---------------------------------------------------------------------------
// WebSocket handler
// ---------------------------------------------------------------------------

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_connection(socket, state))
}

async fn handle_connection(socket: WebSocket, app_state: AppState) {
    let conn_id = next_conn_id();
    log::info!("[conn={}] connected — starting interview flow", conn_id);

    // ---- VAD ----
    let vad_analyzer = match SileroVadNative::new(16_000) {
        Ok(v) => Arc::new(v),
        Err(e) => {
            log::error!("[conn={}] VAD init failed: {}", conn_id, e);
            return;
        }
    };

    // ---- Transport ----
    let transport = WebSocketTransport::new(
        &format!("WsTransport-{}", conn_id),
        WebSocketParams {
            transport: TransportParams {
                audio_in_enabled:         true,
                audio_in_sample_rate:     Some(16_000),
                audio_in_channels:        1,
                audio_in_passthrough:     true,
                audio_in_stream_on_start: true,
                vad_analyzer:             Some(vad_analyzer),
                vad_params:               VadParams {
                    confidence: 0.4,
                    min_volume: 0.1,
                    ..VadParams::default()
                },
                turn_config:              Some(SmartTurnConfig::default()),
                ..TransportParams::default()
            },
        },
    );

    // ---- Dhara flow (fresh per connection) ----
    let flow = build_flow();

    // ---- RAVI ----
    let ravi = RaviProcessor::new(RaviParams {
        context: Some(flow.context.clone()),
        ..RaviParams::default()
    });

    let ravi_observer: Arc<dyn BaseObserver> = Arc::new(
        RaviProcessor::create_observer(&ravi, RaviObserverParams::default()),
    );

    // ---- STT ----
    let stt = SarvamSttHandler::new(SarvamSttConfig {
        api_key:  app_state.sarvam_api_key.clone(),
        model:    "saaras:v3".to_string(),
        language: Some("en-IN".to_string()),
        mode:     Some("transcribe".to_string()),
        ..SarvamSttConfig::default()
    })
    .into_processor();

    // ---- Aggregators ----
    let user_agg      = LLMUserAggregator::new(flow.context.clone());
    let assistant_agg = LLMAssistantAggregator::new(flow.context.clone());

    // ---- LLM with Dhara transition hook ----
    let mut llm_handler = OpenAILLMHandler::with_shared_registry(
        OpenAILLMConfig {
            api_key:         app_state.openai_api_key.clone(),
            model:           "gpt-4o-mini".to_string(),
            max_tool_rounds: 5,
            ..OpenAILLMConfig::default()
        },
        flow.registry.clone(),
    );
    llm_handler.set_transition_hook(flow.transition_hook);
    let llm = llm_handler.into_processor();

    // ---- TTS (Deepgram) ----
    let tts = match DeepgramTtsHandler::new(DeepgramTtsConfig {
        api_key: app_state.deepgram_api_key.clone(),
        ..DeepgramTtsConfig::default()
    }) {
        Ok(t) => t.into_processor(),
        Err(e) => {
            log::error!("[conn={}] TTS init failed: {}", conn_id, e);
            return;
        }
    };

    // ---- Pipeline ----
    let task = PipelineTask::new(
        vec![
            transport.input(),
            ravi,
            stt,
            user_agg,
            llm,
            assistant_agg,
            tts,
            transport.output(),
        ],
        PipelineParams { allow_interruptions: true, ..PipelineParams::default() },
    );

    // Wire the deferred push sender
    let _ = flow.push_tx.set(task.push_sender());

    let push_tx = task.push_sender();

    // Kick off the first LLM run so the bot greets proactively.
    let startup_tx = task.push_sender();
    let startup_ctx = flow.context.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        let _ = startup_tx.send((Frame::llm_context(startup_ctx), FrameDirection::Downstream)).await;
    });

    tokio::join!(
        async { task.run(system_clock(), Some(ravi_observer)).await.ok(); },
        transport.run_socket(socket, push_tx),
    );

    log::info!("[conn={}] disconnected", conn_id);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
    .init();

    let sarvam_api_key = std::env::var("SARVAM_API_KEY")
        .expect("SARVAM_API_KEY env var not set");

    let openai_api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY env var not set");

    let deepgram_api_key = std::env::var("DEEPGRAM_API_KEY")
        .expect("DEEPGRAM_API_KEY env var not set");

    let app_state = AppState { sarvam_api_key, openai_api_key, deepgram_api_key };

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    let port = std::env::var("PORT").unwrap_or_else(|_| "10000".to_string());
    let addr = format!("0.0.0.0:{}", port);

    log::info!("EU Volt Interview Bot on ws://{}/ws", addr);
    log::info!("Flow: intro → q1 → q2 → … → q10 → farewell");

    let listener = tokio::net::TcpListener::bind(&addr).await
        .unwrap_or_else(|e| panic!("Failed to bind {}: {}", addr, e));

    axum::serve(listener, app).await.unwrap();
}