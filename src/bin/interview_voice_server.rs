//! EU Volt Interview Bot — voice interview server (Rustvani + Dhara).
//!
//! Rust port of the Python Pipecat Gemini Live interviewer.
//! Uses Dhara conversation flow for structured interview phases:
//!   intro → interview → farewell
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
// Deferred push sender — set after PipelineTask::new(), used by handlers
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

#[derive(Debug, Clone)]
struct ScoreBreakdown {
    technical_knowledge: u32,
    problem_solving:     u32,
    safety_awareness:    u32,
    soft_skills:         u32,
    cultural_fit:        u32,
    total_score:         u32,
}

#[derive(Debug, Clone, Default)]
struct InterviewState {
    candidate_name:  Option<String>,
    questions_asked: u32,
}

impl InterviewState {
    fn summary_for_llm(&self) -> String {
        format!(
            "Questions asked so far: {}/10. Candidate name: {}.",
            self.questions_asked,
            self.candidate_name.as_deref().unwrap_or("not yet provided"),
        )
    }
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
// Tool schemas
// ---------------------------------------------------------------------------

fn begin_interview_schema() -> FunctionSchema {
    FunctionSchema::new(
        "begin_interview",
        "Transition from introduction to the interview phase. \
         Call this after the candidate has introduced themselves \
         and you are ready to start asking interview questions."
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

fn end_conversation_schema() -> FunctionSchema {
    FunctionSchema::new(
        "end_conversation",
        "Gracefully end the interview by providing a score breakdown and summary. \
         Call this after asking all 10 questions, or if the candidate asks to stop."
    )
    .with_parameters(json!({
        "type": "object",
        "properties": {
            "technical_knowledge": {
                "type": "integer",
                "description": "Score for technical knowledge — HMI/PLC, production lines, equipment. 0 to 30."
            },
            "problem_solving": {
                "type": "integer",
                "description": "Score for problem-solving and troubleshooting abilities. 0 to 25."
            },
            "safety_awareness": {
                "type": "integer",
                "description": "Score for safety protocols and chemical handling knowledge. 0 to 20."
            },
            "soft_skills": {
                "type": "integer",
                "description": "Score for communication, teamwork, and stress management. 0 to 15."
            },
            "cultural_fit": {
                "type": "integer",
                "description": "Score for alignment with company values and career goals. 0 to 10."
            },
            "total_score": {
                "type": "integer",
                "description": "Total interview score, sum of all categories. 0 to 100."
            },
            "red_flags": {
                "type": "array",
                "items": { "type": "string" },
                "description": "List of any red flags identified during the interview"
            },
            "summary": {
                "type": "string",
                "description": "Interview summary (max 10 sentences): key topics, strengths, weaknesses, highlights"
            }
        },
        "required": [
            "technical_knowledge", "problem_solving", "safety_awareness",
            "soft_skills", "cultural_fit", "total_score", "red_flags", "summary"
        ]
    }))
}

// ---------------------------------------------------------------------------
// System prompts
// ---------------------------------------------------------------------------

const INTRO_SYSTEM_PROMPT: &str = "\
You are an interviewer named William, conducting interviews on behalf of EU Volt, \
a company at the forefront of sustainable energy storage specializing in advanced \
battery production for vehicles. EU Volt has 1000 skilled professionals and is \
committed to driving innovation in the green transition through Courage, Integrity, \
Collaboration, and Innovation.

You are interviewing for a Production Technician position in Zurich.

Keep your responses brief since they will be converted to audio. \
Be friendly and professional. Do not use special characters in your responses.";

const INTRO_TASK: &str = "\
Greet the candidate warmly with this introduction: \
Hi, great to see you and thanks for joining us today. My name is William \
and I am coordinating this interview on behalf of EU Volt. We are currently \
expanding our team and looking for skilled production technicians to support \
our operations. This interview will take around 10 minutes. We will start \
with a few questions about your background and experience, then move into \
some technical scenarios and workplace topics. Feel free to ask if anything \
is unclear along the way. Before we dive in, could you briefly introduce yourself?

After the candidate introduces themselves, call begin_interview with their name.";

const INTERVIEW_SYSTEM_PROMPT: &str = "\
You are an interviewer named William for EU Volt, a sustainable energy storage company \
specializing in advanced battery production. You are interviewing for a Production \
Technician position in Zurich. This role involves operating and monitoring battery \
production lines, maintaining equipment, handling chemicals safely, troubleshooting \
HMI errors, training operators, and ensuring quality standards.

Keep your responses brief since they will be converted to audio. Be friendly but professional. \
Do not use special characters. Ask one question at a time and wait for the response. \
Be encouraging and positive throughout.

PRIORITY QUESTIONS TO COVER (aim for all 10):
1. What part of your previous experience will be most valuable in this Production Technician role?
2. You encounter an error message on the HMI that halts production. How do you diagnose and fix it?
3. Do you have experience handling chemicals? Which chemicals, and what safety protocols did you follow?
4. If you notice a recurring deviation in product quality, what steps would you take?
5. You and another technician disagree on how to troubleshoot an HMI error with production halted. How do you handle it?
6. What do you do if there is a chemical spill in your workplace?
7. If tasked with reducing downtime by 10 percent, what methods or tools would you use?
8. How do you handle stress when production deadlines are tight?
9. What are your salary expectations for this role?
10. Where do you see yourself in five years — technician, specialist, or leadership?

IMPORTANT RULES:
- Keep track of how many questions you have asked
- After asking all 10 questions (or if the candidate asks to stop), call end_conversation with detailed scoring
- You can ask follow-up questions if an answer is unclear, but these count toward your total
- After the final question, thank the candidate, then immediately call end_conversation

SCORING CRITERIA (use when calling end_conversation):

1. Technical Knowledge (30 points): HMI/PLC experience, production line understanding, \
   equipment maintenance, battery/chemical production familiarity.
2. Problem-Solving (25 points): Systematic diagnostics, logical thinking under pressure, \
   process improvement mindset, quality issue resolution.
3. Safety Awareness (20 points): Chemical handling protocols, emergency response, PPE usage, \
   risk assessment.
4. Soft Skills (15 points): Teamwork, stress management, communication, training ability.
5. Cultural Fit (10 points): Alignment with EU Volt values (Courage, Integrity, Collaboration, \
   Innovation), realistic career goals, sustainability commitment, reasonable salary expectations.

RED FLAGS to note: No chemical safety knowledge, no relevant technical experience, \
poor teamwork or conflict resolution, unrealistic salary demands, no interest in sustainability.

SCORING SCALE:
90-100 Exceptional — strongly recommend hire
75-89  Good — recommend with minor reservations
60-74  Average — may require additional training
45-59  Below average — significant gaps
0-44   Poor fit — do not recommend";

const INTERVIEW_TASK: &str = "\
You are now in the interview phase. Begin asking the priority questions. \
Ask one question at a time, wait for the candidate's response, acknowledge it briefly, \
then move to the next question. After all questions are covered or the candidate \
asks to stop, call end_conversation with a complete score breakdown.";

const FAREWELL_TASK: &str = "\
The interview has been completed and scored. Thank the candidate for their time, \
let them know they will hear back from the EU Volt recruitment team soon, \
and say goodbye warmly. Keep it brief — two or three sentences.";

// ---------------------------------------------------------------------------
// Node configs
// ---------------------------------------------------------------------------

fn intro_node() -> NodeConfig {
    NodeConfig::new("intro")
        .with_system_prompt(INTRO_SYSTEM_PROMPT)
        .with_task_message(INTRO_TASK)
        .with_tools(ToolsSchema::new(vec![begin_interview_schema()]))
        .with_respond_immediately(true)
}

fn interview_node() -> NodeConfig {
    NodeConfig::new("interview")
        .with_system_prompt(INTERVIEW_SYSTEM_PROMPT)
        .with_task_message(INTERVIEW_TASK)
        .with_tools(ToolsSchema::new(vec![end_conversation_schema()]))
        .with_context_strategy(ContextStrategy::Append)
        .with_respond_immediately(true)
}

fn farewell_node() -> NodeConfig {
    NodeConfig::new("farewell")
        .with_task_message(FAREWELL_TASK)
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
            let name = parsed["candidate_name"]
                .as_str()
                .map(String::from);

            {
                let mut state = interview.lock().unwrap();
                state.candidate_name = name.clone();
            }

            log::info!(
                "Interview started for candidate: {}",
                name.as_deref().unwrap_or("unnamed")
            );

            let result = json!({
                "status": "interview_started",
                "candidate_name": name,
                "instruction": "Begin asking the priority questions now. Start with question 1.",
            });

            TransitionResult::transition(result.to_string(), "interview")
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

            // Clamp scores to valid ranges
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

            let summary = parsed["summary"]
                .as_str()
                .unwrap_or("")
                .to_string();

            let candidate_name = interview.lock().unwrap().candidate_name.clone();

            // ---- Console output (mirrors the Python version) ----
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

            // ---- Push score to client via Ravi server-message ----
            let score_payload = json!({
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
            });
            push_ravi_msg(&push, score_payload).await;

            let result = json!({
                "status": "interview_scored",
                "total_score": total,
                "instruction": "Thank the candidate warmly and say goodbye.",
            });

            TransitionResult::transition(result.to_string(), "farewell")
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

fn build_flow() -> ConnectionFlow {
    let interview = Arc::new(Mutex::new(InterviewState::default()));
    let context   = Arc::new(Mutex::new(LLMContext::new(None)));
    let registry  = Arc::new(Mutex::new(FunctionRegistry::new()));
    let push_tx: DeferredPush = Arc::new(std::sync::OnceLock::new());

    let mut dhara = DharaManager::new(context.clone(), registry.clone());

    // intro — greet candidate, then begin_interview transitions to interview
    dhara.register_node("intro", intro_node(), vec![
        ("begin_interview", make_begin_interview_handler(interview.clone())),
    ]);

    // interview — the main Q&A phase, end_conversation transitions to farewell
    dhara.register_node("interview", interview_node(), vec![
        ("end_conversation", make_end_conversation_handler(interview.clone(), push_tx.clone())),
    ]);

    // farewell — no Dhara handlers needed
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
    log::info!("Flow: intro -> interview -> farewell");
    log::info!("Tools: begin_interview (Dhara), end_conversation (Dhara)");

    let listener = tokio::net::TcpListener::bind(&addr).await
        .unwrap_or_else(|e| panic!("Failed to bind {}: {}", addr, e));

    axum::serve(listener, app).await.unwrap();
}