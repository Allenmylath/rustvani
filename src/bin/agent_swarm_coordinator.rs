//! Router swarm, take 2 — the coordinator is now a `CoordinatorProcessor` that
//! sits INLINE in the pipeline `vec![]`, dispatching to peer agents over the
//! bus from inside `on_process_frame`. No hand-rolled `LlmBrain`, no post-setup
//! `task_ctx` polling — the framework injects the bus handle.
//!
//! Coordinator pipeline:  [ CoordinatorProcessor → LLMAssistantAggregator ]
//!
//! Inside the processor's closure it: asks a `planner` LLM which agents to call,
//! dispatches the chosen ones (clock / dice / weather), then asks a `composer`
//! LLM to write the answer — all via `cx.call(...)`. This is the same logic as
//! `agent_swarm_router`, but living in the LLM's slot in a pipeline instead of a
//! stdin loop. Swap the REPL driver for transport→STT / TTS and it's voiced.
//!
//! Setup: put `OPENAI_API_KEY=sk-...` in `.env`.
//! Run:   cargo run --bin agent_swarm_coordinator

use std::collections::HashSet;
use std::io::Write as _;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Map, Value};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;

use rustvani::agents::{
    AgentRunner, BaseAgent, CoordinatorCtx, CoordinatorProcessor, LocalAgentBus, TaskHandler,
    TaskRequestCtx, TaskResult, TaskStatus,
};
use rustvani::context::Message;
use rustvani::{
    shared_context, system_clock, ControlFrame, DataFrame, Frame, FrameDirection, FrameInner,
    FrameKind, LLMAssistantAggregator, LLMContext, OpenAILLMConfig, OpenAILLMHandler,
    PipelineParams, PipelineTask,
};

const KNOWN_AGENTS: [&str; 3] = ["clock", "dice", "weather"];

const ROUTER_SYS: &str = "\
You are a coordinator that can call helper agents. Available agents:
- \"clock\": current UTC time. No args.
- \"dice\": roll a random number. args: {\"sides\": <int, default 20>}
- \"weather\": (mock) current weather for a city. args: {\"city\": <string>}
Given the user's message, decide which helpers (if any) are needed.
Reply with ONLY a JSON object, no prose, no code fences:
{\"calls\": [{\"agent\": \"<name>\", \"args\": {..}}], \"reply\": \"<direct answer if no calls needed, else empty>\"}
For greetings or smalltalk, return an empty calls list and answer in \"reply\".";

const COMPOSE_SYS: &str = "\
You are a helpful assistant. You asked helper agents for data and got their
JSON results. Answer the user's original question naturally in 1-3 sentences,
using the results. Do not mention agents, tools, or JSON.";

// ---------------------------------------------------------------------------
// Non-LLM agents
// ---------------------------------------------------------------------------

fn fn_agent(name: &str, handler: TaskHandler) -> Arc<BaseAgent> {
    Arc::new(
        BaseAgent::new(name, PipelineTask::new(vec![], PipelineParams::default()), None, true)
            .on_task("ask", handler),
    )
}

fn rand_u64() -> u64 {
    uuid::Uuid::new_v4().as_u128() as u64
}

fn clock_agent() -> Arc<BaseAgent> {
    fn_agent(
        "clock",
        Arc::new(|ctx: TaskRequestCtx| {
            Box::pin(async move {
                ctx.complete(
                    TaskStatus::Completed,
                    Some(json!({ "time_utc": chrono::Utc::now().to_rfc3339() })),
                )
                .await;
            })
        }),
    )
}

fn dice_agent() -> Arc<BaseAgent> {
    fn_agent(
        "dice",
        Arc::new(|ctx: TaskRequestCtx| {
            Box::pin(async move {
                let sides = ctx
                    .payload
                    .as_ref()
                    .and_then(|p| p.get("sides"))
                    .and_then(Value::as_u64)
                    .unwrap_or(20)
                    .max(1);
                ctx.complete(
                    TaskStatus::Completed,
                    Some(json!({ "roll": rand_u64() % sides + 1, "sides": sides })),
                )
                .await;
            })
        }),
    )
}

fn weather_agent() -> Arc<BaseAgent> {
    fn_agent(
        "weather",
        Arc::new(|ctx: TaskRequestCtx| {
            Box::pin(async move {
                let city = ctx
                    .payload
                    .as_ref()
                    .and_then(|p| p.get("city"))
                    .and_then(Value::as_str)
                    .unwrap_or("somewhere")
                    .to_string();
                tokio::time::sleep(Duration::from_millis(300)).await;
                const C: [&str; 5] = ["sunny", "cloudy", "rainy", "windy", "foggy"];
                ctx.complete(
                    TaskStatus::Completed,
                    Some(json!({
                        "city": city,
                        "condition": C[(rand_u64() as usize) % C.len()],
                        "temp_c": 12 + rand_u64() % 20,
                    })),
                )
                .await;
            })
        }),
    )
}

// ---------------------------------------------------------------------------
// LLM peer agents (planner + composer) — driven by their own pipeline.
// ---------------------------------------------------------------------------

struct LlmWorker {
    context: Arc<Mutex<LLMContext>>,
    push_tx: mpsc::Sender<(Frame, FrameDirection)>,
    reply_rx: tokio::sync::Mutex<mpsc::Receiver<String>>,
    system_prompt: String,
}

impl LlmWorker {
    async fn ask(&self, user: &str) -> String {
        let mut rx = self.reply_rx.lock().await;
        {
            let mut ctx = self.context.lock().unwrap();
            *ctx = LLMContext::new(Some(self.system_prompt.clone()));
            ctx.add_user_message(user);
        }
        if self
            .push_tx
            .send((Frame::llm_context(self.context.clone()), FrameDirection::Downstream))
            .await
            .is_err()
        {
            return String::new();
        }
        rx.recv().await.unwrap_or_default()
    }
}

fn llm_agent(name: &str, system_prompt: &str, api_key: &str) -> Arc<BaseAgent> {
    let context = shared_context(Some(system_prompt.to_string()));
    let llm = OpenAILLMHandler::new(OpenAILLMConfig {
        api_key: api_key.to_string(),
        ..OpenAILLMConfig::default()
    })
    .into_processor();
    let task = PipelineTask::new(
        vec![llm, LLMAssistantAggregator::new(context.clone())],
        PipelineParams::default(),
    );

    let (reply_tx, reply_rx) = mpsc::channel::<String>(8);
    task.set_downstream_filter(HashSet::from([
        FrameKind::LLMText,
        FrameKind::LLMFullResponseStart,
        FrameKind::LLMFullResponseEnd,
    ]));
    let buffer: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    task.add_on_frame_reached_downstream(move |frame| {
        let buffer = buffer.clone();
        let reply_tx = reply_tx.clone();
        Box::pin(async move {
            match frame.inner {
                FrameInner::Control(ControlFrame::LLMFullResponseStart) => buffer.lock().unwrap().clear(),
                FrameInner::Data(DataFrame::LLMText(t)) => buffer.lock().unwrap().push_str(&t),
                FrameInner::Control(ControlFrame::LLMFullResponseEnd) => {
                    let reply = buffer.lock().unwrap().trim().to_string();
                    let _ = reply_tx.send(reply).await;
                }
                _ => {}
            }
        })
    });

    let worker = Arc::new(LlmWorker {
        context,
        push_tx: task.push_sender(),
        reply_rx: tokio::sync::Mutex::new(reply_rx),
        system_prompt: system_prompt.to_string(),
    });
    let handler: TaskHandler = {
        let worker = worker.clone();
        Arc::new(move |ctx: TaskRequestCtx| {
            let worker = worker.clone();
            Box::pin(async move {
                let prompt = ctx
                    .payload
                    .as_ref()
                    .and_then(|p| p.get("prompt"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let answer = worker.ask(&prompt).await;
                ctx.complete(TaskStatus::Completed, Some(json!({ "answer": answer })))
                    .await;
            })
        })
    };
    Arc::new(BaseAgent::new(name, task, None, true).on_task("ask", handler))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn answer_str(res: rustvani::Result<TaskResult>) -> String {
    res.ok()
        .and_then(|r| r.response)
        .and_then(|v| v.get("answer").and_then(Value::as_str).map(str::to_string))
        .unwrap_or_default()
}

fn extract_json(s: &str) -> Option<Value> {
    let start = s.find('{')?;
    let end = s.rfind('}')?;
    (end >= start).then(|| serde_json::from_str(&s[start..=end]).ok()).flatten()
}

fn last_user(context: &Arc<Mutex<LLMContext>>) -> String {
    context
        .lock()
        .unwrap()
        .messages
        .iter()
        .rev()
        .find_map(|m| match m {
            Message::User { content } => Some(content.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

/// The coordinator's brain — runs INSIDE the pipeline processor, dispatching
/// to peer agents via `cx.call(...)`.
async fn coordinate(cx: CoordinatorCtx, context: Arc<Mutex<LLMContext>>) -> String {
    let question = last_user(&context);

    // 1. Ask the planner which agents to call.
    let plan = match cx.call("planner", "ask", json!({ "prompt": question })).await {
        Ok(r) => extract_json(&answer_str(Ok(r))),
        Err(e) => return format!("[planner error: {e}]"),
    };

    let chosen: Vec<(String, Value)> = plan
        .as_ref()
        .and_then(|p| p.get("calls"))
        .and_then(Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .filter_map(|c| {
                    let name = c.get("agent").and_then(Value::as_str)?;
                    KNOWN_AGENTS
                        .contains(&name)
                        .then(|| (name.to_string(), c.get("args").cloned().unwrap_or(json!({}))))
                })
                .collect()
        })
        .unwrap_or_default();

    if chosen.is_empty() {
        return plan
            .and_then(|p| p.get("reply").and_then(Value::as_str).map(str::to_string))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "Hi! How can I help?".to_string());
    }

    println!("   [coordinator dispatched: {:?}]", chosen.iter().map(|(n, _)| n).collect::<Vec<_>>());

    // 2. Dispatch the chosen tools.
    let mut results = Map::new();
    for (agent, args) in chosen {
        if let Ok(r) = cx.call(&agent, "ask", args).await {
            results.insert(agent, r.response.unwrap_or(Value::Null));
        }
    }

    // 3. Ask the composer to write the final answer.
    let facts = format!(
        "The user asked: {question}\n\nHelper results (JSON):\n{}\n\nAnswer the user.",
        Value::Object(results)
    );
    match cx.call("composer", "ask", json!({ "prompt": facts })).await {
        Ok(r) => answer_str(Ok(r)),
        Err(e) => format!("[composer error: {e}]"),
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");

    println!("=== rustvani coordinator-processor swarm ===");
    println!("The coordinator sits IN the pipeline and dispatches to peers.");
    println!("Ask anything. `/quit` or Ctrl-D to exit.\n");

    // Coordinator pipeline: CoordinatorProcessor (in the LLM slot) → aggregator.
    let coord_context = shared_context(None);
    let coord_proc =
        CoordinatorProcessor::new("coordinator", |cx, ctx| Box::pin(coordinate(cx, ctx)))
            .into_processor();
    let coord_task = PipelineTask::new(
        vec![coord_proc, LLMAssistantAggregator::new(coord_context.clone())],
        PipelineParams::default(),
    );
    // Print the coordinator's answer at the pipeline tail.
    coord_task.set_downstream_filter(HashSet::from([
        FrameKind::LLMText,
        FrameKind::LLMFullResponseStart,
        FrameKind::LLMFullResponseEnd,
    ]));
    let reply_buf: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    coord_task.add_on_frame_reached_downstream(move |frame| {
        let reply_buf = reply_buf.clone();
        Box::pin(async move {
            match frame.inner {
                FrameInner::Control(ControlFrame::LLMFullResponseStart) => reply_buf.lock().unwrap().clear(),
                FrameInner::Data(DataFrame::LLMText(t)) => reply_buf.lock().unwrap().push_str(&t),
                FrameInner::Control(ControlFrame::LLMFullResponseEnd) => {
                    let reply = reply_buf.lock().unwrap().trim().to_string();
                    println!("\n💬 {reply}\n");
                    print!("ask> ");
                    let _ = std::io::stdout().flush();
                }
                _ => {}
            }
        })
    });
    let coord_push = coord_task.push_sender();
    let coordinator = Arc::new(BaseAgent::new("coordinator", coord_task, None, true));

    let bus = Arc::new(LocalAgentBus::new());
    let runner = Arc::new(AgentRunner::new("swarm", bus, system_clock()));
    runner.add_agent(coordinator.clone()).await?;
    runner.add_agent(clock_agent()).await?;
    runner.add_agent(dice_agent()).await?;
    runner.add_agent(weather_agent()).await?;
    runner.add_agent(llm_agent("planner", ROUTER_SYS, &api_key)).await?;
    runner.add_agent(llm_agent("composer", COMPOSE_SYS, &api_key)).await?;

    let runner_handle = {
        let runner = runner.clone();
        tokio::spawn(async move {
            if let Err(e) = runner.run().await {
                eprintln!("runner error: {e}");
            }
        })
    };

    // Wait until the coordinator agent is running (so bus handle is injected).
    while coordinator.task_ctx().is_none() {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    print!("ask> ");
    std::io::stdout().flush()?;
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    while let Some(line) = lines.next_line().await? {
        let q = line.trim().to_string();
        if q.is_empty() {
            print!("ask> ");
            std::io::stdout().flush()?;
            continue;
        }
        if q == "/quit" || q == "/exit" {
            break;
        }
        // Feed the coordinator agent like any LLM pipeline: user msg + context frame.
        coord_context.lock().unwrap().add_user_message(&q);
        if coord_push
            .send((Frame::llm_context(coord_context.clone()), FrameDirection::Downstream))
            .await
            .is_err()
        {
            break;
        }
    }

    println!("\nshutting down…");
    runner.end(None).await;
    let _ = tokio::time::timeout(Duration::from_secs(10), runner_handle).await;
    println!("=== swarm ended ===");
    Ok(())
}
