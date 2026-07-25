//! Mixed swarm — agents are NOT all LLMs. An agent's work is whatever its
//! `on_task` handler does; only some of them happen to call an LLM.
//!
//!   ┌──────────────┐  dispatch (bus, same for every agent)
//!   │ coordinator  ├───────────────┬──────────────┬───────────────┐
//!   └──────────────┘               ▼              ▼               ▼
//!                            ┌───────────┐  ┌───────────┐  ┌──────────────┐
//!                            │  clock    │  │   dice    │  │   weather    │
//!                            │ (plain fn)│  │ (plain fn)│  │ (mock API)   │
//!                            └───────────┘  └───────────┘  └──────────────┘
//!                                          then ▼
//!                                     ┌──────────────┐
//!                                     │   narrator   │  (LLM — the only one)
//!                                     └──────────────┘
//!
//! The plain agents wrap an *idle* pipeline (empty — it just keeps the agent
//! alive) and do all their work synchronously in the handler. The LLM agent
//! drives its own `[LLM → aggregator]` pipeline. To the coordinator they are
//! indistinguishable: every one is a `dispatch(... "ask" ...)` + `await`.
//!
//! Setup: put `OPENAI_API_KEY=sk-...` in `.env` (only the narrator needs it).
//! Run:   cargo run --bin agent_swarm_mixed
//! Type a place/topic; `/quit` (or Ctrl-D) exits.

use std::collections::HashSet;
use std::io::Write as _;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;

use rustvani::agents::{
    AgentRunner, BaseAgent, LocalAgentBus, TaskHandler, TaskRequestCtx, TaskResult, TaskStatus,
};
use rustvani::{
    shared_context, system_clock, ControlFrame, DataFrame, Frame, FrameDirection, FrameInner,
    FrameKind, LLMAssistantAggregator, LLMContext, OpenAILLMConfig, OpenAILLMHandler,
    PipelineParams, PipelineTask,
};

const ASK_TIMEOUT: Duration = Duration::from_secs(90);

// ---------------------------------------------------------------------------
// Plain (non-LLM) agents — an idle pipeline + a handler that just computes.
// ---------------------------------------------------------------------------

/// Wrap a `TaskHandler` as an agent with an idle pipeline. No LLM, no frames —
/// the empty pipeline only keeps the agent alive so it can receive dispatches.
fn fn_agent(name: &str, handler: TaskHandler) -> Arc<BaseAgent> {
    Arc::new(
        BaseAgent::new(
            name,
            PipelineTask::new(vec![], PipelineParams::default()),
            None,
            true,
        )
        .on_task("ask", handler),
    )
}

/// A random u64 without pulling in the `rand` crate — v4 UUIDs are CSPRNG-backed.
fn rand_u64() -> u64 {
    uuid::Uuid::new_v4().as_u128() as u64
}

fn clock_agent() -> Arc<BaseAgent> {
    fn_agent(
        "clock",
        Arc::new(|ctx: TaskRequestCtx| {
            Box::pin(async move {
                let now = chrono::Utc::now();
                ctx.complete(
                    TaskStatus::Completed,
                    Some(json!({ "time_utc": now.to_rfc3339(), "unix": now.timestamp() })),
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
                let roll = rand_u64() % sides + 1;
                ctx.complete(TaskStatus::Completed, Some(json!({ "roll": roll, "sides": sides })))
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
                // Simulate a real API round-trip.
                tokio::time::sleep(Duration::from_millis(300)).await;
                const CONDITIONS: [&str; 5] = ["sunny", "cloudy", "rainy", "windy", "foggy"];
                let condition = CONDITIONS[(rand_u64() as usize) % CONDITIONS.len()];
                let temp_c = 12 + (rand_u64() % 20); // 12–31 °C
                ctx.complete(
                    TaskStatus::Completed,
                    Some(json!({ "city": city, "condition": condition, "temp_c": temp_c })),
                )
                .await;
            })
        }),
    )
}

// ---------------------------------------------------------------------------
// LLM agent — drives its own [LLM → aggregator] pipeline (as before).
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
    let aggregator = LLMAssistantAggregator::new(context.clone());
    let task = PipelineTask::new(vec![llm, aggregator], PipelineParams::default());

    let (reply_tx, reply_rx) = mpsc::channel::<String>(8);
    let mut filter = HashSet::new();
    filter.insert(FrameKind::LLMText);
    filter.insert(FrameKind::LLMFullResponseStart);
    filter.insert(FrameKind::LLMFullResponseEnd);
    task.set_downstream_filter(filter);

    let buffer: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    task.add_on_frame_reached_downstream(move |frame| {
        let buffer = buffer.clone();
        let reply_tx = reply_tx.clone();
        Box::pin(async move {
            match frame.inner {
                FrameInner::Control(ControlFrame::LLMFullResponseStart) => {
                    buffer.lock().unwrap().clear();
                }
                FrameInner::Data(DataFrame::LLMText(text)) => {
                    buffer.lock().unwrap().push_str(&text);
                }
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

/// The raw JSON response of a dispatched task (or an error object).
fn response_of(res: rustvani::Result<TaskResult>) -> Value {
    match res {
        Ok(TaskResult { response, .. }) => response.unwrap_or(Value::Null),
        Err(e) => json!({ "error": e.to_string() }),
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

    println!("=== rustvani mixed swarm (clock + dice + weather-mock + LLM narrator) ===");
    println!("Type a place or topic. `/quit` or Ctrl-D to exit.\n");

    let coordinator = Arc::new(BaseAgent::new(
        "coordinator",
        PipelineTask::new(vec![], PipelineParams::default()),
        None,
        true,
    ));

    let bus = Arc::new(LocalAgentBus::new());
    let runner = Arc::new(AgentRunner::new("swarm", bus, system_clock()));
    runner.add_agent(coordinator.clone()).await?;
    runner.add_agent(clock_agent()).await?;
    runner.add_agent(dice_agent()).await?;
    runner.add_agent(weather_agent()).await?;
    runner
        .add_agent(llm_agent(
            "narrator",
            "You write a short, upbeat one-paragraph status blurb. Weave the \
             given facts (time, a lucky number, the weather) into it naturally. \
             2-3 sentences, no lists.",
            &api_key,
        ))
        .await?;

    let runner_handle = {
        let runner = runner.clone();
        tokio::spawn(async move {
            if let Err(e) = runner.run().await {
                eprintln!("runner error: {e}");
            }
        })
    };

    let task_ctx = loop {
        if let Some(ctx) = coordinator.task_ctx() {
            break ctx;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    };

    print!("place> ");
    std::io::stdout().flush()?;
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    while let Some(line) = lines.next_line().await? {
        let place = line.trim().to_string();
        if place.is_empty() {
            print!("place> ");
            std::io::stdout().flush()?;
            continue;
        }
        if place == "/quit" || place == "/exit" {
            break;
        }

        // --- Fan out to the three non-LLM agents in parallel ---
        let clock_h = task_ctx.dispatch("coordinator", "clock", Some("ask".into()), None).await;
        let dice_h = task_ctx
            .dispatch("coordinator", "dice", Some("ask".into()), Some(json!({ "sides": 20 })))
            .await;
        let weather_h = task_ctx
            .dispatch("coordinator", "weather", Some("ask".into()), Some(json!({ "city": place })))
            .await;

        let (clock_h, dice_h, weather_h) = match (clock_h, dice_h, weather_h) {
            (Ok(a), Ok(b), Ok(c)) => (a, b, c),
            _ => {
                eprintln!("dispatch failed");
                print!("\nplace> ");
                std::io::stdout().flush()?;
                continue;
            }
        };
        let (t, d, w) = tokio::join!(
            clock_h.await_completion(Some(ASK_TIMEOUT)),
            dice_h.await_completion(Some(ASK_TIMEOUT)),
            weather_h.await_completion(Some(ASK_TIMEOUT)),
        );
        let time = response_of(t);
        let dice = response_of(d);
        let weather = response_of(w);

        println!("\n  clock   → {time}");
        println!("  dice    → {dice}");
        println!("  weather → {weather}");

        // --- Hand the facts to the LLM narrator ---
        let facts = format!(
            "Place: {place}\nTime (UTC): {}\nLucky number: {}\nWeather: {} at {}°C",
            time.get("time_utc").and_then(Value::as_str).unwrap_or("?"),
            dice.get("roll").map(|v| v.to_string()).unwrap_or_else(|| "?".into()),
            weather.get("condition").and_then(Value::as_str).unwrap_or("?"),
            weather.get("temp_c").map(|v| v.to_string()).unwrap_or_else(|| "?".into()),
        );
        let blurb = match task_ctx
            .dispatch("coordinator", "narrator", Some("ask".into()), Some(json!({ "prompt": facts })))
            .await
        {
            Ok(h) => response_of(h.await_completion(Some(ASK_TIMEOUT)).await)
                .get("answer")
                .and_then(Value::as_str)
                .unwrap_or("[no narration]")
                .to_string(),
            Err(e) => format!("[dispatch error: {e}]"),
        };

        println!("\n📣 {blurb}\n");
        print!("place> ");
        std::io::stdout().flush()?;
    }

    println!("\nshutting down…");
    runner.end(None).await;
    let _ = tokio::time::timeout(Duration::from_secs(10), runner_handle).await;
    println!("=== swarm ended ===");
    Ok(())
}
