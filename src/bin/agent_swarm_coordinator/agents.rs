//! The swarm's peer agents — reached by the coordinator over the bus.
//!
//! Three plain (non-LLM) tool agents wrapping an idle pipeline, and one LLM
//! agent builder used for the `planner` and `composer` roles.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};
use tokio::sync::mpsc;

use rustvani::agents::{BaseAgent, TaskHandler, TaskRequestCtx, TaskStatus};
use rustvani::{
    shared_context, ControlFrame, DataFrame, Frame, FrameDirection, FrameInner, FrameKind,
    LLMAssistantAggregator, LLMContext, OpenAILLMConfig, OpenAILLMHandler, PipelineParams,
    PipelineTask,
};

// ---------------------------------------------------------------------------
// Non-LLM tool agents — idle pipeline + a handler that just computes.
// ---------------------------------------------------------------------------

fn fn_agent(name: &str, handler: TaskHandler) -> Arc<BaseAgent> {
    Arc::new(
        BaseAgent::new(name, PipelineTask::new(vec![], PipelineParams::default()), None, true)
            .on_task("ask", handler),
    )
}

/// A random u64 without pulling in the `rand` crate — v4 UUIDs are CSPRNG-backed.
fn rand_u64() -> u64 {
    uuid::Uuid::new_v4().as_u128() as u64
}

pub fn clock_agent() -> Arc<BaseAgent> {
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

pub fn dice_agent() -> Arc<BaseAgent> {
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

pub fn weather_agent() -> Arc<BaseAgent> {
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
                tokio::time::sleep(Duration::from_millis(300)).await; // mock API latency
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
// LLM agent — drives its own [LLM → aggregator] pipeline behind `on_task`.
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

/// Build an LLM agent with a fixed system prompt, answering `on_task("ask")`
/// requests of shape `{ "prompt": <str> }` with `{ "answer": <str> }`.
pub fn llm_agent(name: &str, system_prompt: &str, api_key: &str) -> Arc<BaseAgent> {
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
