//! Multi-agent text demo using OpenAI.
//!
//! Two agents run under a single AgentRunner, each with its own LLM pipeline:
//!   - "coordinator" — answers the user's question
//!   - "critic"      — provides a counter-perspective / nuance
//!
//! This demonstrates:
//!   - AgentRunner managing multiple BaseAgents
//!   - Each agent owning its own PipelineTask + OpenAI LLM
//!   - LocalAgentBus for inter-agent messaging
//!   - AgentRegistry tracking ready agents
//!
//! Setup:
//!   Add to `.env` in project root:
//!     OPENAI_API_KEY=sk-...
//!
//! Run:
//!   cargo run --bin agent_text_demo
//!   cargo run --bin agent_text_demo -- "Why is Rust fast?"

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rustvani::agents::*;
use rustvani::{
    system_clock, ControlFrame, DataFrame, Frame, FrameDirection, FrameInner, FrameKind,
    LLMAssistantAggregator, OpenAILLMConfig, OpenAILLMHandler, PipelineParams, PipelineTask,
};
use rustvani::context::shared_context;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn"),
    )
    .init();

    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let question = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "What is Rust's ownership model?".to_string());

    println!("=== rustvani multi-agent text demo ===\n");
    println!("Question: {}\n", question);

    // ---- Build coordinator agent ----
    let coord_task = build_pipeline(
        "coordinator",
        &api_key,
        "You are a concise technical explainer. Answer in 2-3 sentences.",
        &question,
    );
    attach_printer(&coord_task, "coordinator");
    let coordinator = Arc::new(BaseAgent::new("coordinator", coord_task, None, true));

    // ---- Build critic agent ----
    let critic_task = build_pipeline(
        "critic",
        &api_key,
        "You are a devil's advocate. Give one counter-argument or nuance to the explanation. One sentence.",
        &format!("The question was: {}", question),
    );
    attach_printer(&critic_task, "critic");
    let critic = Arc::new(BaseAgent::new("critic", critic_task, None, false));

    // ---- Set up runner ----
    let bus = Arc::new(LocalAgentBus::new());
    let runner = AgentRunner::new("demo-runner", bus.clone(), system_clock());

    runner.add_agent(coordinator).await?;
    runner.add_agent(critic).await?;

    // Run with timeout
    let result = tokio::time::timeout(Duration::from_secs(60), runner.run()).await;

    match result {
        Ok(Ok(())) => {
            println!("\n\n=== demo complete ===");
        }
        Ok(Err(e)) => {
            println!("\n\n=== runner error: {} ===", e);
        }
        Err(_) => {
            println!("\n\n=== demo timed out ===");
        }
    }

    // Show registry state
    let locals = runner.registry().local_agents().await;
    let remotes = runner.registry().remote_agents().await;
    println!("\nRegistry — local: {:?}, remote: {:?}", locals, remotes);

    Ok(())
}

// ---------------------------------------------------------------------------
// Build a PipelineTask with an OpenAI LLM + assistant aggregator
// ---------------------------------------------------------------------------

fn build_pipeline(
    _name: &str,
    api_key: &str,
    system_prompt: &str,
    user_message: &str,
) -> PipelineTask {
    let context = shared_context(Some(system_prompt.to_string()));
    context.lock().unwrap().add_user_message(user_message);

    let llm = OpenAILLMHandler::new(OpenAILLMConfig {
        api_key: api_key.to_string(),
        ..OpenAILLMConfig::default()
    })
    .into_processor();

    let aggregator = LLMAssistantAggregator::new(context.clone());

    let task = PipelineTask::new(vec![llm, aggregator], PipelineParams::default());

    // Inject context frame after pipeline starts, then end after 15s
    let push_tx = task.push_sender();
    let ctx_frame = Frame::llm_context(context.clone());
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        let _ = push_tx.send((ctx_frame, FrameDirection::Downstream)).await;
        tokio::time::sleep(Duration::from_secs(15)).await;
        let _ = push_tx.send((Frame::end(), FrameDirection::Downstream)).await;
    });

    task
}

// ---------------------------------------------------------------------------
// Attach a printer callback that buffers each agent's response
// and prints it atomically when complete. This avoids interleaving.
// ---------------------------------------------------------------------------

fn attach_printer(task: &PipelineTask, label: &str) {
    let mut filter = HashSet::new();
    filter.insert(FrameKind::LLMText);
    filter.insert(FrameKind::LLMFullResponseStart);
    filter.insert(FrameKind::LLMFullResponseEnd);
    task.set_downstream_filter(filter);

    let label = label.to_string();
    let buffer: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));

    task.add_on_frame_reached_downstream(move |frame| {
        let label = label.clone();
        let buffer = buffer.clone();
        Box::pin(async move {
            match frame.inner {
                FrameInner::Control(ControlFrame::LLMFullResponseStart) => {
                    buffer.lock().unwrap().clear();
                }
                FrameInner::Data(DataFrame::LLMText(text)) => {
                    buffer.lock().unwrap().push_str(&text);
                }
                FrameInner::Control(ControlFrame::LLMFullResponseEnd) => {
                    let text = buffer.lock().unwrap().trim().to_string();
                    if !text.is_empty() {
                        println!("\n[{}] {}\n", label, text);
                    }
                }
                _ => {}
            }
        })
    });
}
