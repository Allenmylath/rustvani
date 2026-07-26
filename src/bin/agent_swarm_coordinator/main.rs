//! Router swarm — the coordinator is a `CoordinatorProcessor` that sits INLINE
//! in the pipeline `vec![]`, dispatching to peer agents over the bus from inside
//! `on_process_frame`. No hand-rolled bus plumbing — the framework injects the
//! bus handle.
//!
//! This binary is organised as a folder:
//!   - `main.rs`   — wiring + REPL driver
//!   - `agents.rs` — the peer agents (clock / dice / weather + LLM agents)
//!   - `brain.rs`  — the coordinator's decision logic + prompts
//!
//! Coordinator pipeline:  [ CoordinatorProcessor → LLMAssistantAggregator ]
//!
//! Setup: put `OPENAI_API_KEY=sk-...` in `.env`.
//! Run:   cargo run --bin agent_swarm_coordinator

mod agents;
mod brain;

use std::collections::HashSet;
use std::io::Write as _;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, BufReader};

use rustvani::agents::{AgentRunner, BaseAgent, CoordinatorProcessor, LocalAgentBus};
use rustvani::{
    shared_context, system_clock, ControlFrame, DataFrame, Frame, FrameDirection, FrameInner,
    FrameKind, LLMAssistantAggregator, PipelineParams, PipelineTask,
};

use agents::{clock_agent, dice_agent, llm_agent, weather_agent};
use brain::{coordinate, COMPOSE_SYS, ROUTER_SYS};

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

    // Assemble the swarm.
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

    // Wait until the coordinator agent is running (so its bus handle is injected).
    while coordinator.task_ctx().is_none() {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // REPL: feed the coordinator agent like any LLM pipeline — user msg + context frame.
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
