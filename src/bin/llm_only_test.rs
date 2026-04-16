//! LLM-only test — no audio, no VAD, no STT.
//!
//! Pipeline:
//!   SarvamLLM → LLMAssistantAggregator
//!
//! BaseObserver logs all frame movements with timestamps automatically.
//! StartFrame is auto-sent by task.run().
//! LLMContextFrame is queued via push_sender().
//!
//! Run:
//!   SARVAM_API_KEY=your-key cargo run --bin llm_only_test
//!   SARVAM_API_KEY=your-key cargo run --bin llm_only_test -- "your question"

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use rustvani::{
    shared_context, system_clock, DataFrame, Frame, FrameDirection,
    FrameInner, FrameKind, LLMAssistantAggregator, PipelineParams, PipelineTask, SystemFrame,
};
use rustvani::observer::{BaseObserver, FrameProcessed, FramePushed};
use rustvani::services::{SarvamLLMConfig, SarvamLLMHandler};

// ---------------------------------------------------------------------------
// FrameLogger — hooks into every frame via BaseObserver, prints with timestamp
// ---------------------------------------------------------------------------

struct FrameLogger;

#[async_trait]
impl BaseObserver for FrameLogger {
    async fn on_process_frame(&self, event: FrameProcessed) {
        let dir = match event.direction {
            FrameDirection::Downstream => "↓",
            FrameDirection::Upstream   => "↑",
        };
        match &event.frame.inner {
            FrameInner::System(SystemFrame::LLMFullResponseStart) => {
                println!("[{:.3}] PROCESS {} {}  @ {}", event.timestamp, dir, event.frame.name(), event.processor_name);
                print!("[{:.3}] [LLM]   ", event.timestamp);
            }
            FrameInner::Data(DataFrame::LLMText(text)) => {
                print!("{}", text);
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
            FrameInner::System(SystemFrame::LLMFullResponseEnd) => {
                println!(); // close token line
                println!("[{:.3}] PROCESS {} {}  @ {}", event.timestamp, dir, event.frame.name(), event.processor_name);
            }
            _ => {
                println!("[{:.3}] PROCESS {} {}  @ {}", event.timestamp, dir, event.frame.name(), event.processor_name);
            }
        }
    }

    async fn on_push_frame(&self, event: FramePushed) {
        if event.frame.kind() == FrameKind::LLMText {
            return; // printed inline above
        }
        let dir = match event.direction {
            FrameDirection::Downstream => "↓",
            FrameDirection::Upstream   => "↑",
        };
        println!(
            "[{:.3}] PUSH    {} {}  {} → {}",
            event.timestamp, dir, event.frame.name(),
            event.source_name, event.destination_name
        );
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("error"),
    )
    .init();

    let api_key = std::env::var("SARVAM_API_KEY").expect("SARVAM_API_KEY not set");
    let question = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "What is the capital of Kerala?".to_string());

    println!("=== rustvani LLM test ===");
    println!("[user]  {}\n", question);

    let context = shared_context(Some(
        "You are a concise assistant. Answer in one or two sentences.".to_string(),
    ));
    context.lock().unwrap().add_message("user", &question);

    let task = PipelineTask::new(
        vec![
            SarvamLLMHandler::new(SarvamLLMConfig {
                api_key,
                model: "sarvam-m".to_string(),
                ..SarvamLLMConfig::default()
            }).into_processor(),
            LLMAssistantAggregator::new(context.clone()),
        ],
        PipelineParams::default(),
    );

    let push_tx = task.push_sender();
    let ctx_frame = Frame::llm_context(context.clone());

    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        let _ = push_tx.send((ctx_frame, FrameDirection::Downstream)).await;
        tokio::time::sleep(Duration::from_secs(8)).await;
        let _ = push_tx.send((Frame::end(), FrameDirection::Downstream)).await;
    });

    task.run(system_clock(), Some(Arc::new(FrameLogger))).await?;

    println!("\n=== final context ===");
    for msg in &context.lock().unwrap().messages {
        println!("[{}]  {}", msg.role, msg.content);
    }

    Ok(())
}
