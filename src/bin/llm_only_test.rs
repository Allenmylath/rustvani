//! LLM-only test — no audio, no VAD, no STT.
//!
//! Pipeline:
//!   SarvamLLM → LLMAssistantAggregator
//!
//! Timestamps logged at:
//!   - StartFrame / EndFrame
//!   - LLMFullResponseStart  (first token — may be inside <think>)
//!   - </think> close tag    (real reply begins here)
//!   - LLMFullResponseEnd
//!
//! Run:
//!   SARVAM_API_KEY=your-key cargo run --bin llm_only_test
//!   SARVAM_API_KEY=your-key cargo run --bin llm_only_test -- "your question"

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use rustvani::{
    shared_context, system_clock, ControlFrame, DataFrame, Frame, FrameDirection,
    FrameInner, FrameKind, LLMAssistantAggregator, PipelineParams, PipelineTask, SystemFrame,
};
use rustvani::observer::{BaseObserver, FrameProcessed, FramePushed};
use rustvani::services::{SarvamLLMConfig, SarvamLLMHandler};

// ---------------------------------------------------------------------------
// FrameLogger
// ---------------------------------------------------------------------------

struct FrameLogger {
    /// true while we are still inside the <think>...</think> block
    in_think: Mutex<bool>,
    /// rolling buffer for cross-chunk </think> detection
    think_buf: Mutex<String>,
}

impl FrameLogger {
    fn new() -> Self {
        Self {
            in_think: Mutex::new(false),
            think_buf: Mutex::new(String::new()),
        }
    }

    /// Called on each LLMText token.
    /// Returns true if this token contained the </think> boundary.
    fn handle_text_token(&self, text: &str, timestamp: f64) -> bool {
        let mut in_think = self.in_think.lock().unwrap();

        if !*in_think {
            // Already past think block — print normally
            print!("{}", text);
            return false;
        }

        // Still in think mode — buffer and scan
        let mut buf = self.think_buf.lock().unwrap();
        buf.push_str(text);

        if let Some(pos) = buf.find("</think>") {
            // Print everything up to and including </think>
            let close_end = pos + "</think>".len();
            print!("{}", &buf[..close_end]);
            println!();

            // Timestamp the real reply start
            println!(
                "[{:.3}] [LLM]   </think> — real reply starts now",
                timestamp
            );
            print!("[{:.3}] [LLM]   ", timestamp);

            // Print whatever came after </think> in this chunk
            let remainder = &buf[close_end..];
            if !remainder.is_empty() {
                print!("{}", remainder);
            }

            *in_think = false;
            return true;
        }

        // Still inside think, no close tag yet — print what we have
        print!("{}", text);
        false
    }
}

#[async_trait]
impl BaseObserver for FrameLogger {
    async fn on_process_frame(&self, event: FrameProcessed) {
        let dir = match event.direction {
            FrameDirection::Downstream => "↓",
            FrameDirection::Upstream   => "↑",
        };

        // LLM text frames propagate through every downstream processor.
        // Gate to the first receiver so we don't print/buffer 3× per token.
        let is_first_receiver = event.processor_name == "LLMAssistantAggregator";

        match &event.frame.inner {
            FrameInner::Control(ControlFrame::LLMFullResponseStart) => {
                println!(
                    "[{:.3}] PROCESS {} {}  @ {}",
                    event.timestamp, dir, event.frame.name(), event.processor_name
                );
                if is_first_receiver {
                    // Reset think-detection state for this response
                    *self.in_think.lock().unwrap() = true;
                    self.think_buf.lock().unwrap().clear();
                    print!("[{:.3}] [LLM]   ", event.timestamp);
                }
            }

            FrameInner::Data(DataFrame::LLMText(text)) => {
                if is_first_receiver {
                    self.handle_text_token(text, event.timestamp);
                    use std::io::Write;
                    std::io::stdout().flush().ok();
                }
            }

            FrameInner::Control(ControlFrame::LLMFullResponseEnd) => {
                if is_first_receiver {
                    println!(); // close the token line
                }
                println!(
                    "[{:.3}] PROCESS {} {}  @ {}",
                    event.timestamp, dir, event.frame.name(), event.processor_name
                );
            }

            _ => {
                println!(
                    "[{:.3}] PROCESS {} {}  @ {}",
                    event.timestamp, dir, event.frame.name(), event.processor_name
                );
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
                model: "sarvam-30b".to_string(),
                // reasoning_effort: None → non-think mode (default).
                // temperature: 0.2 → recommended for non-think (default).
                // Both are already set correctly in SarvamLLMConfig::default().
                ..SarvamLLMConfig::default()
            }).into_processor(),
            LLMAssistantAggregator::new(context.clone()),
        ],
        PipelineParams::default(),
    );

    let push_tx = task.push_sender();
    let ctx_frame = Frame::llm_context(context.clone());

    // No artificial delay — the channel buffers the frame and the pipeline
    // picks it up as soon as run() initialises its internal machinery.
    tokio::spawn(async move {
        let _ = push_tx.send((ctx_frame, FrameDirection::Downstream)).await;
        tokio::time::sleep(Duration::from_secs(30)).await;
        let _ = push_tx.send((Frame::end(), FrameDirection::Downstream)).await;
    });

    task.run(system_clock(), Some(Arc::new(FrameLogger::new()))).await?;

    println!("\n=== final context ===");
    for msg in &context.lock().unwrap().messages {
        println!("[{}]  {}", msg.role, msg.content);
    }

    Ok(())
}