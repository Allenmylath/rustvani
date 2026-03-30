//! Echo pipeline demo — fake STT → LLM → TTS processors.
//!
//! Run with:  cargo run --example echo_pipeline
//!
//! What happens:
//!   1. Pipeline starts (StartFrame flows through all processors)
//!   2. Simulated audio input arrives every 500ms (fake STT → TextFrame)
//!   3. LLM echoes the text back with a prefix
//!   4. TTS "speaks" the text (prints it)
//!   5. After 3 rounds, EndTask shuts everything down gracefully

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use pipecat_core::{
    clock::system_clock,
    frames::FrameDirection,
    error::Result,
    frames::{DataFrame, DataFrameData, Frame, FrameInner, FrameKind, StartFrameData, SystemFrame},
    pipeline::{FinishReason, PipelineParams, PipelineTask},
    frames::{FrameHandler, FrameProcessor},
};

// ---------------------------------------------------------------------------
// A simple text frame we carry inside DataFrame payload (JSON-encoded string)
// ---------------------------------------------------------------------------

fn text_frame(text: &str) -> Frame {
    Frame::data(text.as_bytes().to_vec())
}

fn frame_text(frame: &Frame) -> Option<String> {
    if let FrameInner::Data(DataFrame::Data(DataFrameData { content, .. })) = &frame.inner {
        String::from_utf8(content.clone()).ok()
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// STT processor — receives raw audio bytes, emits a "transcription" text frame
// ---------------------------------------------------------------------------

struct FakeSTT;

#[async_trait]
impl FrameHandler for FakeSTT {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match &frame.inner {
            FrameInner::System(SystemFrame::Start(_)) => {
                println!("[STT ] Pipeline started — ready for audio");
                processor.push_frame(frame, direction).await?;
            }
            FrameInner::Data(DataFrame::Data(d)) if d.metadata.contains_key("audio") => {
                let audio_len = d.content.len();
                let transcription = format!(
                    "User said: \"{}\"",
                    String::from_utf8_lossy(&d.content)
                );
                println!("[STT ] Received {}B audio → transcribed: {}", audio_len, transcription);
                processor.push_frame(text_frame(&transcription), direction).await?;
            }
            _ => {
                processor.push_frame(frame, direction).await?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LLM processor — receives text, generates a response
// ---------------------------------------------------------------------------

struct FakeLLM;

#[async_trait]
impl FrameHandler for FakeLLM {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match &frame.inner {
            FrameInner::System(SystemFrame::Start(_)) => {
                println!("[LLM ] Pipeline started — model loaded");
                processor.push_frame(frame, direction).await?;
            }
            FrameInner::Data(_) => {
                if let Some(text) = frame_text(&frame) {
                    // Simulate a small thinking delay
                    tokio::time::sleep(Duration::from_millis(80)).await;
                    let response = format!("Bot response to: \"{}\" → I heard you!", text);
                    println!("[LLM ] Generating response: {}", response);
                    processor.push_frame(text_frame(&response), direction).await?;
                } else {
                    processor.push_frame(frame, direction).await?;
                }
            }
            _ => {
                processor.push_frame(frame, direction).await?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TTS processor — receives text, "speaks" it
// ---------------------------------------------------------------------------

struct FakeTTS;

#[async_trait]
impl FrameHandler for FakeTTS {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match &frame.inner {
            FrameInner::System(SystemFrame::Start(_)) => {
                println!("[TTS ] Pipeline started — voice ready");
                processor.push_frame(frame, direction).await?;
            }
            FrameInner::Data(_) => {
                if let Some(text) = frame_text(&frame) {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    println!("[TTS ] 🔊 Speaking: {}", text);
                    // Emit a BotSpeaking frame upstream to reset idle timer
                    processor.push_frame(Frame::bot_speaking(), FrameDirection::Upstream).await?;
                } else {
                    processor.push_frame(frame, direction).await?;
                }
            }
            _ => {
                processor.push_frame(frame, direction).await?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    env_logger::init();

    println!("=== pipecat-core echo pipeline demo ===\n");

    let stt = FrameProcessor::new("STT", Box::new(FakeSTT), false);
    let llm = FrameProcessor::new("LLM", Box::new(FakeLLM), false);
    let tts = FrameProcessor::new("TTS", Box::new(FakeTTS), false);

    let mut params = PipelineParams::default();
    params.enable_heartbeats = true;
    params.heartbeat_seconds = 2.0;
    params.idle_timeout = Some(Duration::from_secs(5));
    params.cancel_on_idle_timeout = true;

    let task = PipelineTask::new(vec![stt, llm, tts], params);

    // Lifecycle callbacks
    task.add_on_pipeline_started(|_| Box::pin(async {
        println!("\n✓ Pipeline running\n");
    }));

    task.add_on_pipeline_finished(|_, reason| Box::pin(async move {
        println!("\n✓ Pipeline finished — reason: {:?}\n", reason);
    }));

    task.add_on_idle_timeout(|| Box::pin(async {
        println!("\n⚠ Idle timeout reached");
    }));

    // Set up downstream filter to watch heartbeats
    let mut filter = std::collections::HashSet::new();
    filter.insert(FrameKind::Heartbeat);
    task.set_downstream_filter(filter);
    task.add_on_frame_reached_downstream(|frame| Box::pin(async move {
        if frame.kind() == FrameKind::Heartbeat {
            println!("[♥  ] Heartbeat — pipeline healthy");
        }
    }));

    let tx = task.push_sender();

    // Simulate 3 rounds of audio input then graceful shutdown
    tokio::spawn(async move {
        let phrases = [
            "hello world",
            "how are you today",
            "goodbye pipeline",
        ];

        for phrase in &phrases {
            tokio::time::sleep(Duration::from_millis(500)).await;

            println!("\n--- Injecting audio: \"{}\" ---", phrase);

            // Build an audio frame with metadata marker
            let mut data = DataFrameData {
                content: phrase.as_bytes().to_vec(),
                ..Default::default()
            };
            data.metadata.insert("audio".to_string(), "true".to_string());

            let audio_frame = Frame {
                id: pipecat_core::frames::next_frame_id(),
                sibling_id: None,
                inner: pipecat_core::frames::FrameInner::Data(
                    pipecat_core::frames::DataFrame::Data(data)
                ),
            };

            tx.send((audio_frame, FrameDirection::Downstream)).await.ok();

            // Wait for TTS to finish before sending next
            tokio::time::sleep(Duration::from_millis(400)).await;
        }

        // Graceful shutdown
        println!("\n--- Sending EndTask ---");
        tx.send((Frame::end_task(), FrameDirection::Upstream)).await.ok();
    });

    match task.run(system_clock(), None).await {
        Ok(()) => println!("Demo complete."),
        Err(e) => eprintln!("Pipeline error: {}", e),
    }
}
