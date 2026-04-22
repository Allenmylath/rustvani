//! Sarvam TTS pipeline test.
//!
//! Feeds LLMText frames manually, verifies OutputAudioRaw comes back.
//!
//! Run:
//!   SARVAM_API_KEY=your-key cargo run --bin sarvam_tts_test

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use rustvani::{
    system_clock, ControlFrame, DataFrame, Frame, FrameDirection, FrameHandler, FrameInner,
    FrameKind, FrameProcessor, PipelineParams, PipelineTask, Result, SystemFrame,
};
use rustvani::services::{SarvamTtsConfig, SarvamTtsHandler};

// ---------------------------------------------------------------------------
// AudioPrinter — logs OutputAudioRaw frames
// ---------------------------------------------------------------------------

struct AudioPrinter {
    received: std::sync::Mutex<usize>,
}

impl AudioPrinter {
    fn new() -> Self {
        Self { received: std::sync::Mutex::new(0) }
    }
}

#[async_trait]
impl FrameHandler for AudioPrinter {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match &frame.inner {
            FrameInner::Data(DataFrame::OutputAudioRaw(audio)) => {
                let mut count = self.received.lock().unwrap();
                *count += 1;
                println!(
                    "[audio] chunk #{} — {} bytes at {}Hz",
                    count, audio.audio.len(), audio.sample_rate
                );
            }
            FrameInner::Control(ControlFrame::LLMFullResponseEnd) => {
                println!("[tts] LLMFullResponseEnd received");
            }
            _ if frame.kind() == FrameKind::Error => {
                println!("[error] {:?}", frame.name());
            }
            _ => {}
        }
        processor.push_frame(frame, direction).await
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
    .init();

    let api_key = std::env::var("SARVAM_API_KEY").expect("SARVAM_API_KEY not set");

    println!("=== rustvani Sarvam TTS pipeline test ===\n");

    let tts = SarvamTtsHandler::new(SarvamTtsConfig {
        api_key,
        model:    "bulbul:v2".to_string(),
        voice:    "anushka".to_string(),
        language: "en-IN".to_string(),
        ..SarvamTtsConfig::default()
    })
    .expect("TTS config failed")
    .into_processor();

    let printer = FrameProcessor::new("AudioPrinter", Box::new(AudioPrinter::new()), false);

    let task = PipelineTask::new(
        vec![tts, printer],
        PipelineParams::default(),
    );

    let push_tx = task.push_sender();

    // Feed a short LLM turn
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(500)).await;

        let sentences = [
            "Hello, this is a test of the Sarvam TTS service.",
            " The quick brown fox jumps over the lazy dog.",
            " How are you today?",
        ];

        // Simulate LLM streaming token by token
        push_tx
            .send((Frame::llm_full_response_start(), FrameDirection::Downstream))
            .await
            .unwrap();

        for sentence in &sentences {
            println!("[llm] token: {:?}", sentence);
            push_tx
                .send((Frame::llm_text(sentence.to_string()), FrameDirection::Downstream))
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        push_tx
            .send((Frame::llm_full_response_end(), FrameDirection::Downstream))
            .await
            .unwrap();

        println!("[feeder] waiting for audio...");
        tokio::time::sleep(Duration::from_secs(8)).await;

        println!("[feeder] sending End");
        push_tx
            .send((Frame::end(), FrameDirection::Downstream))
            .await
            .unwrap();
    });

    task.run(system_clock(), None).await?;

    println!("\n=== TTS test complete ===");
    Ok(())
}