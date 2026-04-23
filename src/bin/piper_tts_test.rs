//! Piper TTS pipeline test (high quality, hardcoded).
//!
//! Feeds LLMText frames manually, verifies OutputAudioRaw comes back.
//!
//! Run in Docker:
//!   docker run --rm rustvani /app/target/release/piper_tts_test

use std::time::Duration;

use async_trait::async_trait;
use rustvani::{
    system_clock, ControlFrame, DataFrame, Frame, FrameDirection, FrameHandler, FrameInner,
    FrameKind, FrameProcessor, PipelineParams, PipelineTask, Result, SystemFrame,
};
use rustvani::services::{PiperTtsConfig, PiperTtsHandler, PiperQuality};

// ---------------------------------------------------------------------------
// AudioPrinter — logs OutputAudioRaw frames
// ---------------------------------------------------------------------------

struct AudioPrinter {
    received:    std::sync::Mutex<usize>,
    total_bytes: std::sync::Mutex<usize>,
}

impl AudioPrinter {
    fn new() -> Self {
        Self {
            received:    std::sync::Mutex::new(0),
            total_bytes: std::sync::Mutex::new(0),
        }
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
                let mut bytes = self.total_bytes.lock().unwrap();
                *count += 1;
                *bytes += audio.audio.len();
                if *count <= 5 || *count % 50 == 0 {
                    println!(
                        "[audio] chunk #{:<4} — {} bytes at {}Hz  (total: {} bytes)",
                        count, audio.audio.len(), audio.sample_rate, bytes
                    );
                }
            }
            FrameInner::Control(ControlFrame::LLMFullResponseEnd) => {
                println!("[tts] LLMFullResponseEnd received");
                let count = *self.received.lock().unwrap();
                let bytes = *self.total_bytes.lock().unwrap();
                println!(
                    "[tts] Total: {} chunks, {} bytes ({:.1}s of audio at 22050Hz mono)",
                    count, bytes, bytes as f64 / (22050.0 * 2.0)
                );
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

    println!("=== rustvani Piper TTS pipeline test (low) ===\n");

    let tts = PiperTtsHandler::new(PiperTtsConfig {
        quality:   PiperQuality::Low,
        model_dir: std::path::PathBuf::from("./piper-models"),
        ..PiperTtsConfig::default()
    })
    .expect("PiperTts init failed")
    .into_processor();

    let printer = FrameProcessor::new("AudioPrinter", Box::new(AudioPrinter::new()), false);

    let task = PipelineTask::new(
        vec![tts, printer],
        let task = PipelineTask::new(
        vec![
            transport.input(),
            stt,
            user_agg,
            llm,
            assistant_agg,
            tts,
            transport.output(),
        ],
        PipelineParams { allow_interruptions: true, ..PipelineParams::default() }
    );

    let push_tx = task.push_sender();

    tokio::join!(
        async { task.run(system_clock(), None).await.ok(); },
        async {
            tokio::time::sleep(Duration::from_millis(500)).await;

            println!("[test] Sending LLMFullResponseStart");
            let _ = push_tx.send((
                Frame::llm_full_response_start(),
                FrameDirection::Downstream,
            )).await;

            // Sentence 1 — simulates LLM streaming
            let chunks_1 = vec![
                "Hello", " there", "!", " This", " is", " a", " test",
                " of", " the", " Piper", " text", " to", " speech",
                " engine", ".",
            ];
            println!("[test] Streaming: \"Hello there! This is a test of the Piper text to speech engine.\"");
            for chunk in chunks_1 {
                let _ = push_tx.send((
                    Frame::llm_text(chunk.to_string()),
                    FrameDirection::Downstream,
                )).await;
                tokio::time::sleep(Duration::from_millis(30)).await;
            }

            // Sentence 2
            let chunks_2 = vec![
                " The", " quick", " brown", " fox", " jumps",
                " over", " the", " lazy", " dog", ".",
            ];
            println!("[test] Streaming: \"The quick brown fox jumps over the lazy dog.\"");
            for chunk in chunks_2 {
                let _ = push_tx.send((
                    Frame::llm_text(chunk.to_string()),
                    FrameDirection::Downstream,
                )).await;
                tokio::time::sleep(Duration::from_millis(30)).await;
            }

            println!("[test] Sending LLMFullResponseEnd");
            let _ = push_tx.send((
                Frame::llm_full_response_end(),
                FrameDirection::Downstream,
            )).await;

            // Wait for TTS to finish
            tokio::time::sleep(Duration::from_secs(5)).await;

            println!("\n[test] Sending EndFrame — shutting down");
            let _ = push_tx.send((
                Frame::end(),
                FrameDirection::Downstream,
            )).await;

            tokio::time::sleep(Duration::from_millis(500)).await;
        },
    );

    println!("\n=== Test complete ===");
    Ok(())
}
