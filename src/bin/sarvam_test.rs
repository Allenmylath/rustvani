//! Sarvam STT integration test.
//!
//! Reads test.wav, feeds through pipeline with SarvamSttHandler,
//! prints any TranscriptionFrames received.
//!
//! Run:
//!   docker run --rm -e SARVAM_API_KEY=your-key rustvani-sarvam /app/target/release/sarvam_test

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use rustvani::{
    system_clock, Frame, FrameDirection, FrameHandler, FrameInner,
    FrameProcessor, PipelineParams, PipelineTask, Result, VadParams, SileroVadNative,
    DataFrame,
};
use rustvani::transport::{BaseTransport, TransportParams};
use rustvani::services::{SarvamSttConfig, SarvamSttHandler};

const CHUNK_BYTES: usize = 512 * 2;
const WAV_PATH: &str = "/app/test.wav";

// ---------------------------------------------------------------------------
// TranscriptionPrinter
// ---------------------------------------------------------------------------

struct TranscriptionPrinter;

#[async_trait]
impl FrameHandler for TranscriptionPrinter {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        if let FrameInner::Data(DataFrame::Transcription(ref t)) = frame.inner {
            println!(
                "[transcription] text='{}' lang={:?} finalized={}",
                t.text, t.language, t.finalized
            );
        }
        processor.push_frame(frame, direction).await
    }
}

// ---------------------------------------------------------------------------
// WAV reader
// ---------------------------------------------------------------------------

fn read_wav_pcm(path: &str) -> std::result::Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| format!("Failed to open {}: {}", path, e))?;
    let spec = reader.spec();
    println!(
        "[wav] sample_rate={}  channels={}  bits={}",
        spec.sample_rate, spec.channels, spec.bits_per_sample
    );
    assert_eq!(spec.sample_rate, 16_000, "Need 16 kHz");
    assert_eq!(spec.channels, 1, "Need mono");
    let samples: Vec<i16> = reader
        .samples::<i16>()
        .map(|s| s.expect("WAV read error"))
        .collect();
    println!(
        "[wav] samples={}  duration={:.1}s",
        samples.len(),
        samples.len() as f32 / 16_000.0
    );
    Ok(samples.iter().flat_map(|s| s.to_le_bytes()).collect())
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

    let api_key = std::env::var("SARVAM_API_KEY")
        .expect("SARVAM_API_KEY env var not set");

    println!("=== rustvani Sarvam STT integration test ===\n");

    let pcm_bytes = read_wav_pcm(WAV_PATH)?;

    // VAD — same config as pipeline_test
    let vad_analyzer = Arc::new(
        SileroVadNative::new(16_000).map_err(|e| format!("VAD init failed: {}", e))?
    );

    let transport = Arc::new(BaseTransport::new("Test", TransportParams {
        audio_in_enabled:         true,
        audio_in_sample_rate:     Some(16_000),
        audio_in_channels:        1,
        audio_in_passthrough:     true,
        audio_in_stream_on_start: true,
        vad_analyzer:             Some(vad_analyzer),
        vad_params:               VadParams::default(),
        ..TransportParams::default()
    }));

    // Sarvam STT — Malayalam, saarika:v2.5
    let stt = SarvamSttHandler::new(SarvamSttConfig {
        api_key,
        model:    "saaras:v3".to_string(),
        language: Some("en-IN".to_string()),
        mode:     Some("transcribe".to_string()),
        ..SarvamSttConfig::default()
    })
    .into_processor();

    let printer = FrameProcessor::new(
        "TranscriptionPrinter",
        Box::new(TranscriptionPrinter),
        false,
    );

    // Pipeline: input → stt → printer → output
    let task = PipelineTask::new(
        vec![transport.input(), stt, printer, transport.output()],
        PipelineParams::default(),
    );

    let push_tx = task.push_sender();
    let transport_for_feeder = transport.clone();

    let feeder = tokio::spawn(async move {
        let chunks: Vec<&[u8]> = pcm_bytes
            .chunks(CHUNK_BYTES)
            .filter(|c| c.len() == CHUNK_BYTES)
            .collect();

        println!("[feeder] pushing {} chunks", chunks.len());

        for chunk in chunks {
            let data = rustvani::AudioRawData::new(chunk.to_vec(), 16_000, 1);
            transport_for_feeder.push_audio_frame(data).await;
            // Slight throttle — don't flood the WebSocket
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // Wait for final transcript to arrive after last flush
        println!("[feeder] all audio pushed — waiting 3s for final transcript");
        tokio::time::sleep(Duration::from_secs(3)).await;

        println!("[feeder] sending End");
        let _ = push_tx.send((Frame::end(), FrameDirection::Downstream)).await;
    });

    task.run(system_clock(), None).await?;
    let _ = feeder.await;

    println!("\n=== test complete ===");
    Ok(())
}
