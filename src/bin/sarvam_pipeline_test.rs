//! Sarvam STT pipeline integration test.
//!
//! Topology:
//!   transport.input() → SarvamSttHandler → PrintProcessor → transport.output()
//!
//! Audio is fed via BaseTransport::push_audio_frame() — same path as production.
//! VAD runs inside transport.input(). SarvamStt sends audio to Sarvam WebSocket.
//! PrintProcessor logs VAD frames and TranscriptionFrames.
//!
//! Run:
//!   SARVAM_API_KEY=your-key cargo run --bin sarvam_pipeline_test

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use rustvani::{
    system_clock, AudioRawData, DataFrame, Frame, FrameDirection, FrameHandler,
    FrameInner, FrameKind, FrameProcessor, PipelineParams, PipelineTask,
    Result, SileroVad, VadParams,
};
use rustvani::services::{SarvamSttConfig, SarvamSttHandler};
use rustvani::transport::{BaseTransport, TransportParams};

const CHUNK_BYTES: usize = 512 * 2; // 512 samples * 2 bytes (16-bit mono)
const WAV_PATH: &str = "tests/test.wav";
const MAX_SECS: f32 = 10.0;

// ---------------------------------------------------------------------------
// PrintHandler — logs VAD frames and TranscriptionFrames
// ---------------------------------------------------------------------------

struct PrintHandler;

#[async_trait]
impl FrameHandler for PrintHandler {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        match frame.kind() {
            FrameKind::VADUserStartedSpeaking => {
                println!("[{:.3}] VADUserStartedSpeaking", ts);
            }
            FrameKind::VADUserStoppedSpeaking => {
                println!("[{:.3}] VADUserStoppedSpeaking", ts);
            }
            FrameKind::Transcription => {
                if let FrameInner::Data(DataFrame::Transcription(ref t)) = frame.inner {
                    println!(
                        "[{:.3}] TranscriptionFrame  text=\"{}\"  lang={:?}  finalized={}",
                        ts, t.text, t.language, t.finalized
                    );
                }
            }
            _ => {}
        }

        processor.push_frame(frame, direction).await
    }
}

// ---------------------------------------------------------------------------
// WAV reader
// ---------------------------------------------------------------------------

fn read_wav_pcm(path: &str, max_secs: f32) -> std::result::Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| format!("Failed to open {}: {}", path, e))?;
    let spec = reader.spec();
    println!("[wav] {}Hz  {}ch  {}bit", spec.sample_rate, spec.channels, spec.bits_per_sample);
    assert_eq!(spec.sample_rate, 16_000, "Need 16kHz");
    assert_eq!(spec.channels, 1, "Need mono");

    let samples: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
    let max_samples = (spec.sample_rate as f32 * max_secs) as usize;
    let samples = &samples[..samples.len().min(max_samples)];

    println!("[wav] using {:.1}s  ({} samples)", samples.len() as f32 / 16_000.0, samples.len());
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

    let wav_path = std::env::args().nth(1)
        .unwrap_or_else(|| WAV_PATH.to_string());

    println!("=== rustvani Sarvam STT pipeline test ===\n");

    let pcm_bytes = read_wav_pcm(&wav_path, MAX_SECS)?;

    // VAD
    let vad_analyzer = Arc::new(
        SileroVad::new(16_000).map_err(|e| format!("VAD init: {}", e))?
    );

    // Transport
    let transport = Arc::new(BaseTransport::new("Test", TransportParams {
        audio_in_enabled:         true,
        audio_in_sample_rate:     Some(16_000),
        audio_in_channels:        1,
        audio_in_passthrough:     true,  // pass audio to SarvamStt
        audio_in_stream_on_start: true,
        vad_analyzer:             Some(vad_analyzer),
        vad_params:               VadParams::default(),
        ..TransportParams::default()
    }));

    // Sarvam STT
    let stt = SarvamSttHandler::new(SarvamSttConfig {
        api_key,
        model:    "saaras:v3".to_string(),
        language: Some("en-IN".to_string()),
        mode:     Some("transcribe".to_string()),
        ..SarvamSttConfig::default()
    })
    .into_processor();

    // Printer
    let printer = FrameProcessor::new("Printer", Box::new(PrintHandler), false);

    // Pipeline
    let task = PipelineTask::new(
        vec![transport.input(), stt, printer, transport.output()],
        PipelineParams::default(),
    );

    let push_tx = task.push_sender();
    let transport_clone = transport.clone();

    let feeder = tokio::spawn(async move {
        let chunks: Vec<&[u8]> = pcm_bytes
            .chunks(CHUNK_BYTES)
            .filter(|c| c.len() == CHUNK_BYTES)
            .collect();

        println!("[feeder] pushing {} chunks\n", chunks.len());

        for chunk in chunks {
            let data = AudioRawData::new(chunk.to_vec(), 16_000, 1);
            transport_clone.push_audio_frame(data).await;
            // Throttle to ~realtime so VAD has time to react
            tokio::time::sleep(Duration::from_millis(16)).await;
        }

        println!("\n[feeder] done — waiting 5s for final transcripts");
        tokio::time::sleep(Duration::from_secs(5)).await;

        println!("[feeder] sending End");
        let _ = push_tx.send((Frame::end(), FrameDirection::Downstream)).await;
    });

    task.run(system_clock(), None).await?;
    let _ = feeder.await;

    println!("\n=== pipeline test complete ===");
    Ok(())
}
