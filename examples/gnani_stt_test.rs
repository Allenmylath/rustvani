//! Gnani STT standalone test — no LLM, no TTS.
//!
//! Pipeline:
//!   ChannelTransport.input()
//!     → VAD + GnaniStt
//!     → TranscriptionPrinter  (prints + forwards as text)
//!     → ChannelTransport.output()
//!
//! Environment:
//!   GNANI_API_KEY  — required
//!   RUST_LOG       — optional, e.g. "info" or "rustvani=debug"
//!
//! Run:
//!   cargo run --example gnani_stt_test -- tests/test.wav

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc;

use rustvani::{
    frames::{
        DataFrame, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor,
    },
    pipeline::{PipelineParams, PipelineTask},
    services::{GnaniSttConfig, GnaniSttHandler},
    system_clock, SileroVadNative, VadParams,
    transport::{ChannelMessage, ChannelTransport, TransportParams},
};

const CHUNK_BYTES: usize = 512 * 2; // 512 samples * 2 bytes (16-bit mono)

/// Read a 16kHz mono WAV into raw PCM bytes.
fn read_wav_pcm(path: &str) -> std::result::Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| format!("Failed to open {}: {}", path, e))?;
    let spec = reader.spec();
    println!("[wav] {}Hz  {}ch  {}bit", spec.sample_rate, spec.channels, spec.bits_per_sample);
    assert_eq!(spec.sample_rate, 16_000, "Need 16kHz");
    assert_eq!(spec.channels, 1, "Need mono");

    let samples: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
    println!("[wav] {:.1}s  ({} samples)", samples.len() as f32 / 16_000.0, samples.len());
    Ok(samples.iter().flat_map(|s| s.to_le_bytes()).collect())
}

// ---------------------------------------------------------------------------
// Simple processor: prints transcription and forwards it as a text message
// ---------------------------------------------------------------------------

struct TranscriptionPrinter;

#[async_trait]
impl FrameHandler for TranscriptionPrinter {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> rustvani::error::Result<()> {
        match &frame.inner {
            FrameInner::Data(DataFrame::Transcription(data)) => {
                let text = data.text.trim().to_string();
                let finalized = data.finalized;
                if !text.is_empty() {
                    println!("\n[transcription] {} (final={})\n", text, finalized);
                }
                // Forward the transcription frame so downstream can see it
                processor.push_frame(frame, direction).await?;

                // Also send a RaviServerMessage so the output channel gets text
                let msg = format!("{{\"type\":\"transcription\",\"text\":\"{}\"}}", text);
                processor.push_frame(
                    Frame::ravi_server_message(msg),
                    FrameDirection::Downstream,
                ).await?;
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
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();

    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
    .init();

    let gnani_api_key = std::env::var("GNANI_API_KEY")
        .map_err(|_| "GNANI_API_KEY environment variable not set")?;

    let wav_path = std::env::args().nth(1)
        .unwrap_or_else(|| "tests/test.wav".to_string());

    println!("=== Gnani STT standalone test ===\n");

    // -----------------------------------------------------------------------
    // 1. Channels
    // -----------------------------------------------------------------------
    let (incoming_tx, incoming_rx) = mpsc::channel::<ChannelMessage>(100);
    let (outgoing_tx, mut outgoing_rx) = mpsc::channel::<ChannelMessage>(100);

    // -----------------------------------------------------------------------
    // 2. Transport + Gnani STT + printer
    // -----------------------------------------------------------------------
    let vad_analyzer = Arc::new(SileroVadNative::new(16_000)?);

    let transport = ChannelTransport::new(
        "ChannelTransport",
        TransportParams {
            audio_in_enabled:      true,
            audio_in_sample_rate:  Some(16_000),
            audio_in_channels:     1,
            audio_in_passthrough:  true,
            audio_in_stream_on_start: true,
            audio_out_enabled:     true,
            audio_out_sample_rate: Some(16_000),
            audio_out_channels:    1,
            audio_out_10ms_chunks: 4,
            vad_analyzer:          Some(vad_analyzer),
            vad_params:            VadParams {
                confidence: 0.4,
                min_volume: 0.1,
                ..VadParams::default()
            },
            ..TransportParams::default()
        },
        incoming_rx,
    );

    let stt = GnaniSttHandler::new(GnaniSttConfig {
        api_key:       gnani_api_key,
        language_code: "en-IN".into(),
        sample_rate:   16_000,
        format:        "verbatim".into(),
        itn_native_numerals: false,
        ..Default::default()
    })
    .into_processor();

    let printer = FrameProcessor::new("TranscriptionPrinter", Box::new(TranscriptionPrinter), false);

    // -----------------------------------------------------------------------
    // 3. Pipeline
    // -----------------------------------------------------------------------
    let task = PipelineTask::new(
        vec![
            transport.input(),
            stt,
            printer,
            transport.output(),
        ],
        PipelineParams {
            allow_interruptions: false,
            ..PipelineParams::default()
        },
    );

    let push_tx = task.push_sender();

    let _pipeline_handle = tokio::spawn(async move {
        if let Err(e) = task.run(system_clock(), None).await {
            log::error!("Pipeline error: {}", e);
        }
        println!("[pipeline] stopped");
    });

    let push_tx_transport = push_tx.clone();
    let _transport_handle = tokio::spawn(async move {
        transport.run(push_tx_transport, outgoing_tx).await;
        println!("[transport] stopped");
    });

    tokio::time::sleep(Duration::from_millis(200)).await;
    println!("[main] pushing audio...\n");

    // -----------------------------------------------------------------------
    // 4. Feed audio
    // -----------------------------------------------------------------------
    let pcm_bytes = read_wav_pcm(&wav_path)?;
    let chunks: Vec<&[u8]> = pcm_bytes
        .chunks(CHUNK_BYTES)
        .filter(|c| c.len() == CHUNK_BYTES)
        .collect();

    for chunk in chunks {
        let data = rustvani::frames::AudioRawData::new(chunk.to_vec(), 16_000, 1);
        if incoming_tx.send(ChannelMessage::Audio(data.audio.to_vec())).await.is_err() {
            println!("[main] incoming channel closed");
            break;
        }
        tokio::time::sleep(Duration::from_millis(16)).await;
    }

    println!("[main] audio sent — draining output...\n");

    // -----------------------------------------------------------------------
    // 5. Drain output for a few seconds then end
    // -----------------------------------------------------------------------
    tokio::time::sleep(Duration::from_secs(5)).await;

    push_tx.send((rustvani::frames::Frame::end_task(), rustvani::frames::FrameDirection::Upstream)).await.ok();

    let mut text_count = 0usize;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);

    while tokio::time::timeout_at(deadline, outgoing_rx.recv()).await.is_ok() {
        text_count += 1;
    }

    println!("\n=== done ===");
    println!("Output messages received: {}", text_count);
    Ok(())
}
