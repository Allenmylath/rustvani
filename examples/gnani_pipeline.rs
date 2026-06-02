//! Gnani STT voice pipeline — Gnani STT → OpenAI LLM → Deepgram TTS
//!
//! This example demonstrates the full Gnani (Vachana) STT integration
//! using a ChannelTransport for easy local testing without a WebSocket client.
//!
//! Pipeline topology:
//!   ChannelTransport.input()
//!     → VAD + GnaniStt
//!     → LLMUserAggregator
//!     → OpenAILLM
//!     → LLMAssistantAggregator
//!     → DeepgramTts
//!     → ChannelTransport.output()
//!
//! Environment variables:
//!   GNANI_API_KEY    — required (Gnani Vachana STT)
//!   OPENAI_API_KEY   — required (OpenAI LLM)
//!   DEEPGRAM_API_KEY — required (Deepgram TTS)
//!
//! Run:
//!   GNANI_API_KEY=xxx OPENAI_API_KEY=xxx DEEPGRAM_API_KEY=xxx \
//!     cargo run --example gnani_pipeline -- tests/test.wav

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;

use rustvani::{
    context::shared_context,
    frames::AudioRawData,
    pipeline::{PipelineParams, PipelineTask},
    processors::{
        llm_assistant_aggregator::LLMAssistantAggregator,
        llm_user_aggregator::LLMUserAggregator,
    },
    services::{
        DeepgramTtsConfig, DeepgramTtsHandler,
        GnaniSttConfig, GnaniSttHandler,
        OpenAILLMConfig, OpenAILLMHandler,
    },
    system_clock, SileroVadNative, VadParams,
    transport::{ChannelMessage, ChannelTransport, TransportParams},
};

const CHUNK_BYTES: usize = 512 * 2; // 512 samples * 2 bytes (16-bit mono)

/// Read a 16kHz mono WAV into raw PCM bytes.
fn read_wav_pcm(path: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();

    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
    .init();

    // -----------------------------------------------------------------------
    // 1. Read API keys from environment
    // -----------------------------------------------------------------------
    let gnani_api_key    = std::env::var("GNANI_API_KEY")
        .map_err(|_| "GNANI_API_KEY environment variable not set")?;
    let openai_api_key   = std::env::var("OPENAI_API_KEY")
        .map_err(|_| "OPENAI_API_KEY environment variable not set")?;
    let deepgram_api_key = std::env::var("DEEPGRAM_API_KEY")
        .map_err(|_| "DEEPGRAM_API_KEY environment variable not set")?;

    let wav_path = std::env::args().nth(1)
        .unwrap_or_else(|| "tests/test.wav".to_string());

    println!("=== rustvani Gnani STT pipeline ===\n");
    println!("Pipeline: audio → VAD → GnaniSTT → OpenAILLM → DeepgramTTS → audio\n");

    // -----------------------------------------------------------------------
    // 2. Create channels — these are the handles you hold to push/drain
    // -----------------------------------------------------------------------
    let (incoming_tx, incoming_rx) = mpsc::channel::<ChannelMessage>(100);
    let (outgoing_tx, mut outgoing_rx) = mpsc::channel::<ChannelMessage>(100);

    // -----------------------------------------------------------------------
    // 3. Build transport + processors
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
            audio_out_sample_rate: Some(24_000), // Deepgram default
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

    let context = shared_context(Some(
        "You are a helpful voice assistant. Keep answers concise and conversational.".into()
    ));

    // Gnani STT — reads GNANI_API_KEY from env above
    let stt = GnaniSttHandler::new(GnaniSttConfig {
        api_key:       gnani_api_key,
        language_code: "en-IN".into(),   // BCP-47: en-IN, hi-IN, ta-IN, etc.
        sample_rate:   16_000,           // must match audio source
        format:        "verbatim".into(),// "verbatim" or "transcribe" (ITN)
        itn_native_numerals: false,
        ..Default::default()
    })
    .into_processor();

    let user_agg      = LLMUserAggregator::new(context.clone());
    let assistant_agg = LLMAssistantAggregator::new(context.clone());

    let llm = OpenAILLMHandler::new(OpenAILLMConfig {
        api_key: openai_api_key,
        model:   "gpt-4o-mini".into(),
        ..OpenAILLMConfig::default()
    })
    .into_processor();

    let tts = DeepgramTtsHandler::new(DeepgramTtsConfig {
        api_key: deepgram_api_key,
        ..DeepgramTtsConfig::default()
    })?
    .into_processor();

    // -----------------------------------------------------------------------
    // 4. Build pipeline task
    // -----------------------------------------------------------------------
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
        PipelineParams {
            allow_interruptions: true,
            ..PipelineParams::default()
        },
    );

    let push_tx = task.push_sender();

    // -----------------------------------------------------------------------
    // 5. Start pipeline + transport in background
    // -----------------------------------------------------------------------
    let _pipeline_handle = tokio::spawn(async move {
        if let Err(e) = task.run(system_clock(), None).await {
            log::error!("Pipeline error: {}", e);
        }
        println!("[pipeline] stopped");
    });

    let _transport_handle = tokio::spawn(async move {
        transport.run(push_tx, outgoing_tx).await;
        println!("[transport] stopped");
    });

    // Give the pipeline a moment to reach StartFrame
    tokio::time::sleep(Duration::from_millis(200)).await;
    println!("[main] pipeline is running — push audio into incoming_tx, drain from outgoing_rx\n");

    // -----------------------------------------------------------------------
    // 6. Read WAV and push audio into the pipeline
    // -----------------------------------------------------------------------
    let pcm_bytes = read_wav_pcm(&wav_path)?;
    let chunks: Vec<&[u8]> = pcm_bytes
        .chunks(CHUNK_BYTES)
        .filter(|c| c.len() == CHUNK_BYTES)
        .collect();

    println!("[main] pushing {} audio chunks...\n", chunks.len());

    for chunk in chunks {
        let data = AudioRawData::new(chunk.to_vec(), 16_000, 1);
        if incoming_tx.send(ChannelMessage::Audio(data.audio)).await.is_err() {
            println!("[main] incoming channel closed — stopping feed");
            break;
        }
        // Throttle to ~realtime (16ms ≈ 512 samples @ 16kHz)
        tokio::time::sleep(Duration::from_millis(16)).await;
    }

    println!("[main] done sending audio — waiting for pipeline to finish...\n");

    // -----------------------------------------------------------------------
    // 7. Drain output from the pipeline until it closes
    // -----------------------------------------------------------------------
    let mut total_audio_bytes = 0usize;
    let mut total_text_msgs   = 0usize;

    while let Some(msg) = outgoing_rx.recv().await {
        match msg {
            ChannelMessage::Audio(bytes) => {
                total_audio_bytes += bytes.len();
                print!("\r[out] TTS audio received: {} bytes total", total_audio_bytes);
            }
            ChannelMessage::Text(text) => {
                total_text_msgs += 1;
                println!("\r[out] text msg #{}: {}", total_text_msgs, text);
            }
            ChannelMessage::Interruption => {
                println!("\r[out] interruption — clearing buffers");
            }
            ChannelMessage::ClientVadStart(ts) => {
                println!("\r[out] VAD start detected @ {:.3}s", ts);
            }
            ChannelMessage::ClientVadStop(ts) => {
                println!("\r[out] VAD stop detected @ {:.3}s", ts);
            }
        }
    }

    println!("\n\n=== pipeline complete ===");
    println!("Total TTS audio: {} bytes", total_audio_bytes);
    println!("Total text msgs: {}", total_text_msgs);
    Ok(())
}
