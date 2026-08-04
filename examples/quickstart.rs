//! The README Quick Start, verbatim.
//!
//! This file exists so the snippet in README.md is compiler-checked. If you
//! change one, change the other — `cargo build --example quickstart` must pass
//! with **default features only**.
//!
//! A complete voice agent on ws://0.0.0.0:8080/ws:
//!   audio → VAD → Sarvam STT → OpenAI LLM → Deepgram TTS → audio
//!
//! Run:
//!   SARVAM_API_KEY=… OPENAI_API_KEY=… DEEPGRAM_API_KEY=… \
//!     cargo run --example quickstart

use std::sync::Arc;

use rustvani::axum::{
    extract::{ws::WebSocket, State, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
    Router,
};
use rustvani::processors::{
    llm_assistant_aggregator::LLMAssistantAggregator, llm_user_aggregator::LLMUserAggregator,
};
use rustvani::services::{
    DeepgramTtsConfig, DeepgramTtsHandler, OpenAILLMConfig, OpenAILLMHandler, SarvamSttConfig,
    SarvamSttHandler,
};
use rustvani::transport::{TransportParams, WebSocketParams, WebSocketTransport};
use rustvani::{
    shared_context, system_clock, PipelineParams, PipelineTask, SileroVadNative, VadParams,
};

#[derive(Clone)]
struct AppState {
    sarvam_key: String,
    openai_key: String,
    deepgram_key: String,
}

/// One fully isolated pipeline per connection.
async fn handle_connection(socket: WebSocket, state: AppState) {
    // 1. VAD — pure Rust, no ONNX Runtime, no .so files to bundle.
    let vad = match SileroVadNative::new(16_000) {
        Ok(v) => Arc::new(v),
        Err(e) => return log::error!("VAD init failed: {e}"),
    };

    // 2. Transport — owns the VAD and the audio I/O.
    let transport = WebSocketTransport::new(
        "quickstart",
        WebSocketParams {
            transport: TransportParams {
                audio_in_enabled: true,
                audio_in_sample_rate: Some(16_000),
                audio_out_enabled: true,
                audio_out_sample_rate: Some(24_000), // Deepgram TTS default
                vad_analyzer: Some(vad),
                vad_params: VadParams { confidence: 0.4, min_volume: 0.1, ..Default::default() },
                ..TransportParams::default()
            },
        },
    );

    // 3. Shared conversation context — the aggregators read and write it.
    let context = shared_context(Some(
        "You are a helpful voice assistant. Keep answers to one or two sentences.".into(),
    ));

    // 4. Services. The speech-enhancement chain (HPF → RNNoise → AGC → limiter)
    //    is already on inside the STT handler; nothing to wire up.
    let stt = SarvamSttHandler::new(SarvamSttConfig {
        api_key: state.sarvam_key,
        model: "saaras:v3".into(),
        language: Some("en-IN".into()),
        ..SarvamSttConfig::default()
    })
    .into_processor();

    let llm = OpenAILLMHandler::new(OpenAILLMConfig {
        api_key: state.openai_key,
        model: "gpt-4o-mini".into(),
        ..OpenAILLMConfig::default()
    })
    .into_processor();

    let tts = match DeepgramTtsHandler::new(DeepgramTtsConfig {
        api_key: state.deepgram_key,
        ..DeepgramTtsConfig::default()
    }) {
        Ok(t) => t.into_processor(),
        Err(e) => return log::error!("TTS init failed: {e}"),
    };

    // 5. Aggregators bridge VAD/STT ↔ LLM. `new` already returns a
    //    FrameProcessor — no `.into_processor()` here.
    let user_agg = LLMUserAggregator::new(context.clone());
    let assistant_agg = LLMAssistantAggregator::new(context.clone());

    // 6. Assemble and run.
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
        PipelineParams { allow_interruptions: true, ..PipelineParams::default() },
    );

    // Take the injection handle *before* run() — it can only be taken once.
    let push_tx = task.push_sender();

    tokio::join!(
        async { task.run(system_clock(), None).await.ok(); },
        transport.run_socket(socket, push_tx),
    );
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_connection(socket, state))
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let state = AppState {
        sarvam_key: std::env::var("SARVAM_API_KEY").expect("SARVAM_API_KEY"),
        openai_key: std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY"),
        deepgram_key: std::env::var("DEEPGRAM_API_KEY").expect("DEEPGRAM_API_KEY"),
    };

    let app = Router::new().route("/ws", get(ws_handler)).with_state(state);
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();

    log::info!("listening on ws://0.0.0.0:8080/ws");
    rustvani::axum::serve(listener, app).await.unwrap();
}
