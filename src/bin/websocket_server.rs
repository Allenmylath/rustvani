//! rustvani WebSocket voice agent server.
//!
//! Listens on ws://0.0.0.0:8080/ws
//!
//! Each connection gets a fully isolated pipeline:
//!   WebSocketTransport.input()
//!     → SarvamStt
//!     → LLMUserAggregator
//!     → SarvamLLM
//!     → LLMAssistantAggregator
//!     → SarvamTts
//!     → WebSocketTransport.output()
//!
//! Wire protocol:
//!   Client → Server : Binary WebSocket — raw i16 LE PCM, 16 kHz mono, 512-sample chunks
//!   Server → Client : Binary WebSocket — raw i16 LE PCM at TTS sample rate (22050 Hz bulbul:v2)
//!
//! Environment variables:
//!   SARVAM_API_KEY   — required
//!   SYSTEM_PROMPT    — optional, defaults to a general assistant prompt
//!   RUST_LOG         — log level, e.g. "info" or "rustvani=debug,info"
//!
//! Run:
//!   SARVAM_API_KEY=your-key cargo run --release --bin websocket_server

use std::sync::Arc;

use axum::{
    Router,
    extract::{State, WebSocketUpgrade, ws::WebSocket},
    response::IntoResponse,
    routing::get,
};
use tower_http::cors::CorsLayer;

use rustvani::{
    shared_context, system_clock, SileroVad, VadParams,
    PipelineParams, PipelineTask,
};
use rustvani::processors::{
    llm_assistant_aggregator::LLMAssistantAggregator,
    llm_user_aggregator::LLMUserAggregator,
};
use rustvani::services::{
    SarvamLLMConfig, SarvamLLMHandler,
    SarvamSttConfig, SarvamSttHandler,
    SarvamTtsConfig, SarvamTtsHandler,
};
use rustvani::transport::websocket::{WebSocketParams, WebSocketTransport};
use rustvani::transport::TransportParams;

// ---------------------------------------------------------------------------
// Connection ID counter
// ---------------------------------------------------------------------------

static CONN_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_conn_id() -> u64 {
    CONN_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Shared app state
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AppState {
    sarvam_api_key: String,
    system_prompt:  String,
}

// ---------------------------------------------------------------------------
// WebSocket upgrade handler
// ---------------------------------------------------------------------------

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_connection(socket, state))
}

async fn handle_connection(socket: WebSocket, app_state: AppState) {
    let conn_id = next_conn_id();
    log::info!("[conn={}] connected", conn_id);

    // ---- VAD ----
    let vad_analyzer = match SileroVad::new(16_000) {
        Ok(v) => Arc::new(v),
        Err(e) => {
            log::error!("[conn={}] VAD init failed: {}", conn_id, e);
            return;
        }
    };

    // ---- Transport ----
    let transport = WebSocketTransport::new(
        &format!("WsTransport-{}", conn_id),
        WebSocketParams {
            transport: TransportParams {
                audio_in_enabled:         true,
                audio_in_sample_rate:     Some(16_000),
                audio_in_channels:        1,
                audio_in_passthrough:     true,
                audio_in_stream_on_start: true,
                vad_analyzer:             Some(vad_analyzer),
                vad_params:               VadParams::default(),
                ..TransportParams::default()
            },
        },
    );

    // ---- Shared conversation context ----
    let context = shared_context(Some(app_state.system_prompt.clone()));

    // ---- Pipeline processors ----

    let stt = SarvamSttHandler::new(SarvamSttConfig {
        api_key:  app_state.sarvam_api_key.clone(),
        model:    "saaras:v3".to_string(),
        language: Some("en-IN".to_string()),
        mode:     Some("transcribe".to_string()),
        ..SarvamSttConfig::default()
    })
    .into_processor();

    let user_agg = LLMUserAggregator::new(context.clone());

    let llm = SarvamLLMHandler::new(SarvamLLMConfig {
        api_key: app_state.sarvam_api_key.clone(),
        model:   "sarvam-30b".to_string(),
        ..SarvamLLMConfig::default()
    })
    .into_processor();

    let assistant_agg = LLMAssistantAggregator::new(context.clone());

    let tts = match SarvamTtsHandler::new(SarvamTtsConfig {
        api_key:  app_state.sarvam_api_key.clone(),
        model:    "bulbul:v2".to_string(),
        voice:    "anushka".to_string(),
        language: "en-IN".to_string(),
        ..SarvamTtsConfig::default()
    }) {
        Ok(t) => t.into_processor(),
        Err(e) => {
            log::error!("[conn={}] TTS init failed: {}", conn_id, e);
            return;
        }
    };

    // ---- Pipeline ----
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
        PipelineParams::default(),
    );

    let push_tx = task.push_sender();

    // Run pipeline and socket loop concurrently.
    // Either side terminating (socket close or EndFrame) shuts both down.
    tokio::join!(
        async { task.run(system_clock(), None).await.ok(); },
        transport.run_socket(socket, push_tx),
    );

    log::info!("[conn={}] disconnected", conn_id);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
    .init();

    let sarvam_api_key = std::env::var("SARVAM_API_KEY")
        .expect("SARVAM_API_KEY env var not set");

    let system_prompt = std::env::var("SYSTEM_PROMPT").unwrap_or_else(|_| {
        "You are a helpful voice assistant. \
         Keep your answers concise and conversational — \
         one or two sentences unless the user asks for more detail."
            .to_string()
    });

    let app_state = AppState { sarvam_api_key, system_prompt };

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    let addr = "0.0.0.0:8080";
    log::info!("rustvani voice agent listening on ws://{}/ws", addr);
    log::info!("Pipeline: audio → VAD → STT → LLM → TTS → audio");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}