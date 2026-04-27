//! rustvani RAVI WebSocket voice agent server.
//!
//! Listens on ws://0.0.0.0:{PORT}/ws  (PORT env var, default 10000 for Render)
//!
//! Each connection gets a fully isolated pipeline:
//!   WebSocketTransport.input()
//!     → RaviProcessor          (client-ready / disconnect-bot / send-text)
//!     → SarvamStt
//!     → LLMUserAggregator
//!     → OpenAILLM
//!     → LLMAssistantAggregator
//!     → SarvamTts
//!     → WebSocketTransport.output()
//!
//! RaviObserver watches the pipeline and emits RAVI protocol events back to
//! the client (bot-ready, speaking events, transcriptions, LLM start/stop).
//!
//! Wire protocol:
//!   Client → Server : Binary  — raw i16 LE PCM, 16 kHz mono, 512-sample chunks
//!                     Text    — RAVI JSON messages (label == "ravi")
//!   Server → Client : Binary  — raw i16 LE PCM at TTS sample rate
//!                     Text    — RAVI JSON protocol events
//!
//! Environment variables:
//!   PORT             — listen port (default: 10000, Render default)
//!   SARVAM_API_KEY   — required (STT + TTS)
//!   OPENAI_API_KEY   — required (LLM)
//!   SYSTEM_PROMPT    — optional

use std::sync::Arc;

use async_trait::async_trait;
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
use rustvani::observer::{BaseObserver, FrameProcessed, FramePushed};
use rustvani::processors::{
    llm_assistant_aggregator::LLMAssistantAggregator,
    llm_user_aggregator::LLMUserAggregator,
};
use rustvani::ravi::{
    RaviObserver, RaviObserverParams,
    processor::{RaviParams, RaviProcessor},
};
use rustvani::services::{
    OpenAILLMConfig, OpenAILLMHandler,
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
    openai_api_key: String,
    system_prompt:  String,
}

// ---------------------------------------------------------------------------
// NullObserver — RaviObserver handles all client-facing events;
// this satisfies the PipelineTask observer slot when we want nothing extra.
// ---------------------------------------------------------------------------

struct NullObserver;

#[async_trait]
impl BaseObserver for NullObserver {
    async fn on_process_frame(&self, _: FrameProcessed) {}
    async fn on_push_frame(&self, _: FramePushed) {}
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
                vad_params:               VadParams {
                    confidence: 0.4,
                    min_volume: 0.1,
                    ..VadParams::default()
                },
                ..TransportParams::default()
            },
        },
    );

    // ---- Shared conversation context ----
    let context = shared_context(Some(app_state.system_prompt.clone()));

    // ---- Pipeline processors ----

    let ravi = RaviProcessor::new(RaviParams {
        context: Some(context.clone()),
        ..RaviParams::default()
    });

    // RaviObserver watches the pipeline and pushes RAVI events back to client
    // through the ravi processor → output transport → WS text frames.
    let ravi_observer: Arc<dyn BaseObserver> = Arc::new(
        RaviProcessor::create_observer(&ravi, RaviObserverParams::default()),
    );

    let stt = SarvamSttHandler::new(SarvamSttConfig {
        api_key:  app_state.sarvam_api_key.clone(),
        model:    "saaras:v3".to_string(),
        language: Some("en-IN".to_string()),
        mode:     Some("transcribe".to_string()),
        ..SarvamSttConfig::default()
    })
    .into_processor();

    let user_agg = LLMUserAggregator::new(context.clone());

    let llm = OpenAILLMHandler::new(OpenAILLMConfig {
        api_key: app_state.openai_api_key.clone(),
        model:   "gpt-4o-mini".to_string(),
        ..OpenAILLMConfig::default()
    })
    .into_processor();

    let assistant_agg = LLMAssistantAggregator::new(context.clone());

    let tts = match SarvamTtsHandler::new(SarvamTtsConfig {
        api_key:  app_state.sarvam_api_key.clone(),
        model:    "bulbul:v3".to_string(),
        voice:    "aditya".to_string(),
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
            ravi,
            stt,
            user_agg,
            llm,
            assistant_agg,
            tts,
            transport.output(),
        ],
        PipelineParams { allow_interruptions: true, ..PipelineParams::default() },
    );

    let push_tx = task.push_sender();

    tokio::join!(
        async { task.run(system_clock(), Some(ravi_observer)).await.ok(); },
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

    let openai_api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY env var not set");

    let system_prompt = std::env::var("SYSTEM_PROMPT").unwrap_or_else(|_| {
        "You are a helpful voice assistant. \
         Keep your answers concise and conversational — \
         one or two sentences unless the user asks for more detail."
            .to_string()
    });

    let app_state = AppState { sarvam_api_key, openai_api_key, system_prompt };

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    let port = std::env::var("PORT").unwrap_or_else(|_| "10000".to_string());
    let addr = format!("0.0.0.0:{}", port);

    log::info!("rustvani RAVI voice agent listening on ws://{}/ws", addr);
    log::info!("Pipeline: audio → VAD → RaviProcessor → STT(Sarvam) → LLM(OpenAI) → TTS(Sarvam) → audio");

    let listener = tokio::net::TcpListener::bind(&addr).await
        .unwrap_or_else(|e| panic!("Failed to bind {}: {}", addr, e));

    axum::serve(listener, app).await.unwrap();
}
