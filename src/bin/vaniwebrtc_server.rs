//! rustvani peer-to-peer WebRTC voice agent server.
//!
//! Signaling endpoint: ws://0.0.0.0:8080/rtc  (SDP offer/answer + trickle ICE)
//! Media: Opus/RTP/SRTP carried peer-to-peer (no SFU).
//!
//! Each connection gets a fully isolated pipeline, identical to the WebSocket
//! server:
//!   VaniWebRTCTransport.input()
//!     → SarvamStt → LLMUserAggregator → OpenAILLM
//!     → LLMAssistantAggregator → SarvamTts → VaniWebRTCTransport.output()
//!
//! Build/run (opt-in feature; needs a C toolchain for audiopus/libopus):
//!   cargo run --bin vaniwebrtc_server --features vaniwebrtc
//!
//! Environment variables:
//!   SARVAM_API_KEY   — required (STT + TTS)
//!   OPENAI_API_KEY   — required (LLM)
//!   SYSTEM_PROMPT    — optional
//!   PORT             — optional (default 8080)
//!   RUST_LOG         — e.g. "info" or "rustvani=debug,info"

use std::sync::Arc;

use axum::{
    extract::{ws::WebSocket, State, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
    Router,
};
use tower_http::cors::CorsLayer;

use rustvani::{
    shared_context, system_clock, PipelineParams, PipelineTask, SileroVadNative, VadParams,
};
use rustvani::processors::{
    llm_assistant_aggregator::LLMAssistantAggregator,
    llm_user_aggregator::LLMUserAggregator,
};
use rustvani::services::{
    OpenAILLMConfig, OpenAILLMHandler, SarvamSttConfig, SarvamSttHandler, SarvamTtsConfig,
    SarvamTtsHandler,
};
use rustvani::transport::{TransportParams, VaniWebRTCParams, VaniWebRTCTransport};

static CONN_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_conn_id() -> u64 {
    CONN_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

#[derive(Clone)]
struct AppState {
    sarvam_api_key: String,
    openai_api_key: String,
    system_prompt:  String,
}

async fn rtc_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_connection(socket, state))
}

async fn handle_connection(socket: WebSocket, app_state: AppState) {
    let conn_id = next_conn_id();
    log::info!("[conn={}] connected", conn_id);

    // ---- VAD ----
    let vad_analyzer = match SileroVadNative::new(16_000) {
        Ok(v) => Arc::new(v),
        Err(e) => {
            log::error!("[conn={}] VAD init failed: {}", conn_id, e);
            return;
        }
    };

    // ---- Transport (P2P WebRTC) ----
    let transport = VaniWebRTCTransport::new(
        &format!("VaniWebRTC-{}", conn_id),
        VaniWebRTCParams {
            transport: TransportParams {
                audio_in_enabled:         true,
                audio_in_sample_rate:     Some(16_000),
                audio_in_channels:        1,
                audio_in_passthrough:     true,
                audio_in_stream_on_start: true,
                audio_out_enabled:        true,
                audio_out_sample_rate:    Some(24_000), // Sarvam bulbul:v3 output rate
                audio_out_channels:       1,
                vad_analyzer:             Some(vad_analyzer),
                vad_params:               VadParams {
                    confidence: 0.4,
                    min_volume: 0.1,
                    ..VadParams::default()
                },
                ..TransportParams::default()
            },
            ..VaniWebRTCParams::default()
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
        async { task.run(system_clock(), None).await.ok(); },
        transport.run(socket, push_tx),
    );

    log::info!("[conn={}] disconnected", conn_id);
}

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let sarvam_api_key = std::env::var("SARVAM_API_KEY").expect("SARVAM_API_KEY env var not set");
    let openai_api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY env var not set");
    let system_prompt = std::env::var("SYSTEM_PROMPT").unwrap_or_else(|_| {
        "You are a helpful voice assistant. Keep your answers concise and \
         conversational — one or two sentences unless the user asks for more detail."
            .to_string()
    });

    let app_state = AppState { sarvam_api_key, openai_api_key, system_prompt };

    let app = Router::new()
        .route("/rtc", get(rtc_handler))
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let addr = format!("0.0.0.0:{}", port);
    log::info!("rustvani P2P WebRTC voice agent — signaling on ws://{}/rtc", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
