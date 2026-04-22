//! rustvani WebSocket voice agent server.
//!
//! Listens on ws://0.0.0.0:8080/ws
//!
//! Each connection gets a fully isolated pipeline:
//!   WebSocketTransport.input()
//!     → SarvamStt
//!     → LLMUserAggregator
//!     → OpenAILLM
//!     → LLMAssistantAggregator
//!     → SarvamTts
//!     → WebSocketTransport.output()
//!
//! LatencyObserver logs every frame (except InputAudioRaw/OutputAudioRaw)
//! with processor name, plus per-turn stage deltas:
//!   t0  VADUserStoppedSpeaking  — user finished speaking
//!   t1  Transcription           — STT returned
//!   t2  LLMFullResponseStart    — first LLM token
//!   t3  LLMFullResponseEnd      — LLM stream complete
//!   t4  OutputAudioRaw (first)  — TTS first audio chunk out
//!
//! Wire protocol:
//!   Client → Server : Binary WebSocket — raw i16 LE PCM, 16 kHz mono, 512-sample chunks
//!   Server → Client : Binary WebSocket — raw i16 LE PCM at TTS sample rate (22050 Hz bulbul:v2)
//!
//! Environment variables:
//!   SARVAM_API_KEY   — required (STT + TTS)
//!   OPENAI_API_KEY   — required (LLM)
//!   SYSTEM_PROMPT    — optional
//!   RUST_LOG         — e.g. "info" or "rustvani=debug,info"
//!
//! Run:
//!   SARVAM_API_KEY=your-key OPENAI_API_KEY=your-key cargo run --release --bin websocket_server

use std::sync::Arc;
use std::sync::Mutex;

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
    PipelineParams, PipelineTask, FrameKind,
};
use rustvani::observer::{BaseObserver, FrameProcessed, FramePushed};
use rustvani::processors::{
    llm_assistant_aggregator::LLMAssistantAggregator,
    llm_user_aggregator::LLMUserAggregator,
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
// LatencyObserver — inline, one instance per connection
// ---------------------------------------------------------------------------

struct TurnState {
    turn:        u32,
    in_turn:     bool,
    t_vad:       f64,
    t_stt:       Option<f64>,
    t_llm_start: Option<f64>,
    t_llm_end:   Option<f64>,
    tts_first:   bool,
}

impl TurnState {
    fn new() -> Self {
        Self {
            turn:        0,
            in_turn:     false,
            t_vad:       0.0,
            t_stt:       None,
            t_llm_start: None,
            t_llm_end:   None,
            tts_first:   false,
        }
    }
}

struct LatencyObserver {
    conn_id: u64,
    state:   Mutex<TurnState>,
}

impl LatencyObserver {
    fn new(conn_id: u64) -> Self {
        Self { conn_id, state: Mutex::new(TurnState::new()) }
    }
}

#[async_trait]
impl BaseObserver for LatencyObserver {
    async fn on_process_frame(&self, event: FrameProcessed) {
        let ts  = event.timestamp;
        let cid = self.conn_id;

        // ---- Full frame log (skip audio flood) ----
        match event.frame.kind() {
            FrameKind::InputAudioRaw | FrameKind::OutputAudioRaw => {}
            _ => {
                log::info!(
                    "[conn={}] [{:.3}] {:>40}  @  {}",
                    cid, ts,
                    event.frame.name(),
                    event.processor_name,
                );
            }
        }

        // ---- Latency tracking ----
        let mut s = self.state.lock().unwrap();

        match event.frame.kind() {
            // t0 — user stopped speaking
            FrameKind::VADUserStoppedSpeaking => {
                if !s.in_turn {
                    s.turn       += 1;
                    s.in_turn     = true;
                    s.t_vad       = ts;
                    s.t_stt       = None;
                    s.t_llm_start = None;
                    s.t_llm_end   = None;
                    s.tts_first   = false;
                    log::info!(
                        "[conn={}] [turn={}] ┌ t0  VAD stop",
                        cid, s.turn
                    );
                }
            }

            // t1 — STT transcript
            FrameKind::Transcription => {
                if s.in_turn && s.t_stt.is_none() {
                    s.t_stt = Some(ts);
                    log::info!(
                        "[conn={}] [turn={}] │ t1  STT transcript   {:>+7.3}s",
                        cid, s.turn,
                        ts - s.t_vad
                    );
                }
            }

            // t2 — LLM first token
            FrameKind::LLMFullResponseStart => {
                if s.in_turn && s.t_llm_start.is_none() {
                    s.t_llm_start = Some(ts);
                    let from = s.t_stt.unwrap_or(s.t_vad);
                    log::info!(
                        "[conn={}] [turn={}] │ t2  LLM first token  {:>+7.3}s",
                        cid, s.turn,
                        ts - from
                    );
                }
            }

            // t3 — LLM stream complete
            FrameKind::LLMFullResponseEnd => {
                if s.in_turn && s.t_llm_end.is_none() {
                    s.t_llm_end = Some(ts);
                    let from = s.t_llm_start.unwrap_or(s.t_vad);
                    log::info!(
                        "[conn={}] [turn={}] │ t3  LLM complete     {:>+7.3}s",
                        cid, s.turn,
                        ts - from
                    );
                }
            }

            // t4 — TTS first audio chunk
            FrameKind::OutputAudioRaw => {
                if s.in_turn && !s.tts_first {
                    s.tts_first = true;
                    let from    = s.t_llm_end.unwrap_or(s.t_vad);
                    let total   = ts - s.t_vad;
                    log::info!(
                        "[conn={}] [turn={}] │ t4  TTS first audio  {:>+7.3}s",
                        cid, s.turn,
                        ts - from
                    );
                    log::info!(
                        "[conn={}] [turn={}] └ TOTAL               {:>+7.3}s",
                        cid, s.turn, total
                    );
                    s.in_turn = false;
                }
            }

            _ => {}
        }
    }

    async fn on_push_frame(&self, _event: FramePushed) {}
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

    let llm = OpenAILLMHandler::new(OpenAILLMConfig {
        api_key: app_state.openai_api_key.clone(),
        model:   "gpt-4o-mini".to_string(),
        ..OpenAILLMConfig::default()
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

    let push_tx  = task.push_sender();
    let observer = Arc::new(LatencyObserver::new(conn_id));

    tokio::join!(
        async { task.run(system_clock(), Some(observer)).await.ok(); },
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

    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let addr = format!("0.0.0.0:{}", port);
    log::info!("rustvani voice agent listening on ws://{}/ws", addr);
    log::info!("Pipeline: audio → VAD → STT(Sarvam) → LLM(OpenAI) → TTS(Sarvam) → audio");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}