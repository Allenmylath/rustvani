//! rustvani WebSocket server with Sarvam STT.
//!
//! Listens on ws://0.0.0.0:8080/ws
//! Each connection gets a fully isolated pipeline:
//!   WebSocketTransport.input() → SarvamSttHandler → PrintProcessor → WebSocketTransport.output()
//!
//! VAD runs inside the transport's audio task.
//! SarvamStt opens its own WebSocket to Sarvam per connection.
//! PrintProcessor logs VAD frames and TranscriptionFrames to stdout.
//!
//! Wire protocol (client → server):
//!   Binary WebSocket messages — raw i16 LE PCM, 16kHz mono, 512-sample chunks
//!
//! Run:
//!   SARVAM_API_KEY=your-key cargo run --release --bin websocket_server

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use axum::{
    Router,
    extract::{State, WebSocketUpgrade, ws::WebSocket},
    response::IntoResponse,
    routing::get,
};
use tower_http::cors::CorsLayer;

use rustvani::{
    system_clock, DataFrame, Frame, FrameDirection, FrameHandler, FrameInner,
    FrameKind, FrameProcessor, PipelineParams, PipelineTask, Result, SileroVad,
    VadParams,
};
use rustvani::services::{SarvamSttConfig, SarvamSttHandler};
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
}

// ---------------------------------------------------------------------------
// PrintHandler — logs VAD frames and TranscriptionFrames with connection ID
// ---------------------------------------------------------------------------

struct PrintHandler {
    conn_id: u64,
}

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
                println!("[conn={}] [{:.3}] VADUserStartedSpeaking", self.conn_id, ts);
            }
            FrameKind::VADUserStoppedSpeaking => {
                println!("[conn={}] [{:.3}] VADUserStoppedSpeaking", self.conn_id, ts);
            }
            FrameKind::Transcription => {
                if let FrameInner::Data(DataFrame::Transcription(ref t)) = frame.inner {
                    println!(
                        "[conn={}] [{:.3}] Transcript  text=\"{}\"  lang={:?}  finalized={}",
                        self.conn_id, ts, t.text, t.language, t.finalized
                    );
                }
            }
            _ => {}
        }

        processor.push_frame(frame, direction).await
    }
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
    println!("[conn={}] connected", conn_id);

    // Fresh VAD per connection — isolated LSTM state.
    let vad_analyzer = match SileroVad::new(16_000) {
        Ok(v) => Arc::new(v),
        Err(e) => {
            println!("[conn={}] VAD init failed: {}", conn_id, e);
            return;
        }
    };

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

    // Sarvam STT — one WebSocket connection to Sarvam per client connection.
    let stt = SarvamSttHandler::new(SarvamSttConfig {
        api_key:  app_state.sarvam_api_key.clone(),
        model:    "saaras:v3".to_string(),
        language: Some("en-IN".to_string()),
        mode:     Some("transcribe".to_string()),
        ..SarvamSttConfig::default()
    })
    .into_processor();

    let printer = FrameProcessor::new(
        format!("Printer-{}", conn_id),
        Box::new(PrintHandler { conn_id }),
        false,
    );

    let task = PipelineTask::new(
        vec![transport.input(), stt, printer, transport.output()],
        PipelineParams::default(),
    );

    let push_tx = task.push_sender();

    tokio::join!(
        async { task.run(system_clock(), None).await.ok(); },
        transport.run_socket(socket, push_tx),
    );

    println!("[conn={}] disconnected", conn_id);
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

    let app_state = AppState { sarvam_api_key };

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    let addr = "0.0.0.0:8080";
    println!("rustvani WebSocket server listening on ws://{}/ws", addr);
    println!("Pipeline: audio → VAD → Sarvam STT → print");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}