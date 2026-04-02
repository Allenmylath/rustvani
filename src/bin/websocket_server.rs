//! rustvani WebSocket server.
//!
//! Listens on ws://localhost:8080/ws
//! Each connection gets a fully isolated pipeline:
//!   WebSocketTransport.input() → VadPrintProcessor → WebSocketTransport.output()
//!
//! VAD runs inside the transport's audio task — configure via WebSocketParams.
//!
//! Wire protocol (client → server):
//!   Binary WebSocket messages — raw i16 LE PCM, 16kHz mono, 512-sample chunks
//!
//! Run: cargo run --release --bin websocket_server

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use axum::{
    Router,
    extract::{WebSocketUpgrade, ws::WebSocket},
    response::IntoResponse,
    routing::get,
};
use tower_http::cors::CorsLayer;

use rustvani::{
    system_clock, Frame, FrameDirection, FrameHandler, FrameKind,
    FrameProcessor, PipelineParams, PipelineTask, Result, VadParams, SileroVad,
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
// VadPrintHandler — prints only VAD events with connection ID
// ---------------------------------------------------------------------------

struct VadPrintHandler {
    conn_id: u64,
}

#[async_trait]
impl FrameHandler for VadPrintHandler {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match frame.kind() {
            FrameKind::VADUserStartedSpeaking | FrameKind::VADUserStoppedSpeaking => {
                let ts = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64();
                println!(
                    "[conn={}] ts={:.6}  {}",
                    self.conn_id,
                    ts,
                    frame.name(),
                );
            }
            _ => {}
        }
        processor.push_frame(frame, direction).await
    }
}

// ---------------------------------------------------------------------------
// WebSocket upgrade handler
// ---------------------------------------------------------------------------

async fn ws_handler(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(handle_connection)
}

async fn handle_connection(socket: WebSocket) {
    let conn_id = next_conn_id();
    println!("[conn={}] connected", conn_id);

    // Build a fresh Silero VAD instance per connection — isolated LSTM state.
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

    let printer = FrameProcessor::new(
        format!("VadPrint-{}", conn_id),
        Box::new(VadPrintHandler { conn_id }),
        false,
    );

    // Pipeline: VAD now runs inside transport.input() — no VadProcessor stage needed.
    let task = PipelineTask::new(
        vec![transport.input(), printer, transport.output()],
        PipelineParams::default(),
    );

    let push_tx = task.push_sender();

    tokio::join!(
        async { task.run(system_clock(), None).await.ok(); },
        transport.run_socket(socket, push_tx),
    );

    println!("[conn={}] pipeline shut down", conn_id);
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

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .layer(CorsLayer::permissive());

    let addr = "0.0.0.0:8080";
    println!("rustvani WebSocket server listening on ws://{}/ws", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}