//! rustvani WebSocket server.
//!
//! Listens on ws://localhost:8080/ws
//! Each connection gets a fully isolated pipeline:
//!   WebSocketTransport.input() → VadProcessor → VadPrintProcessor → WebSocketTransport.output()
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
    FrameProcessor, PipelineParams, PipelineTask, Result, VadParams, VadProcessor,
};
use rustvani::transport::websocket::{WebSocketParams, WebSocketTransport};

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

    // Fresh isolated transport + pipeline per connection.
    let transport = WebSocketTransport::new(
        &format!("WsTransport-{}", conn_id),
        WebSocketParams::default(),
    );

    let vad = match VadProcessor::new(16_000, VadParams::default()) {
        Ok(v) => v.into_processor(),
        Err(e) => {
            println!("[conn={}] VAD init failed: {}", conn_id, e);
            return;
        }
    };

    let printer = FrameProcessor::new(
        format!("VadPrint-{}", conn_id),
        Box::new(VadPrintHandler { conn_id }),
        false,
    );

    let task = PipelineTask::new(
        vec![transport.input(), vad, printer, transport.output()],
        PipelineParams::default(),
    );

    let push_tx = task.push_sender();

    // run_socket feeds audio into the pipeline and sends Frame::end() on close.
    // task.run() blocks until Frame::end() reaches TaskSink.
    // Both complete together.
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