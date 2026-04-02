//! axum WebSocket transport for rustvani.
//!
//! Wraps `BaseTransport` with an axum WebSocket handler.
//! Each connection calls `transport.run_socket()` which reads binary PCM
//! from the WebSocket and feeds `push_audio_frame()`, then sends `Frame::end()`
//! immediately when the socket closes.
//!
//! # Usage
//!
//! ```text
//! let transport = WebSocketTransport::new("MyTransport", WebSocketParams::default());
//!
//! let task = PipelineTask::new(
//!     vec![transport.input(), vad, transport.output()],
//!     PipelineParams::default(),
//! );
//!
//! let push_tx = task.push_sender();
//!
//! tokio::join!(
//!     task.run(system_clock(), None),
//!     transport.run_socket(socket, push_tx),
//! );
//! ```

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket};
use tokio::sync::mpsc;

use crate::frames::{AudioRawData, Frame, FrameDirection, FrameProcessor};
use crate::transport::{BaseTransport, TransportParams};

// ---------------------------------------------------------------------------
// WebSocketParams
// ---------------------------------------------------------------------------

/// Configuration for the WebSocket transport.
#[derive(Debug, Clone)]
pub struct WebSocketParams {
    /// Underlying transport configuration.
    pub transport: TransportParams,
}

impl Default for WebSocketParams {
    fn default() -> Self {
        Self {
            transport: TransportParams {
                audio_in_enabled:         true,
                audio_in_sample_rate:     Some(16_000),
                audio_in_channels:        1,
                audio_in_passthrough:     true,
                audio_in_stream_on_start: true,
                ..TransportParams::default()
            },
        }
    }
}

// ---------------------------------------------------------------------------
// WebSocketTransport
// ---------------------------------------------------------------------------

/// WebSocket transport — wraps `BaseTransport` with axum WebSocket handling.
///
/// Each connection should create a new `WebSocketTransport` instance so that
/// VAD state, audio buffers, and pipeline resources are fully isolated.
pub struct WebSocketTransport {
    base: Arc<BaseTransport>,
}

impl WebSocketTransport {
    /// Create a new WebSocket transport with the given name and params.
    pub fn new(name: &str, params: WebSocketParams) -> Self {
        let base = Arc::new(BaseTransport::new(name, params.transport));
        Self { base }
    }

    /// The input `FrameProcessor` — place first in the pipeline.
    pub fn input(&self) -> FrameProcessor {
        self.base.input()
    }

    /// The output `FrameProcessor` — place last in the pipeline.
    pub fn output(&self) -> FrameProcessor {
        self.base.output()
    }

    /// Drive the WebSocket connection until it closes.
    ///
    /// Reads binary messages from `socket`, feeds them to `push_audio_frame()`.
    /// When the socket closes (or errors), sends `Frame::end()` immediately
    /// through `push_tx` to terminate the pipeline.
    ///
    /// Call this concurrently with `PipelineTask::run()`:
    /// ```text
    /// tokio::join!(task.run(...), transport.run_socket(socket, push_tx));
    /// ```
    pub async fn run_socket(
        &self,
        mut socket: WebSocket,
        push_tx: mpsc::Sender<(Frame, FrameDirection)>,
    ) {
        let base = self.base.clone();

        loop {
            match socket.recv().await {
                Some(Ok(Message::Binary(bytes))) => {
                    let data = AudioRawData::new(bytes.to_vec(), 16_000, 1);
                    base.push_audio_frame(data).await;
                }
                Some(Ok(Message::Close(_))) | None => {
                    break;
                }
                Some(Ok(_)) => {} // ignore text / ping / pong
                Some(Err(e)) => {
                    log::warn!("WebSocketTransport: socket error: {}", e);
                    break;
                }
            }
        }

        // Socket closed — terminate the pipeline immediately.
        // The audio task and its internal mpsc will be cleaned up
        // by pipeline cleanup() via cancel_input_task().
        let _ = push_tx
            .send((Frame::end(), FrameDirection::Downstream))
            .await;
    }
}