//! axum WebSocket transport for rustvani.
//!
//! Wraps `BaseTransport` with an axum WebSocket handler.
//!
//! # Wire protocol
//!
//! Client → Server : Binary WebSocket frames — raw i16 LE PCM, 16 kHz mono, 512-sample chunks.
//! Server → Client : Binary WebSocket frames — raw i16 LE PCM at the TTS sample rate.
//!
//! # Audio out
//!
//! `WebSocketTransport::new` creates an `mpsc` channel and registers the sender
//! with `BaseOutputTransport` via `set_audio_out_tx`. When `OutputAudioRaw`
//! frames flow through the output processor, their bytes are forwarded to the
//! channel. `run_socket` selects on the channel receiver and writes each chunk
//! back to the WebSocket as a binary frame.
//!
//! # Usage
//!
//! ```text
//! let transport = WebSocketTransport::new("MyTransport", WebSocketParams::default());
//!
//! let task = PipelineTask::new(
//!     vec![transport.input(), stt, llm, tts, transport.output()],
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

/// WebSocket transport — wraps `BaseTransport` with bidirectional audio.
///
/// Each connection should create a new `WebSocketTransport` instance so that
/// VAD state, audio buffers, and pipeline resources are fully isolated.
pub struct WebSocketTransport {
    base:         Arc<BaseTransport>,
    /// Receiver end of the audio-out channel.
    /// Taken by `run_socket` — only valid to call once per instance.
    audio_out_rx: std::sync::Mutex<Option<mpsc::Receiver<Vec<u8>>>>,
}

/// Channel capacity for outgoing audio chunks.
/// At 22050 Hz, 512-sample chunks ≈ 23 ms each.  64 slots ≈ 1.5 s of buffer.
const AUDIO_OUT_CHANNEL_CAP: usize = 64;

impl WebSocketTransport {
    /// Create a new WebSocket transport.
    ///
    /// Internally creates the audio-out channel and wires the sender to
    /// `BaseOutputTransport` so `OutputAudioRaw` bytes reach `run_socket`.
    pub fn new(name: &str, params: WebSocketParams) -> Self {
        let base = Arc::new(BaseTransport::new(name, params.transport));

        // Audio-out channel: BaseOutputTransport → run_socket → WebSocket.
        let (audio_out_tx, audio_out_rx) = mpsc::channel::<Vec<u8>>(AUDIO_OUT_CHANNEL_CAP);
        base.set_audio_out_tx(audio_out_tx);

        Self {
            base,
            audio_out_rx: std::sync::Mutex::new(Some(audio_out_rx)),
        }
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
    /// Runs a `select!` loop over two arms:
    ///   - Incoming binary frames from the client → `push_audio_frame` (STT path).
    ///   - Outgoing audio bytes from the pipeline → binary frame back to client (TTS path).
    ///
    /// When the socket closes or errors, sends `Frame::end()` through `push_tx`
    /// to terminate the pipeline.
    ///
    /// Must be called concurrently with `PipelineTask::run()`:
    /// ```text
    /// tokio::join!(task.run(...), transport.run_socket(socket, push_tx));
    /// ```
    pub async fn run_socket(
        &self,
        mut socket: WebSocket,
        push_tx: mpsc::Sender<(Frame, FrameDirection)>,
    ) {
        // Take the receiver — this can only be called once per transport instance.
        let mut audio_out_rx = self
            .audio_out_rx
            .lock()
            .unwrap()
            .take()
            .expect("run_socket called more than once on the same WebSocketTransport");

        let base = self.base.clone();

        loop {
            tokio::select! {
                // ---- Incoming audio from client → pipeline (STT path) ----
                msg = socket.recv() => {
                    match msg {
                        Some(Ok(Message::Binary(bytes))) => {
                            let data = AudioRawData::new(bytes.to_vec(), 16_000, 1);
                            base.push_audio_frame(data).await;
                        }
                        Some(Ok(Message::Close(_))) | None => {
                            log::debug!("WebSocketTransport: client closed connection");
                            break;
                        }
                        Some(Ok(_)) => {} // text / ping / pong — ignore
                        Some(Err(e)) => {
                            log::warn!("WebSocketTransport: socket error: {}", e);
                            break;
                        }
                    }
                }

                // ---- Outgoing audio from pipeline → client (TTS path) ----
                audio_bytes = audio_out_rx.recv() => {
                    match audio_bytes {
                        Some(bytes) => {
                            if socket
                                .send(Message::Binary(bytes.into()))
                                .await
                                .is_err()
                            {
                                log::warn!("WebSocketTransport: failed to send audio to client");
                                break;
                            }
                        }
                        None => {
                            // Channel closed — pipeline has shut down.
                            break;
                        }
                    }
                }
            }
        }

        // Terminate the pipeline.
        let _ = push_tx
            .send((Frame::end(), FrameDirection::Downstream))
            .await;
    }
}