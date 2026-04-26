use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket};
use tokio::sync::mpsc;

use crate::frames::{AudioRawData, Frame, FrameDirection, FrameProcessor};
use crate::transport::{BaseTransport, TransportParams};
use crate::transport::output::OutputMessage;

// ---------------------------------------------------------------------------
// WebSocketParams
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct WebSocketParams {
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

/// WebSocket transport — wraps BaseTransport with bidirectional audio.
///
/// Each connection should create a new instance so that VAD state, audio
/// buffers, and pipeline resources are fully isolated.
pub struct WebSocketTransport {
    base:         Arc<BaseTransport>,
    audio_out_rx: std::sync::Mutex<Option<mpsc::Receiver<OutputMessage>>>,
}

/// Channel capacity sized for chunked audio.
///
/// With 40 ms chunks (audio_out_10ms_chunks=4) a 5-second TTS response
/// produces ~125 chunks.  Cap = 150 gives comfortable headroom.
const AUDIO_OUT_CHANNEL_CAP: usize = 150;

impl WebSocketTransport {
    pub fn new(name: &str, params: WebSocketParams) -> Self {
        let base = Arc::new(BaseTransport::new(name, params.transport));

        let (audio_out_tx, audio_out_rx) = mpsc::channel::<OutputMessage>(AUDIO_OUT_CHANNEL_CAP);
        base.set_audio_out_tx(audio_out_tx);

        Self {
            base,
            audio_out_rx: std::sync::Mutex::new(Some(audio_out_rx)),
        }
    }

    pub fn input(&self) -> FrameProcessor {
        self.base.input()
    }

    pub fn output(&self) -> FrameProcessor {
        self.base.output()
    }

    /// Drive the WebSocket connection until it closes.
    ///
    /// Simple two-arm select:
    ///
    ///   Arm 1 — `socket.recv()`: incoming user audio → pipeline via
    ///            `push_audio_frame`. Always active.
    ///
    ///   Arm 2 — `audio_out_rx.recv()`: outgoing pipeline messages.
    ///            `Audio(bytes)` → send binary frame immediately.
    ///            `Interruption` → drain stale audio from channel,
    ///            then send JSON text frame to client.
    pub async fn run_socket(
        &self,
        mut socket: WebSocket,
        push_tx: mpsc::Sender<(Frame, FrameDirection)>,
    ) {
        let mut audio_out_rx = self
            .audio_out_rx
            .lock()
            .unwrap()
            .take()
            .expect("run_socket called more than once on the same WebSocketTransport");

        let base = self.base.clone();

        loop {
            tokio::select! {
                // ----------------------------------------------------------------
                // Arm 1: incoming user audio → pipeline
                // ----------------------------------------------------------------
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

                // ----------------------------------------------------------------
                // Arm 2: outgoing pipeline messages → client
                // ----------------------------------------------------------------
                output_msg = audio_out_rx.recv() => {
                    match output_msg {
                        Some(OutputMessage::Audio(bytes)) => {
                            if socket.send(Message::Binary(bytes.into())).await.is_err() {
                                log::warn!("WebSocketTransport: failed to send audio to client");
                                break;
                            }
                        }
                        Some(OutputMessage::Interruption) => {
                            // Drain any stale audio chunks queued before
                            // the interruption marker.
                            while let Ok(msg) = audio_out_rx.try_recv() {
                                match msg {
                                    OutputMessage::Audio(_) => {} // discard
                                    OutputMessage::Interruption => break,
                                }
                            }

                            // Tell client to flush its playback buffer.
                            let json = r#"{"type":"interruption"}"#;
                            if socket.send(Message::Text(json.into())).await.is_err() {
                                log::warn!("WebSocketTransport: failed to send interruption to client");
                                break;
                            }
                            log::debug!("WebSocketTransport: sent interruption to client");
                        }
                        None => {
                            // Pipeline shut down.
                            break;
                        }
                    }
                }
            }
        }

        // Signal pipeline shutdown.
        let _ = push_tx
            .send((Frame::end(), FrameDirection::Downstream))
            .await;
    }
}
