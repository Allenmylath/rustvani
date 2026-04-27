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

pub struct WebSocketTransport {
    base:         Arc<BaseTransport>,
    audio_out_rx: std::sync::Mutex<Option<mpsc::Receiver<OutputMessage>>>,
}

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
    /// Arm 1 — `socket.recv()`: incoming messages from the client.
    ///
    ///   - Binary  → raw PCM audio → pipeline via `push_audio_frame`.
    ///   - Text JSON:
    ///       • `{"label":"rtvi-ai", ...}` → parsed as a RAVI inbound message
    ///         and pushed into the pipeline as `RaviClientMessage`.
    ///       • `{"type":"client_interruption"}` → `InterruptionFrame` downstream.
    ///
    /// Arm 2 — `audio_out_rx.recv()`: outgoing pipeline messages.
    ///
    ///   - `Audio(bytes)` → binary WebSocket frame.
    ///   - `Text(json)` → text WebSocket frame (RAVI protocol messages).
    ///   - `Interruption` → drain stale audio, then send JSON clear marker.
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
                // Arm 1: incoming messages → pipeline
                // ----------------------------------------------------------------
                msg = socket.recv() => {
                    match msg {
                        Some(Ok(Message::Binary(bytes))) => {
                            let data = AudioRawData::new(bytes.to_vec(), 16_000, 1);
                            base.push_audio_frame(data).await;
                        }

                        Some(Ok(Message::Text(text))) => {
                            handle_incoming_text(&text, &push_tx).await;
                        }

                        Some(Ok(Message::Close(_))) | None => {
                            log::debug!("WebSocketTransport: client closed connection");
                            break;
                        }

                        Some(Ok(_)) => {} // ping / pong — ignore

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
                                log::warn!("WebSocketTransport: failed to send audio");
                                break;
                            }
                        }

                        Some(OutputMessage::Text(json)) => {
                            // RAVI protocol messages (bot-ready, transcriptions, etc.)
                            if socket.send(Message::Text(json.into())).await.is_err() {
                                log::warn!("WebSocketTransport: failed to send RAVI text message");
                                break;
                            }
                        }

                        Some(OutputMessage::Interruption) => {
                            // Drain stale audio chunks queued before the marker.
                            while let Ok(queued) = audio_out_rx.try_recv() {
                                match queued {
                                    OutputMessage::Audio(_) => {}    // discard
                                    OutputMessage::Interruption => break,
                                    OutputMessage::Text(_) => {}     // discard (shouldn't queue during interruption)
                                }
                            }

                            let json = r#"{"type":"interruption"}"#;
                            if socket.send(Message::Text(json.into())).await.is_err() {
                                log::warn!("WebSocketTransport: failed to send interruption");
                                break;
                            }
                            log::debug!("WebSocketTransport: sent interruption to client");
                        }

                        None => break, // pipeline shut down
                    }
                }
            }
        }

        let _ = push_tx
            .send((Frame::end(), FrameDirection::Downstream))
            .await;
    }
}

// ---------------------------------------------------------------------------
// Incoming text message handler
// ---------------------------------------------------------------------------

/// Parse an incoming text WebSocket message and push the appropriate frame.
///
/// Two protocols are recognised:
///
/// 1. **RAVI** (`label == "rtvi-ai"`) — parsed into a `RaviClientMessage`
///    frame and sent downstream. The `RaviProcessor` in the pipeline handles
///    the protocol logic.
///
/// 2. **Legacy interruption** (`type == "client_interruption"`) — kept for
///    backward-compatibility with clients that pre-date RAVI.
async fn handle_incoming_text(
    text: &str,
    push_tx: &mpsc::Sender<(Frame, FrameDirection)>,
) {
    let Ok(msg) = serde_json::from_str::<serde_json::Value>(text) else {
        log::warn!("WebSocketTransport: ignoring non-JSON text message");
        return;
    };

    let msg_type = msg.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let label    = msg.get("label").and_then(|v| v.as_str()).unwrap_or("");

    if label == "ravi" {
        // RAVI inbound message.
        let Some(msg_id) = msg.get("id").and_then(|v| v.as_str()) else {
            log::warn!("WebSocketTransport: RAVI message missing 'id' field — dropping");
            return;
        };

        // `data` is optional; serialise it back to a JSON string so the
        // RaviProcessor can deserialise it into the appropriate type.
        let data_str = msg.get("data").map(|d| d.to_string());

        let frame = Frame::ravi_client_message(msg_id, msg_type, data_str);
        let _ = push_tx.send((frame, FrameDirection::Downstream)).await;

        log::trace!("WebSocketTransport: RAVI '{}' (id={})", msg_type, msg_id);
        return;
    }

    // Legacy: bare client interruption without RAVI label.
    if msg_type == "client_interruption" {
        log::info!("WebSocketTransport: legacy client-initiated interruption");
        let _ = push_tx
            .send((Frame::interruption(), FrameDirection::Downstream))
            .await;
    }
}
