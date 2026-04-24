use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use axum::extract::ws::{Message, WebSocket};
use tokio::sync::mpsc;

use crate::frames::{AudioRawData, Frame, FrameDirection, FrameProcessor};
use crate::transport::{BaseTransport, TransportParams};

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
    audio_out_rx: std::sync::Mutex<Option<mpsc::Receiver<Vec<u8>>>>,
    /// Shared with BaseOutputTransport. When true, run_socket drops outgoing
    /// audio chunks so the user hears the bot stop immediately on interruption.
    mute_gate:    Arc<AtomicBool>,
}

/// Channel capacity sized for 10 ms chunked audio.
///
/// With 10 ms chunks the output transport pushes many small messages
/// instead of a few large blobs.  At 24 kHz mono (480 B/chunk) a
/// 5-second TTS response produces ~500 chunks.  Cap = 500 gives
/// 5 s of buffering headroom — about 240 KB of memory.
const AUDIO_OUT_CHANNEL_CAP: usize = 500;

impl WebSocketTransport {
    pub fn new(name: &str, params: WebSocketParams) -> Self {
        let base = Arc::new(BaseTransport::new(name, params.transport));

        let (audio_out_tx, audio_out_rx) = mpsc::channel::<Vec<u8>>(AUDIO_OUT_CHANNEL_CAP);
        base.set_audio_out_tx(audio_out_tx);

        // Get the mute gate from the output transport so run_socket can share it.
        let mute_gate = base.mute_gate();

        Self {
            base,
            audio_out_rx: std::sync::Mutex::new(Some(audio_out_rx)),
            mute_gate,
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
    /// Two arms:
    ///   - Incoming binary frames from client → push_audio_frame (VAD/STT path)
    ///   - Outgoing audio bytes from pipeline → send to client (TTS path)
    ///     but ONLY when mute_gate is false — stale chunks after an interruption
    ///     are dropped silently so the user hears the bot stop immediately.
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

        let base      = self.base.clone();
        let mute_gate = self.mute_gate.clone();

        loop {
            tokio::select! {
                // ---- Incoming audio: client → pipeline (STT path) ----
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

                // ---- Outgoing audio: pipeline → client (TTS path) ----
                audio_bytes = audio_out_rx.recv() => {
                    match audio_bytes {
                        Some(bytes) => {
                            // Drop stale chunks while muted. The mute gate is
                            // set true by BaseOutputTransport on InterruptionFrame
                            // and cleared automatically on the first new
                            // OutputAudioRaw after the bot starts speaking again.
                            if mute_gate.load(Ordering::Relaxed) {
                                continue;
                            }
                            if socket.send(Message::Binary(bytes.into())).await.is_err() {
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