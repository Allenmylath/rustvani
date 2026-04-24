use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

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

/// Send interval — one chunk every 5 ms (half of the 10 ms chunk duration).
/// Mirrors Python's `_send_interval = (chunk_size / sample_rate) / 2` logic.
/// Keeps the client buffer near-zero so interruption is felt immediately.
const SEND_INTERVAL: Duration = Duration::from_millis(5);

impl WebSocketTransport {
    pub fn new(name: &str, params: WebSocketParams) -> Self {
        let base = Arc::new(BaseTransport::new(name, params.transport));

        let (audio_out_tx, audio_out_rx) = mpsc::channel::<Vec<u8>>(AUDIO_OUT_CHANNEL_CAP);
        base.set_audio_out_tx(audio_out_tx);

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
    /// Three-arm select loop:
    ///
    ///   Arm 1 — `socket.recv()`: always active.
    ///            Incoming binary frames from the client are pushed into the
    ///            VAD/STT pipeline immediately regardless of send timing.
    ///
    ///   Arm 2 — `audio_out_rx.recv()`: active only when `pending_chunk` is empty.
    ///            Pulls the next 10 ms chunk from the pipeline into a staging slot.
    ///            While muted (interruption in progress) it drains the channel
    ///            silently and resets the send timer so the next bot turn starts clean.
    ///
    ///   Arm 3 — `sleep_until(next_send_time)`: active only when `pending_chunk` is Some.
    ///            Fires at the scheduled deadline and sends the staged chunk.
    ///            Advances the deadline by SEND_INTERVAL after each send.
    ///
    /// This design ensures the incoming-audio arm is never blocked by the
    /// outgoing pacing sleep, preserving VAD / interruption responsiveness.
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

        // Pacing state.
        // `pending_chunk`  — staged audio waiting for its send deadline.
        // `next_send_time` — when to fire arm 3. None means no active bot turn.
        let mut pending_chunk:  Option<Vec<u8>>              = None;
        let mut next_send_time: Option<tokio::time::Instant> = None;

        loop {
            tokio::select! {
                // ----------------------------------------------------------------
                // Arm 1: incoming user audio → pipeline (always active)
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
                // Arm 2: pull next chunk into the pending slot
                // Active only when the slot is empty — prevents skipping the timer.
                // ----------------------------------------------------------------
                audio_bytes = audio_out_rx.recv(), if pending_chunk.is_none() => {
                    match audio_bytes {
                        Some(bytes) => {
                            if mute_gate.load(Ordering::Relaxed) {
                                // Interruption active — drain silently.
                                // Reset timer so next bot turn starts with a
                                // fresh deadline rather than an inherited offset.
                                next_send_time = None;
                                continue;
                            }

                            pending_chunk = Some(bytes);

                            // Initialise deadline at the start of each bot turn.
                            if next_send_time.is_none() {
                                next_send_time =
                                    Some(tokio::time::Instant::now() + SEND_INTERVAL);
                            }
                        }
                        None => {
                            // Pipeline shut down — exit the loop.
                            break;
                        }
                    }
                }

                // ----------------------------------------------------------------
                // Arm 3: paced send — fires when the deadline arrives
                // Active only when there is a staged chunk to send.
                // `std::future::pending()` makes this arm dormant (no busy-spin)
                // when there is no active bot turn.
                // ----------------------------------------------------------------
                _ = async {
                    if let Some(t) = next_send_time {
                        tokio::time::sleep_until(t).await
                    } else {
                        std::future::pending::<()>().await
                    }
                }, if pending_chunk.is_some() => {
                    if let Some(chunk) = pending_chunk.take() {
                        if !mute_gate.load(Ordering::Relaxed) {
                            if socket.send(Message::Binary(chunk.into())).await.is_err() {
                                log::warn!("WebSocketTransport: failed to send audio to client");
                                break;
                            }
                        }
                        // Advance the deadline whether we sent or were muted,
                        // so timing stays consistent across an interrupted turn.
                        next_send_time = next_send_time.map(|t| t + SEND_INTERVAL);
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
