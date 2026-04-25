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
    /// Send interval — derived from `audio_out_10ms_chunks`.
    /// Mirrors Python's `_send_interval = (chunk_size / sample_rate) / 2`
    /// which simplifies to `(10ms × chunks) / 2`.
    send_interval: Duration,
}

/// Channel capacity sized for chunked audio.
///
/// With 40 ms chunks (audio_out_10ms_chunks=4) a 5-second TTS response
/// produces ~125 chunks.  Cap = 150 gives comfortable headroom.
const AUDIO_OUT_CHANNEL_CAP: usize = 150;

impl WebSocketTransport {
    pub fn new(name: &str, params: WebSocketParams) -> Self {
        let chunks = params.transport.audio_out_10ms_chunks.max(1) as u64;
        let send_interval = Duration::from_millis((10 * chunks) / 2);

        let base = Arc::new(BaseTransport::new(name, params.transport));

        let (audio_out_tx, audio_out_rx) = mpsc::channel::<Vec<u8>>(AUDIO_OUT_CHANNEL_CAP);
        base.set_audio_out_tx(audio_out_tx);

        let mute_gate = base.mute_gate();

        Self {
            base,
            audio_out_rx: std::sync::Mutex::new(Some(audio_out_rx)),
            mute_gate,
            send_interval,
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
    ///            Pulls the next chunk from the pipeline into a staging slot.
    ///            When muted (interruption in progress) it drains the channel
    ///            silently and resets next_send_time to None so the next bot
    ///            turn always starts with a fresh deadline.
    ///
    ///   Arm 3 — `sleep_until(next_send_time)`: active only when `pending_chunk` is Some.
    ///            Fires at the scheduled deadline and sends the staged chunk.
    ///            Uses Python-style drift handling: if the deadline was already
    ///            in the past (no actual sleep), resets to now + interval.
    ///            If we slept normally, advances the absolute timeline.
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
        let send_interval = self.send_interval;

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
                // Only anchors the deadline when None (first chunk or post-
                // interruption). Drift correction happens in arm 3.
                // ----------------------------------------------------------------
                audio_bytes = audio_out_rx.recv(), if pending_chunk.is_none() => {
                    match audio_bytes {
                        Some(bytes) => {
                            if mute_gate.load(Ordering::Relaxed) {
                                // Interruption active — drain silently.
                                // Reset deadline so the next bot turn starts
                                // with a fresh anchor.
                                next_send_time = None;
                                continue;
                            }

                            pending_chunk = Some(bytes);

                            // Only set deadline if there isn't one.
                            // First chunk of a new turn or resume after
                            // interruption — arm 3 handles everything else.
                            if next_send_time.is_none() {
                                next_send_time = Some(tokio::time::Instant::now() + send_interval);
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
                //
                // Python-style drift handling after each send:
                //   - If deadline was in the past (sleep_duration == 0),
                //     we fell behind → reset to now + interval.
                //   - If we actually slept, we're on time → advance the
                //     absolute timeline by interval.
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

                        // Python-style drift correction.
                        let now = tokio::time::Instant::now();
                        if let Some(t) = next_send_time {
                            if t <= now {
                                // Behind schedule — reset clock to now.
                                next_send_time = Some(now + send_interval);
                            } else {
                                // On time — advance absolute timeline.
                                next_send_time = Some(t + send_interval);
                            }
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
