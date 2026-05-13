use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use log;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;

use crate::error::Result;
use crate::frames::{
    DataFrame, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor, SystemFrame,
};

use super::params::TransportParams;

// ---------------------------------------------------------------------------
// OutputMessage
// ---------------------------------------------------------------------------

/// Sent through the audio-out channel to the concrete transport.
#[derive(Debug)]
pub enum OutputMessage {
    Audio(Vec<u8>),
    /// Serialised JSON text frame — used by RAVI for protocol messages.
    Text(String),
    Interruption,
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

struct OutputTransportState {
    params: TransportParams,
    bot_speaking: AtomicBool,
    audio_out_tx: Mutex<Option<mpsc::Sender<OutputMessage>>>,
    audio_buffer: Mutex<Vec<u8>>,
    chunk_size: AtomicU32,
}

// ---------------------------------------------------------------------------
// BaseOutputTransport
// ---------------------------------------------------------------------------

pub struct BaseOutputTransport {
    state: Arc<OutputTransportState>,
}

impl BaseOutputTransport {
    pub fn new(params: TransportParams) -> Self {
        Self {
            state: Arc::new(OutputTransportState {
                params,
                bot_speaking: AtomicBool::new(false),
                audio_out_tx: Mutex::new(None),
                audio_buffer: Mutex::new(Vec::with_capacity(8192)),
                chunk_size: AtomicU32::new(0),
            }),
        }
    }

    pub fn set_audio_out_tx(&self, tx: mpsc::Sender<OutputMessage>) {
        *self.state.audio_out_tx.lock().unwrap() = Some(tx);
    }

    pub fn is_bot_speaking(&self) -> bool {
        self.state.bot_speaking.load(Ordering::Relaxed)
    }

    /// Check if the output channel is alive (not closed).
    fn is_output_alive(&self) -> bool {
        let guard = self.state.audio_out_tx.lock().unwrap();
        match guard.as_ref() {
            Some(tx) => !tx.is_closed(),
            None => false,
        }
    }

    /// Clear the dead sender and audio buffer when the channel closes.
    fn clear_dead_output(&self) {
        let mut tx_guard = self.state.audio_out_tx.lock().unwrap();
        *tx_guard = None;
        drop(tx_guard);

        let mut buf = self.state.audio_buffer.lock().unwrap();
        buf.clear();
        // Force shrink if buffer grew large
        if buf.capacity() > 65536 {
            *buf = Vec::with_capacity(8192);
        }
        drop(buf);

        self.state.bot_speaking.store(false, Ordering::Relaxed);
    }

    /// Send a message to the output channel.
    /// Returns `true` if the channel is still alive after the attempt.
    fn try_send_output(&self, msg: OutputMessage) -> bool {
        let tx = {
            let guard = self.state.audio_out_tx.lock().unwrap();
            guard.clone()
        };

        let Some(tx) = tx else {
            return false;
        };

        match tx.try_send(msg) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) => {
                log::warn!("BaseOutputTransport: output channel full — dropping message");
                true // Channel alive but backpressured
            }
            Err(TrySendError::Closed(_)) => {
                log::warn!("BaseOutputTransport: output channel closed — WebSocket disconnected");
                self.clear_dead_output();
                false
            }
        }
    }

    /// Send a text (JSON) message to the client without going through the
    /// frame pipeline — used by RAVI server message / response frames.
    fn send_text(&self, payload: &str) {
        self.try_send_output(OutputMessage::Text(payload.to_string()));
    }
}

// ---------------------------------------------------------------------------
// FrameHandler impl
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameHandler for BaseOutputTransport {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match &frame.inner {
            // ---- Audio output ----
            FrameInner::Data(DataFrame::OutputAudioRaw(audio)) => {
                // Early exit: don't buffer audio if output is dead
                if !self.is_output_alive() {
                    log::debug!("BaseOutputTransport: output dead, dropping audio frame");
                    processor.push_frame(frame, direction).await?;
                    return Ok(());
                }

                let channels = audio.num_channels.max(1) as u32;
                let multiplier = self.state.params.audio_out_10ms_chunks.max(1);
                let base_10ms = (audio.sample_rate / 100) * channels * 2;
                let new_chunk_size = base_10ms * multiplier;

                if new_chunk_size == 0 {
                    log::warn!(
                        "BaseOutputTransport: invalid sample_rate={} — skipping frame",
                        audio.sample_rate
                    );
                    processor.push_frame(frame, direction).await?;
                    return Ok(());
                }

                let prev = self.state.chunk_size.swap(new_chunk_size, Ordering::Relaxed);
                if prev != new_chunk_size {
                    log::info!(
                        "BaseOutputTransport: chunk_size={}B ({}ms) (sr={}, ch={}, 10ms_chunks={})",
                        new_chunk_size, multiplier * 10,
                        audio.sample_rate, channels, multiplier,
                    );
                }

                let chunk_size = new_chunk_size as usize;

                if !self.state.bot_speaking.swap(true, Ordering::Relaxed) {
                    log::debug!("BaseOutputTransport: bot started speaking");
                    processor.broadcast_frame(Frame::bot_started_speaking()).await?;
                }

                let chunks: Vec<Vec<u8>> = {
                    let mut buf = self.state.audio_buffer.lock().unwrap();
                    buf.extend_from_slice(&audio.audio);

                    // Safety cap: if buffer grew way beyond chunk_size, something is wrong
                    let max_buffered = chunk_size * 50; // ~500ms max backlog
                    if buf.len() > max_buffered {
                        log::warn!(
                            "BaseOutputTransport: audio buffer exceeded {}B, draining {}B",
                            max_buffered,
                            buf.len() - max_buffered
                        );
                        let drain = buf.len() - max_buffered;
                        buf.drain(..drain);
                    }

                    let mut out = Vec::with_capacity(buf.len() / chunk_size + 1);
                    while buf.len() >= chunk_size {
                        out.push(buf.drain(..chunk_size).collect());
                    }
                    out
                };

                for chunk in chunks {
                    if !self.try_send_output(OutputMessage::Audio(chunk)) {
                        break; // Channel died mid-send, stop wasting CPU
                    }
                }

                // Shrink buffer if it grew too large and is now empty
                {
                    let mut buf = self.state.audio_buffer.lock().unwrap();
                    if buf.is_empty() && buf.capacity() > 65536 {
                        *buf = Vec::with_capacity(8192);
                    }
                }

                processor.push_frame(frame, direction).await?;
            }

            // ---- RAVI outbound messages ----
            FrameInner::System(SystemFrame::RaviServerMessage { payload }) => {
                log::trace!("BaseOutputTransport: sending RAVI server message");
                self.send_text(payload);
                processor.push_frame(frame, direction).await?;
            }

            FrameInner::System(SystemFrame::RaviServerResponse { payload, .. }) => {
                log::trace!("BaseOutputTransport: sending RAVI server response");
                self.send_text(payload);
                processor.push_frame(frame, direction).await?;
            }

            // ---- Interruption ----
            FrameInner::System(SystemFrame::Interruption) => {
                self.state.audio_buffer.lock().unwrap().clear();

                self.try_send_output(OutputMessage::Interruption);

                if self.state.bot_speaking.swap(false, Ordering::Relaxed) {
                    log::debug!("BaseOutputTransport: bot stopped speaking (interruption)");
                    processor.broadcast_frame(Frame::bot_stopped_speaking()).await?;
                }
                processor.push_frame(frame, direction).await?;
            }

            // ---- End / Cancel ----
            FrameInner::Control(_) | FrameInner::System(SystemFrame::Cancel { .. }) => {
                self.state.audio_buffer.lock().unwrap().clear();

                if self.state.bot_speaking.swap(false, Ordering::Relaxed) {
                    log::debug!("BaseOutputTransport: bot stopped speaking (end/cancel)");
                    processor
                        .push_frame(Frame::bot_stopped_speaking(), FrameDirection::Upstream)
                        .await?;
                }
                processor.push_frame(frame, direction).await?;
            }

            // ---- Everything else ----
            _ => {
                processor.push_frame(frame, direction).await?;
            }
        }

        Ok(())
    }
}
