use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use log;
use tokio::sync::mpsc;

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

    /// Send a text (JSON) message to the client without going through the
    /// frame pipeline — used by RAVI server message / response frames.
    fn send_text(&self, payload: &str) {
        let tx = self.state.audio_out_tx.lock().unwrap().clone();
        if let Some(tx) = tx {
            if tx.try_send(OutputMessage::Text(payload.to_string())).is_err() {
                log::warn!("BaseOutputTransport: text output channel full — dropping RAVI message");
            }
        }
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
                    let mut out = Vec::with_capacity(buf.len() / chunk_size + 1);
                    while buf.len() >= chunk_size {
                        out.push(buf.drain(..chunk_size).collect());
                    }
                    out
                };

                let tx = self.state.audio_out_tx.lock().unwrap().clone();
                if let Some(tx) = tx {
                    for chunk in chunks {
                        if tx.try_send(OutputMessage::Audio(chunk)).is_err() {
                            log::warn!("BaseOutputTransport: audio_out channel full — dropping chunk");
                        }
                    }
                }

                processor.push_frame(frame, direction).await?;
            }

            // ---- RAVI outbound messages ----
            // These are system frames so they bypass the ordered data queue.
            // Serialize → text channel → WebSocket sends them to the client.

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

                let tx = self.state.audio_out_tx.lock().unwrap().clone();
                if let Some(tx) = tx {
                    let _ = tx.try_send(OutputMessage::Interruption);
                }

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
