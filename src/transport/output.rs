//! Base output transport.
//!
//! `BaseOutputTransport` handles the common output logic:
//!   - Emitting `BotStartedSpeaking` / `BotStoppedSpeaking` signals.
//!   - Forwarding `OutputAudioRaw` bytes to the concrete transport via an
//!     optional `mpsc::Sender<Vec<u8>>`.
//!
//! The concrete transport (e.g. `WebSocketTransport`) creates the channel,
//! calls `set_audio_out_tx()` on construction, and owns the receiver end in
//! its socket loop — which writes the bytes onto the wire.

use std::sync::atomic::{AtomicBool, Ordering};
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
// State
// ---------------------------------------------------------------------------

struct OutputTransportState {
    #[allow(dead_code)]
    params: TransportParams,

    /// True while `OutputAudioRaw` frames are being sent.
    bot_speaking: AtomicBool,

    /// Set by the concrete transport after construction.
    /// When `Some`, raw PCM bytes from `OutputAudioRaw` frames are forwarded
    /// here so the socket loop can write them to the wire.
    audio_out_tx: Mutex<Option<mpsc::Sender<Vec<u8>>>>,
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
            }),
        }
    }

    /// Wire up the audio output channel.
    ///
    /// Called once by the concrete transport (e.g. `WebSocketTransport::new`)
    /// before the pipeline starts. The receiver end lives in `run_socket`.
    pub fn set_audio_out_tx(&self, tx: mpsc::Sender<Vec<u8>>) {
        *self.state.audio_out_tx.lock().unwrap() = Some(tx);
    }

    pub fn is_bot_speaking(&self) -> bool {
        self.state.bot_speaking.load(Ordering::Relaxed)
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
                // Emit BotStartedSpeaking on the first audio frame of a turn.
                if !self.state.bot_speaking.swap(true, Ordering::Relaxed) {
                    log::debug!("BaseOutputTransport: bot started speaking");
                    processor.broadcast_frame(Frame::bot_started_speaking()).await?;
                }

                // Forward bytes to the concrete transport's socket loop.
                // `try_send` — if the channel is full we drop the chunk rather
                // than blocking the pipeline (real-time audio: drop > delay).
                let tx = self.state.audio_out_tx.lock().unwrap().clone();
                if let Some(tx) = tx {
                    if tx.try_send(audio.audio.clone()).is_err() {
                        log::warn!("BaseOutputTransport: audio_out channel full — dropping chunk");
                    }
                }

                processor.push_frame(frame, direction).await?;
            }

            // ---- Interruption: bot stops speaking ----
            FrameInner::System(SystemFrame::Interruption) => {
                if self.state.bot_speaking.swap(false, Ordering::Relaxed) {
                    log::debug!("BaseOutputTransport: bot stopped speaking (interruption)");
                    processor.broadcast_frame(Frame::bot_stopped_speaking()).await?;
                }
                processor.push_frame(frame, direction).await?;
            }

            // ---- End / Cancel: clear speaking state ----
            FrameInner::Control(_) | FrameInner::System(SystemFrame::Cancel { .. }) => {
                if self.state.bot_speaking.swap(false, Ordering::Relaxed) {
                    log::debug!("BaseOutputTransport: bot stopped speaking (end/cancel)");
                    processor
                        .push_frame(Frame::bot_stopped_speaking(), FrameDirection::Upstream)
                        .await?;
                }
                processor.push_frame(frame, direction).await?;
            }

            // ---- Everything else: pass through ----
            _ => {
                processor.push_frame(frame, direction).await?;
            }
        }

        Ok(())
    }
}