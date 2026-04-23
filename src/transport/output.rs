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

    /// True while OutputAudioRaw frames are being sent.
    bot_speaking: AtomicBool,

    /// Set by the concrete transport after construction.
    audio_out_tx: Mutex<Option<mpsc::Sender<Vec<u8>>>>,

    /// Mute gate — set true on InterruptionFrame so run_socket drops
    /// any stale audio bytes already sitting in the channel.
    /// Cleared automatically on the first OutputAudioRaw of the next
    /// bot turn so audio flows again without any manual reset.
    muted: Arc<AtomicBool>,
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
                muted: Arc::new(AtomicBool::new(false)),
            }),
        }
    }

    /// Wire up the audio output channel.
    /// Called once by the concrete transport before the pipeline starts.
    pub fn set_audio_out_tx(&self, tx: mpsc::Sender<Vec<u8>>) {
        *self.state.audio_out_tx.lock().unwrap() = Some(tx);
    }

    pub fn is_bot_speaking(&self) -> bool {
        self.state.bot_speaking.load(Ordering::Relaxed)
    }

    /// Clone the mute gate Arc so run_socket can share it.
    /// When this returns true, run_socket should silently drop audio chunks.
    pub fn mute_gate(&self) -> Arc<AtomicBool> {
        self.state.muted.clone()
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
                // New bot audio arriving — clear the mute gate so run_socket
                // starts sending again. This handles the case where the bot
                // was interrupted and is now responding to the new user turn.
                self.state.muted.store(false, Ordering::Relaxed);

                if !self.state.bot_speaking.swap(true, Ordering::Relaxed) {
                    log::debug!("BaseOutputTransport: bot started speaking");
                    processor.broadcast_frame(Frame::bot_started_speaking()).await?;
                }

                let tx = self.state.audio_out_tx.lock().unwrap().clone();
                if let Some(tx) = tx {
                    if tx.try_send(audio.audio.clone()).is_err() {
                        log::warn!("BaseOutputTransport: audio_out channel full — dropping chunk");
                    }
                }

                processor.push_frame(frame, direction).await?;
            }

            // ---- Interruption: mute immediately + stop bot ----
            FrameInner::System(SystemFrame::Interruption) => {
                // Set mute gate BEFORE anything else so run_socket drops
                // in-flight chunks as fast as possible.
                self.state.muted.store(true, Ordering::Relaxed);

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
