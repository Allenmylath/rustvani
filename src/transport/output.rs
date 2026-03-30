//! Base output transport.
//!
//! `BaseOutputTransport` is a [`FrameHandler`] that handles the common output
//! logic shared by all concrete transports:
//!
//! - Emitting `BotStartedSpeaking` / `BotStoppedSpeaking` signals when audio
//!   starts and stops flowing.
//! - Draining queued `OutputAudioRaw` frames on interruption.
//!
//! # Status
//!
//! This is a **stub** — the full audio-out queue, mixing, and silence-padding
//! logic will be added once `pipecat-audio` output processing is wired in.
//! The structure is intentionally minimal so concrete transports can depend
//! on it today.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use log;

use crate::error::Result;
use crate::frames::{
    DataFrame, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor, SystemFrame,
};

use super::params::TransportParams;

// ---------------------------------------------------------------------------
// BaseOutputTransport
// ---------------------------------------------------------------------------

struct OutputTransportState {
    #[allow(dead_code)]
    params: TransportParams,

    /// True while `OutputAudioRaw` frames are being sent.
    bot_speaking: AtomicBool,
}

/// Base output transport — shared logic for all concrete output transports.
pub struct BaseOutputTransport {
    state: Arc<OutputTransportState>,
}

impl BaseOutputTransport {
    pub fn new(params: TransportParams) -> Self {
        Self {
            state: Arc::new(OutputTransportState {
                params,
                bot_speaking: AtomicBool::new(false),
            }),
        }
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
            //
            // When the first OutputAudioRaw arrives, emit BotStartedSpeaking.
            // When no more audio is queued, emit BotStoppedSpeaking.
            //
            // TODO: this is a placeholder. The real implementation will:
            //   1. Feed audio into a resampler / mixer.
            //   2. Chunk into transport-appropriate sizes.
            //   3. Call the concrete transport's send_audio() method.
            //   4. Emit BotStartedSpeaking / BotStoppedSpeaking at the right moments.
            FrameInner::Data(DataFrame::OutputAudioRaw(_)) => {
                if !self.state.bot_speaking.swap(true, Ordering::Relaxed) {
                    // Transition: was silent → now speaking.
                    log::debug!("BaseOutputTransport: bot started speaking");
                    let started = Frame::bot_started_speaking();
                    processor
                        .push_frame(started, FrameDirection::Downstream)
                        .await?;
                }

                // Pass the audio frame through to the concrete transport.
                processor.push_frame(frame, direction).await?;
            }

            // ---- Interruption: drain queued audio ----
            //
            // When an interruption arrives, any OutputAudioRaw frames still
            // queued should not be sent. The pipeline's interruption mechanism
            // already drains the process queue, so we just need to emit
            // BotStoppedSpeaking if we were speaking.
            FrameInner::System(SystemFrame::Interruption) => {
                if self.state.bot_speaking.swap(false, Ordering::Relaxed) {
                    log::debug!("BaseOutputTransport: bot stopped speaking (interruption)");
                    let stopped = Frame::bot_stopped_speaking();
                    // Broadcast so both upstream (BaseInputTransport) and
                    // downstream (concrete transport) see the signal.
                    processor.broadcast_frame(stopped).await?;
                }
                processor.push_frame(frame, direction).await?;
            }

            // ---- End / Cancel: clear speaking state ----
            FrameInner::Control(_) | FrameInner::System(SystemFrame::Cancel { .. }) => {
                if self.state.bot_speaking.swap(false, Ordering::Relaxed) {
                    log::debug!("BaseOutputTransport: bot stopped speaking (end/cancel)");
                    let stopped = Frame::bot_stopped_speaking();
                    processor
                        .push_frame(stopped, FrameDirection::Upstream)
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