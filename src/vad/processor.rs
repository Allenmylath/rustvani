//! VAD processor.
//!
//! `VadProcessor` is a [`FrameHandler`] that:
//! 1. Receives `InputAudioRaw` frames from the transport.
//! 2. Accumulates PCM until a full Silero inference window is ready.
//! 3. Runs inference via the configured `VadAnalyzer` backend.
//! 4. Advances the state machine with the result.
//! 5. Emits `VADUserStartedSpeaking` / `VADUserStoppedSpeaking` on transitions.
//!
//! Supports both `SileroVadNative` (pure Rust) and `SileroVadOrt` (ONNX Runtime)
//! via the `VadAnalyzer` trait.

use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use log;

use crate::error::Result;
use crate::frames::{
    Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor, SystemFrame,
};

use super::analyzer::VadAnalyzer;
use super::params::VadParams;
use super::state::{StateMachine, VadState};
use super::{VadBackend, create_vad};

// ---------------------------------------------------------------------------
// VadProcessorState — shared between handler and async tasks
// ---------------------------------------------------------------------------

struct VadProcessorState {
    machine:    StateMachine,
    model:      Arc<dyn VadAnalyzer>,
    start_secs: f32,
    stop_secs:  f32,
}

// ---------------------------------------------------------------------------
// VadProcessor
// ---------------------------------------------------------------------------

/// VAD processor — place between transport input and STT in the pipeline.
pub struct VadProcessor {
    state: Arc<Mutex<VadProcessorState>>,
}

impl VadProcessor {
    /// Create a new VAD processor with the specified backend.
    ///
    /// `sample_rate` must be 8000 or 16000.
    /// `backend` selects Native (pure Rust) or Ort (ONNX Runtime).
    pub fn new(
        sample_rate: u32,
        params: VadParams,
        backend: VadBackend,
    ) -> Result<Self> {
        let model = create_vad(backend, sample_rate)
            .map_err(|e| crate::error::PipecatError::pipeline(e))?;

        let start_secs = params.start_secs;
        let stop_secs  = params.stop_secs;
        let machine    = StateMachine::new(sample_rate, params);

        Ok(Self {
            state: Arc::new(Mutex::new(VadProcessorState {
                machine,
                model,
                start_secs,
                stop_secs,
            })),
        })
    }

    /// Create with default backend (Native).
    pub fn with_defaults(sample_rate: u32, params: VadParams) -> Result<Self> {
        Self::new(sample_rate, params, VadBackend::default())
    }

    /// Build a `FrameProcessor` wrapping this handler.
    pub fn into_processor(self) -> FrameProcessor {
        FrameProcessor::new("VadProcessor", Box::new(self), false)
    }
}

// ---------------------------------------------------------------------------
// FrameHandler impl
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameHandler for VadProcessor {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        // Only intercept InputAudioRaw downstream — everything else passes through.
        if let FrameInner::System(SystemFrame::InputAudioRaw(ref audio_data)) = frame.inner {
            if direction == FrameDirection::Downstream {
                // Pass audio through so STT still gets it.
                processor.push_frame(frame.clone(), direction).await?;

                // Check if we have a full window ready.
                let window_opt = {
                    let mut guard = self.state.lock().unwrap();
                    guard.machine.next_window(&audio_data.audio)
                };

                if let Some(window) = window_opt {
                    // Clone model out of the lock — Arc::clone is cheap.
                    let model = {
                        self.state.lock().unwrap().model.clone()
                    };

                    // Run inference outside the lock.
                    let confidence = model.voice_confidence(window.clone()).await;

                    // Advance state machine — get previous and new state.
                    let (prev_state, new_state, start_secs, stop_secs) = {
                        let mut guard = self.state.lock().unwrap();
                        let prev = guard.machine.state;
                        let next = guard.machine.advance(confidence, &window);
                        (prev, next, guard.start_secs, guard.stop_secs)
                    };

                    // Emit VAD frames only on transitions.
                    let ts = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs_f64();

                    match (prev_state, new_state) {
                        // Quiet/Starting → Speaking: speech confirmed
                        (s, VadState::Speaking) if s != VadState::Speaking => {
                            log::debug!("VAD: user started speaking (confidence={:.3})", confidence);
                            let vad_frame = Frame::vad_user_started_speaking(start_secs, ts);
                            processor
                                .push_frame(vad_frame, FrameDirection::Downstream)
                                .await?;
                        }

                        // Speaking/Stopping → Quiet: silence confirmed
                        (s, VadState::Quiet) if s != VadState::Quiet => {
                            log::debug!("VAD: user stopped speaking (confidence={:.3})", confidence);
                            let vad_frame = Frame::vad_user_stopped_speaking(stop_secs, ts);
                            processor
                                .push_frame(vad_frame, FrameDirection::Downstream)
                                .await?;
                        }

                        _ => {}
                    }
                }

                return Ok(());
            }
        }

        // All other frames: pass through unchanged.
        processor.push_frame(frame, direction).await
    }
}
