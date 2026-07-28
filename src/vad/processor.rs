//! VAD processor.
//!
//! `VadProcessor` is a [`FrameHandler`] that:
//! 1. Receives `InputAudioRaw` frames from the transport or directly from the pipeline.
//! 2. Accumulates PCM until a full Silero inference window is ready.
//! 3. Runs inference via the configured `VadAnalyzer` backend.
//! 4. Advances the state machine with the result.
//! 5. Emits `VADUserStartedSpeaking` / `VADUserStoppedSpeaking` on transitions.
//! 6. Optionally integrates with `SmartTurnAnalyzer` for ML-based end-of-turn detection.
//!
//! Supports both `SileroVadNative` (pure Rust) and `SileroVadOrt` (ONNX Runtime)
//! via the `VadAnalyzer` trait.
//!
//! # Usage without a transport
//!
//! ```ignore
//! let vad = VadProcessor::new(16_000, VadParams::default(), VadBackend::Native)?
//!     .into_processor();
//!
//! let task = PipelineTask::new(vec![vad, stt, llm, tts], PipelineParams::default());
//!
//! // Push raw audio directly:
//! let audio = Frame::input_audio_raw(AudioRawData::new(bytes, 16_000, 1));
//! task.push_frame(audio, FrameDirection::Downstream).await?;
//!
//! // Consume VAD events:
//! let mut filter = HashSet::new();
//! filter.insert(FrameKind::VADUserStartedSpeaking);
//! filter.insert(FrameKind::VADUserStoppedSpeaking);
//! task.set_downstream_filter(filter);
//! task.add_on_frame_reached_downstream(|frame| Box::pin(async move {
//!     println!("VAD event: {:?}", frame.kind());
//! }));
//! ```

use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use log;

use crate::error::Result;
use crate::frames::{
    ControlFrame, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor, SystemFrame,
};
use crate::turn::{EndOfTurnState, SmartTurnAnalyzer, SmartTurnConfig};

use super::analyzer::VadAnalyzer;
use super::params::VadParams;
use super::state::{StateMachine, VadState};
use super::{VadBackend, create_vad};

// ---------------------------------------------------------------------------
// VadProcessorState
// ---------------------------------------------------------------------------

struct VadProcessorState {
    machine: StateMachine,
    model: Arc<dyn VadAnalyzer>,
    params: VadParams,
    sample_rate: u32,
    last_is_speech: bool,
    emitted_speaking: bool,
    user_speaking: bool,
    bot_speaking: bool,
    turn_analyzer: Option<SmartTurnAnalyzer>,
}

// ---------------------------------------------------------------------------
// VadProcessor
// ---------------------------------------------------------------------------

/// VAD processor — place between transport input and STT in the pipeline,
/// or use without a transport by pushing `InputAudioRaw` frames directly.
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

        let machine = StateMachine::new(sample_rate, params.clone());

        Ok(Self {
            state: Arc::new(Mutex::new(VadProcessorState {
                machine,
                model,
                params,
                sample_rate,
                last_is_speech: false,
                emitted_speaking: false,
                user_speaking: false,
                bot_speaking: false,
                turn_analyzer: None,
            })),
        })
    }

    /// Create with default backend (Native).
    pub fn with_defaults(sample_rate: u32, params: VadParams) -> Result<Self> {
        Self::new(sample_rate, params, VadBackend::default())
    }

    /// Enable ML-based smart end-of-turn detection.
    ///
    /// Requires the VAD analyzer to already be configured (i.e. this processor
    /// was created with a valid backend).  
    /// If `turn_config` is `None` the call is a no-op.
    pub fn with_smart_turn(
        self,
        turn_config: Option<&SmartTurnConfig>,
    ) -> Result<Self> {
        let Some(config) = turn_config else {
            return Ok(self);
        };

        let sample_rate = self.state.lock().unwrap().sample_rate;
        let vad_start_secs = self.state.lock().unwrap().params.start_secs as f64;

        let mut analyzer = SmartTurnAnalyzer::new(config)
            .map_err(|e| crate::error::PipecatError::pipeline(format!("smart turn: {}", e)))?;

        analyzer.set_sample_rate(sample_rate);
        analyzer.update_vad_start_secs(vad_start_secs);

        self.state.lock().unwrap().turn_analyzer = Some(analyzer);

        log::info!("VadProcessor: smart turn analyzer initialized");
        Ok(self)
    }

    /// Build a `FrameProcessor` wrapping this handler.
    pub fn into_processor(self) -> FrameProcessor {
        FrameProcessor::new("VadProcessor", Box::new(self), false)
    }

    /// Feed `pending` into the VAD buffer and take the next complete window.
    ///
    /// `pending` is emptied by the first call, so a drain loop feeds the frame
    /// in once and then keeps pulling whatever is already buffered. Returns
    /// `None` once less than a full window remains.
    fn next_vad_window(&self, pending: &mut &[u8]) -> Option<Vec<u8>> {
        let mut guard = self.state.lock().unwrap();
        let window = guard.machine.next_window(pending);
        *pending = &[];
        window
    }

    fn reset_state(&self) {
        let mut guard = self.state.lock().unwrap();
        guard.emitted_speaking = false;
        guard.user_speaking = false;
        guard.bot_speaking = false;
        guard.last_is_speech = false;
        guard.machine = StateMachine::new(guard.sample_rate, guard.params.clone());
        if let Some(ref mut ta) = guard.turn_analyzer {
            ta.clear();
        }
        log::debug!("VadProcessor: state reset");
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
        match &frame.inner {
            // -----------------------------------------------------------------
            // Lifecycle
            // -----------------------------------------------------------------
            FrameInner::System(SystemFrame::Start(_)) => {
                processor.push_frame(frame, direction).await?;
                self.reset_state();
            }
            FrameInner::System(SystemFrame::Stop { .. }) => {
                processor.push_frame(frame, direction).await?;
            }
            FrameInner::Control(ControlFrame::End { .. }) => {
                processor.push_frame(frame, direction).await?;
                self.reset_state();
            }
            FrameInner::System(SystemFrame::Cancel { .. }) => {
                processor.push_frame(frame, direction).await?;
                self.reset_state();
            }

            // -----------------------------------------------------------------
            // Speaking state tracking (mirrors BaseInputTransport behaviour)
            // -----------------------------------------------------------------
            FrameInner::System(SystemFrame::BotStartedSpeaking) => {
                self.state.lock().unwrap().bot_speaking = true;
                processor.push_frame(frame, direction).await?;
            }
            FrameInner::System(SystemFrame::BotStoppedSpeaking) => {
                self.state.lock().unwrap().bot_speaking = false;
                processor.push_frame(frame, direction).await?;
            }
            FrameInner::System(SystemFrame::VADUserStartedSpeaking { .. }) => {
                self.state.lock().unwrap().user_speaking = true;
                processor.push_frame(frame, direction).await?;
            }
            FrameInner::System(SystemFrame::VADUserStoppedSpeaking { .. }) => {
                self.state.lock().unwrap().user_speaking = false;
                processor.push_frame(frame, direction).await?;
            }

            // -----------------------------------------------------------------
            // Core VAD + SmartTurn
            // -----------------------------------------------------------------
            FrameInner::System(SystemFrame::InputAudioRaw(ref audio_data)) => {
                if direction == FrameDirection::Downstream {
                    // Always pass audio through so STT still gets it.
                    processor.push_frame(frame.clone(), direction).await?;

                    // --- VAD window processing ---
                    // Drain every window this frame completed, not just the
                    // first. A frame can carry more audio than one window (an
                    // upsampling input path produces frames wider than the
                    // 512-sample window), and stopping after one leaves the
                    // remainder buffered forever, so the VAD falls permanently
                    // further behind the live call.
                    let mut vad_quiet_transition = false;
                    let mut pending: &[u8] = &audio_data.audio;

                    while let Some(window) = self.next_vad_window(&mut pending) {
                        let model = {
                            self.state.lock().unwrap().model.clone()
                        };

                        let confidence = model.voice_confidence(window.clone()).await;

                        let (prev_state, new_state) = {
                            let mut guard = self.state.lock().unwrap();
                            guard.last_is_speech = confidence >= guard.params.confidence;
                            let prev = guard.machine.state;
                            let next = guard.machine.advance(confidence, &window);
                            (prev, next)
                        };

                        let ts = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs_f64();

                        match (prev_state, new_state) {
                            // Quiet/Starting → Speaking: speech confirmed
                            (s, VadState::Speaking) if s != VadState::Speaking => {
                                let maybe_start = {
                                    let mut guard = self.state.lock().unwrap();
                                    if !guard.emitted_speaking {
                                        guard.emitted_speaking = true;
                                        guard.user_speaking = true;
                                        Some(guard.params.start_secs)
                                    } else {
                                        None
                                    }
                                };

                                if let Some(start_secs) = maybe_start {
                                    log::info!(
                                        "VAD: → Speaking (confidence={:.3})",
                                        confidence
                                    );
                                    let vad_frame =
                                        Frame::vad_user_started_speaking(start_secs, ts);
                                    processor
                                        .push_frame(vad_frame, FrameDirection::Downstream)
                                        .await?;
                                }
                            }

                            // Speaking/Stopping → Quiet: silence confirmed
                            (s, VadState::Quiet) if s != VadState::Quiet => {
                                let maybe_stop = {
                                    let mut guard = self.state.lock().unwrap();
                                    if guard.emitted_speaking {
                                        if guard.turn_analyzer.is_some() {
                                            log::info!(
                                                "VAD: → Quiet (confidence={:.3}), \
                                                 deferring to SmartTurn",
                                                confidence
                                            );
                                            vad_quiet_transition = true;
                                            None
                                        } else {
                                            guard.emitted_speaking = false;
                                            guard.user_speaking = false;
                                            Some(guard.params.stop_secs)
                                        }
                                    } else {
                                        None
                                    }
                                };

                                if let Some(stop_secs) = maybe_stop {
                                    log::info!(
                                        "VAD: → Quiet (confidence={:.3})",
                                        confidence
                                    );
                                    let vad_frame =
                                        Frame::vad_user_stopped_speaking(stop_secs, ts);
                                    processor
                                        .push_frame(vad_frame, FrameDirection::Downstream)
                                        .await?;
                                }
                            }

                            _ => {}
                        }
                    }

                    // --- Smart turn processing ---
                    let (last_is_speech, has_smart_turn) = {
                        let guard = self.state.lock().unwrap();
                        (guard.last_is_speech, guard.turn_analyzer.is_some())
                    };

                    if has_smart_turn {
                        let turn_state = {
                            let mut guard = self.state.lock().unwrap();
                            if let Some(ref mut ta) = guard.turn_analyzer {
                                ta.append_audio(&audio_data.audio, last_is_speech)
                            } else {
                                EndOfTurnState::Incomplete
                            }
                        };

                        if turn_state == EndOfTurnState::Complete {
                            let maybe_stop = {
                                let mut guard = self.state.lock().unwrap();
                                if guard.emitted_speaking {
                                    guard.emitted_speaking = false;
                                    guard.user_speaking = false;
                                    Some(guard.params.stop_secs)
                                } else {
                                    None
                                }
                            };

                            if let Some(stop_secs) = maybe_stop {
                                log::info!("SmartTurn: → Complete (silence timeout)");
                                let ts = SystemTime::now()
                                    .duration_since(UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_secs_f64();
                                let vad_frame = Frame::vad_user_stopped_speaking(stop_secs, ts);
                                processor
                                    .push_frame(vad_frame, FrameDirection::Downstream)
                                    .await?;
                            }
                        }

                        if vad_quiet_transition {
                            let (result, metrics) = {
                                let mut guard = self.state.lock().unwrap();
                                if let Some(ref mut ta) = guard.turn_analyzer {
                                    ta.analyze_end_of_turn()
                                } else {
                                    (EndOfTurnState::Incomplete, None)
                                }
                            };

                            if let Some(ref m) = metrics {
                                log::info!(
                                    "SmartTurn: prob={:.3} complete={} time={:.1}ms",
                                    m.probability,
                                    m.is_complete,
                                    m.e2e_processing_time_ms
                                );
                            }

                            if result == EndOfTurnState::Complete {
                                let maybe_stop = {
                                    let mut guard = self.state.lock().unwrap();
                                    if guard.emitted_speaking {
                                        guard.emitted_speaking = false;
                                        guard.user_speaking = false;
                                        Some(guard.params.stop_secs)
                                    } else {
                                        None
                                    }
                                };

                                if let Some(stop_secs) = maybe_stop {
                                    log::info!("SmartTurn: → Complete (ML prediction)");
                                    let ts = SystemTime::now()
                                        .duration_since(UNIX_EPOCH)
                                        .unwrap_or_default()
                                        .as_secs_f64();
                                    let vad_frame =
                                        Frame::vad_user_stopped_speaking(stop_secs, ts);
                                    processor
                                        .push_frame(vad_frame, FrameDirection::Downstream)
                                        .await?;
                                }
                            } else {
                                log::info!("SmartTurn: → Incomplete, waiting for more audio");
                            }
                        }
                    }

                    return Ok(());
                }

                // Upstream audio: just pass through.
                processor.push_frame(frame, direction).await?;
            }

            // -----------------------------------------------------------------
            // Default pass-through
            // -----------------------------------------------------------------
            _ => {
                processor.push_frame(frame, direction).await?;
            }
        }

        Ok(())
    }
}
