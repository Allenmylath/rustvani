//! Base input transport.
//!
//! Owns the audio ingestion pipeline including VAD when configured.
//!
//! The audio task is intentionally separate from the pipeline queue loop
//! so VAD latency is never affected by backpressure downstream.
//!
//! # VAD integration
//!
//! Configure a VAD backend on `TransportParams::vad_analyzer`.
//! The audio task owns the state machine and the `emitted_speaking` flag,
//! mirroring Python's transport-level responsibility for event emission.
//!
//! Only two events are emitted, gated by `emitted_speaking`:
//!   - `VADUserStartedSpeaking` — on `* → Speaking` when not already speaking
//!   - `VADUserStoppedSpeaking` — on `* → Quiet` when currently speaking
//!
//! `Starting` and `Stopping` are internal state machine states and never
//! trigger events — exactly matching Python's behaviour.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use log;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::error::Result;
use crate::frames::{
    AudioRawData, ControlFrame, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor,
    SystemFrame,
};
use crate::vad::{StateMachine, VadState};

use super::params::TransportParams;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const AUDIO_TIMEOUT: Duration = Duration::from_millis(500);
const AUDIO_CHANNEL_CAP: usize = 100;

// ---------------------------------------------------------------------------
// Shared inner state
// ---------------------------------------------------------------------------

struct InputTransportState {
    params: TransportParams,

    sample_rate:  AtomicU32,
    paused:       AtomicBool,
    bot_speaking: AtomicBool,

    /// True while user is considered to be in a speaking turn.
    /// Used by the audio timeout recovery path.
    user_speaking: AtomicBool,

    /// True after we have emitted VADUserStartedSpeaking but before
    /// we emit the matching VADUserStoppedSpeaking.
    /// Guards against duplicate events when the state machine oscillates.
    emitted_speaking: AtomicBool,

    audio_tx:   mpsc::Sender<AudioRawData>,
    audio_rx:   std::sync::Mutex<Option<mpsc::Receiver<AudioRawData>>>,
    audio_task: std::sync::Mutex<Option<JoinHandle<()>>>,

    /// VAD state machine — initialised at StartFrame time when vad_analyzer is Some.
    vad_machine: std::sync::Mutex<Option<StateMachine>>,
}

// ---------------------------------------------------------------------------
// BaseInputTransport
// ---------------------------------------------------------------------------

pub struct BaseInputTransport {
    state: Arc<InputTransportState>,
}

impl BaseInputTransport {
    pub fn new(params: TransportParams) -> Self {
        let (audio_tx, audio_rx) = mpsc::channel(AUDIO_CHANNEL_CAP);
        Self {
            state: Arc::new(InputTransportState {
                params,
                sample_rate:      AtomicU32::new(0),
                paused:           AtomicBool::new(false),
                bot_speaking:     AtomicBool::new(false),
                user_speaking:    AtomicBool::new(false),
                emitted_speaking: AtomicBool::new(false),
                audio_tx,
                audio_rx:    std::sync::Mutex::new(Some(audio_rx)),
                audio_task:  std::sync::Mutex::new(None),
                vad_machine: std::sync::Mutex::new(None),
            }),
        }
    }

    pub async fn push_audio_frame(&self, data: AudioRawData) -> bool {
        if !self.state.params.audio_in_enabled {
            return false;
        }
        if self.state.paused.load(Ordering::Relaxed) {
            return false;
        }
        self.state.audio_tx.send(data).await.is_ok()
    }

    pub fn audio_sender(&self) -> mpsc::Sender<AudioRawData> {
        self.state.audio_tx.clone()
    }

    pub fn sample_rate(&self) -> u32 {
        self.state.sample_rate.load(Ordering::Relaxed)
    }

    pub fn is_paused(&self) -> bool {
        self.state.paused.load(Ordering::Relaxed)
    }

    // ---- Lifecycle ----

    fn on_start(&self, processor: &FrameProcessor) {
        self.state.paused.store(false, Ordering::Relaxed);
        self.state.user_speaking.store(false, Ordering::Relaxed);
        self.state.emitted_speaking.store(false, Ordering::Relaxed);

        let sr = self.state.params.audio_in_sample_rate.unwrap_or(16_000);
        self.state.sample_rate.store(sr, Ordering::Relaxed);

        // Initialise VAD state machine if an analyzer is configured.
        if self.state.params.vad_analyzer.is_some() {
            let machine = StateMachine::new(sr, self.state.params.vad_params.clone());
            *self.state.vad_machine.lock().unwrap() = Some(machine);
        }

        if self.state.params.audio_in_stream_on_start {
            self.spawn_audio_task(processor.clone());
        }
    }

    fn on_stop(&self) {
        self.state.paused.store(true, Ordering::Relaxed);
    }

    fn on_cancel_or_end(&self) {
        self.abort_audio_task();
    }

    fn spawn_audio_task(&self, processor: FrameProcessor) {
        if !self.state.params.audio_in_enabled {
            return;
        }

        let rx = match self.state.audio_rx.lock().unwrap().take() {
            Some(rx) => rx,
            None => {
                log::warn!("BaseInputTransport: audio task already running");
                return;
            }
        };

        let state = self.state.clone();
        let handle = tokio::spawn(run_audio_task(state, rx, processor));
        *self.state.audio_task.lock().unwrap() = Some(handle);
    }

    fn abort_audio_task(&self) {
        if let Some(handle) = self.state.audio_task.lock().unwrap().take() {
            handle.abort();
            log::debug!("BaseInputTransport: audio task aborted");
        }
    }
}

// ---------------------------------------------------------------------------
// FrameHandler impl
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameHandler for BaseInputTransport {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match &frame.inner {
            FrameInner::System(SystemFrame::Start(_)) => {
                processor.push_frame(frame, direction).await?;
                self.on_start(processor);
            }
            FrameInner::System(SystemFrame::Stop { .. }) => {
                self.on_stop();
                processor.push_frame(frame, direction).await?;
            }
            FrameInner::Control(ControlFrame::End { .. }) => {
                self.on_cancel_or_end();
                processor.push_frame(frame, direction).await?;
            }
            FrameInner::System(SystemFrame::Cancel { .. }) => {
                self.on_cancel_or_end();
                processor.push_frame(frame, direction).await?;
            }
            FrameInner::System(SystemFrame::BotStartedSpeaking) => {
                self.state.bot_speaking.store(true, Ordering::Relaxed);
                processor.push_frame(frame, direction).await?;
            }
            FrameInner::System(SystemFrame::BotStoppedSpeaking) => {
                self.state.bot_speaking.store(false, Ordering::Relaxed);
                processor.push_frame(frame, direction).await?;
            }
            // Keep user_speaking in sync with what VAD events we emit,
            // so the audio timeout recovery path stays accurate.
            FrameInner::System(SystemFrame::VADUserStartedSpeaking { .. }) => {
                self.state.user_speaking.store(true, Ordering::Relaxed);
                processor.push_frame(frame, direction).await?;
            }
            FrameInner::System(SystemFrame::VADUserStoppedSpeaking { .. }) => {
                self.state.user_speaking.store(false, Ordering::Relaxed);
                processor.push_frame(frame, direction).await?;
            }
            _ => {
                processor.push_frame(frame, direction).await?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Audio task
// ---------------------------------------------------------------------------

async fn run_audio_task(
    state: Arc<InputTransportState>,
    mut rx: mpsc::Receiver<AudioRawData>,
    processor: FrameProcessor,
) {
    let mut audio_received = false;
    log::debug!("BaseInputTransport: audio task started");

    loop {
        match tokio::time::timeout(AUDIO_TIMEOUT, rx.recv()).await {
            // ---- Frame received ----
            Ok(Some(data)) => {
                audio_received = true;

                if state.paused.load(Ordering::Relaxed) {
                    continue;
                }

                // Passthrough: downstream processors (STT etc.) still get the raw audio.
                if state.params.audio_in_passthrough {
                    let frame = Frame::input_audio_raw(data.clone());
                    if let Err(e) = processor
                        .push_frame(frame, FrameDirection::Downstream)
                        .await
                    {
                        log::error!("BaseInputTransport: push_frame failed: {}", e);
                    }
                }

                // VAD — only runs if an analyzer is configured.
                if let Some(analyzer) = &state.params.vad_analyzer {
                    // Feed into state machine buffer; get a window when ready.
                    let window_opt = {
                        let mut machine = state.vad_machine.lock().unwrap();
                        machine.as_mut().and_then(|m| m.next_window(&data.audio))
                    };

                    if let Some(window) = window_opt {
                        let confidence = analyzer.voice_confidence(window.clone()).await;

                        // Advance state machine with inference result.
                        let new_vad_state = {
                            let mut machine = state.vad_machine.lock().unwrap();
                            machine.as_mut().map(|m| m.advance(confidence, &window))
                        };

                        if let Some(vad_state) = new_vad_state {
                            let ts = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs_f64();

                            let was_speaking = state.emitted_speaking.load(Ordering::Relaxed);

                            // Emit events gated on emitted_speaking flag —
                            // mirrors Python transport's user_speaking guard.
                            // Starting/Stopping are internal states; never trigger emission.
                            match vad_state {
                                VadState::Speaking if !was_speaking => {
                                    state.emitted_speaking.store(true, Ordering::Relaxed);
                                    let frame = Frame::vad_user_started_speaking(
                                        state.params.vad_params.start_secs,
                                        ts,
                                    );
                                    if let Err(e) = processor
                                        .push_frame(frame, FrameDirection::Downstream)
                                        .await
                                    {
                                        log::error!("BaseInputTransport: VAD push failed: {}", e);
                                    }
                                }
                                VadState::Quiet if was_speaking => {
                                    state.emitted_speaking.store(false, Ordering::Relaxed);
                                    let frame = Frame::vad_user_stopped_speaking(
                                        state.params.vad_params.stop_secs,
                                        ts,
                                    );
                                    if let Err(e) = processor
                                        .push_frame(frame, FrameDirection::Downstream)
                                        .await
                                    {
                                        log::error!("BaseInputTransport: VAD push failed: {}", e);
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }

            // ---- Channel closed ----
            Ok(None) => {
                log::debug!("BaseInputTransport: audio channel closed — task exiting");
                break;
            }

            // ---- Timeout ----
            Err(_) => {
                if !audio_received {
                    continue;
                }

                if state.user_speaking.load(Ordering::Relaxed) {
                    log::warn!(
                        "BaseInputTransport: audio timeout while user speaking \
                         — forcing UserStoppedSpeaking"
                    );
                    state.user_speaking.store(false, Ordering::Relaxed);

                    let frame = Frame::user_stopped_speaking();
                    if let Err(e) = processor
                        .push_frame(frame, FrameDirection::Downstream)
                        .await
                    {
                        log::error!(
                            "BaseInputTransport: failed to push UserStoppedSpeaking: {}",
                            e
                        );
                    }
                }
            }
        }
    }

    log::debug!("BaseInputTransport: audio task exited");
}