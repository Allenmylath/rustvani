//! Base input transport.
//!
//! `BaseInputTransport` is a [`FrameHandler`] that handles the common input
//! logic shared by all concrete transports (WebRTC, WebSocket, etc.):
//!
//! - A dedicated audio queue + task, **separate** from the pipeline queues.
//!   This ensures VAD / STT latency is never affected by pipeline backpressure.
//! - Lifecycle management (start / pause / stop / cancel).
//! - Bot and user speaking state tracking.
//! - A clearly marked VAD stub, wired in when `pipecat-audio` is ready.
//!
//! # Usage by concrete transports
//!
//! ```text
//! let base = Arc::new(BaseInputTransport::new(params));
//!
//! // Install as the handler of a FrameProcessor.
//! let processor = FrameProcessor::new("WebRtcInput", Box::new(base.clone()), false);
//!
//! // Inject audio from the network thread / callback.
//! base.push_audio_frame(AudioRawData::new(pcm_bytes, 16_000, 1)).await;
//! ```

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use log;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::error::Result;
use crate::frames::{
    AudioRawData, ControlFrame, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor,
    SystemFrame,
};

use super::params::TransportParams;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// How long the audio task waits for the next frame before checking state.
/// Mirrors Python's `AUDIO_INPUT_TIMEOUT_SECS = 0.5`.
const AUDIO_TIMEOUT: Duration = Duration::from_millis(500);

/// Internal audio channel capacity (frames buffered before back-pressure).
const AUDIO_CHANNEL_CAP: usize = 100;

// ---------------------------------------------------------------------------
// Shared inner state
// ---------------------------------------------------------------------------

/// State shared between `BaseInputTransport` and its audio task.
struct InputTransportState {
    params: TransportParams,

    /// Resolved sample rate — set when `StartFrame` is processed.
    sample_rate: AtomicU32,

    /// Set by `StopFrame`; cleared by the next `StartFrame`.
    paused: AtomicBool,

    /// Tracks whether the bot is currently producing audio.
    /// Set by `BotStartedSpeaking` / `BotStoppedSpeaking` frames.
    bot_speaking: AtomicBool,

    /// Tracks whether the user is currently speaking.
    /// Set by VAD events; also used for timeout recovery.
    user_speaking: AtomicBool,

    /// Sender half of the audio channel.
    /// Concrete transports call `push_audio_frame()` which uses this.
    audio_tx: mpsc::Sender<AudioRawData>,

    /// Receiver half of the audio channel.
    /// Taken once by the audio task on `StartFrame`; `None` after that.
    audio_rx: std::sync::Mutex<Option<mpsc::Receiver<AudioRawData>>>,

    /// Handle for the running audio task; `None` before start / after stop.
    audio_task: std::sync::Mutex<Option<JoinHandle<()>>>,
}

// ---------------------------------------------------------------------------
// BaseInputTransport
// ---------------------------------------------------------------------------

/// Base input transport — shared logic for all concrete input transports.
///
/// Implement [`FrameHandler`] on your concrete transport and delegate to
/// `BaseInputTransport` for the frames you don't need to customise, or use
/// it directly as the handler if no custom behaviour is needed.
pub struct BaseInputTransport {
    state: Arc<InputTransportState>,
}

impl BaseInputTransport {
    /// Create a new base input transport with the given parameters.
    pub fn new(params: TransportParams) -> Self {
        let (audio_tx, audio_rx) = mpsc::channel(AUDIO_CHANNEL_CAP);
        Self {
            state: Arc::new(InputTransportState {
                params,
                sample_rate:  AtomicU32::new(0),
                paused:       AtomicBool::new(false),
                bot_speaking: AtomicBool::new(false),
                user_speaking: AtomicBool::new(false),
                audio_tx,
                audio_rx: std::sync::Mutex::new(Some(audio_rx)),
                audio_task: std::sync::Mutex::new(None),
            }),
        }
    }

    // ---- Public API for concrete transports ----

    /// Push a raw audio chunk into the audio queue.
    ///
    /// Called by concrete transports from their network callbacks.
    /// Returns `true` if the frame was queued, `false` if it was dropped
    /// (audio input disabled, transport paused, or channel full).
    pub async fn push_audio_frame(&self, data: AudioRawData) -> bool {
        if !self.state.params.audio_in_enabled {
            return false;
        }
        if self.state.paused.load(Ordering::Relaxed) {
            return false;
        }
        self.state.audio_tx.send(data).await.is_ok()
    }

    /// Clone the audio sender for concrete transports that need to own it
    /// (e.g. to move it into a network callback closure).
    pub fn audio_sender(&self) -> mpsc::Sender<AudioRawData> {
        self.state.audio_tx.clone()
    }

    /// Current resolved sample rate (0 before `StartFrame`).
    pub fn sample_rate(&self) -> u32 {
        self.state.sample_rate.load(Ordering::Relaxed)
    }

    /// Whether the transport is currently paused.
    pub fn is_paused(&self) -> bool {
        self.state.paused.load(Ordering::Relaxed)
    }

    // ---- Internal lifecycle helpers ----

    fn on_start(&self, processor: &FrameProcessor) {
        self.state.paused.store(false, Ordering::Relaxed);
        self.state.user_speaking.store(false, Ordering::Relaxed);

        // Resolve sample rate: params take priority, then fall back to 16 kHz.
        let sr = self.state.params.audio_in_sample_rate.unwrap_or(16_000);
        self.state.sample_rate.store(sr, Ordering::Relaxed);

        if self.state.params.audio_in_stream_on_start {
            self.spawn_audio_task(processor.clone(), sr);
        }
    }

    fn on_stop(&self) {
        // Pause: new audio arriving via push_audio_frame() will be dropped.
        // The audio task keeps running but will drain to empty then timeout.
        self.state.paused.store(true, Ordering::Relaxed);
    }

    fn on_cancel_or_end(&self) {
        self.abort_audio_task();
    }

    fn spawn_audio_task(&self, processor: FrameProcessor, sample_rate: u32) {
        if !self.state.params.audio_in_enabled {
            return;
        }

        // Take the receiver — can only happen once.
        let rx = match self.state.audio_rx.lock().unwrap().take() {
            Some(rx) => rx,
            None => {
                log::warn!("BaseInputTransport: audio task already running");
                return;
            }
        };

        let state = self.state.clone();
        let handle = tokio::spawn(run_audio_task(state, rx, processor, sample_rate));
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
            // ---- Lifecycle ----

            FrameInner::System(SystemFrame::Start(_)) => {
                // Push Start first so all downstream processors initialise
                // before we potentially push audio frames.
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

            // ---- Speaking state tracking ----

            FrameInner::System(SystemFrame::BotStartedSpeaking) => {
                self.state.bot_speaking.store(true, Ordering::Relaxed);
                processor.push_frame(frame, direction).await?;
            }

            FrameInner::System(SystemFrame::BotStoppedSpeaking) => {
                self.state.bot_speaking.store(false, Ordering::Relaxed);
                processor.push_frame(frame, direction).await?;
            }

            // Update user_speaking when VAD events arrive so the timeout
            // recovery path in the audio task has accurate state.
            FrameInner::System(SystemFrame::VADUserStartedSpeaking { .. }) => {
                self.state.user_speaking.store(true, Ordering::Relaxed);
                processor.push_frame(frame, direction).await?;
            }

            FrameInner::System(SystemFrame::VADUserStoppedSpeaking { .. }) => {
                self.state.user_speaking.store(false, Ordering::Relaxed);
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

// ---------------------------------------------------------------------------
// Audio task
// ---------------------------------------------------------------------------

/// The audio processing task.
///
/// Runs independently of the pipeline queue loop so that VAD latency is
/// never affected by backpressure in the main processor chain.
///
/// Current behaviour: passthrough only.
/// VAD integration is stubbed — see the clearly marked section below.
async fn run_audio_task(
    state: Arc<InputTransportState>,
    mut rx: mpsc::Receiver<AudioRawData>,
    processor: FrameProcessor,
    _sample_rate: u32,
) {
    // Don't fire timeout warnings until the first frame arrives.
    // This avoids noise before the client has connected.
    let mut audio_received = false;

    log::debug!("BaseInputTransport: audio task started");

    loop {
        match tokio::time::timeout(AUDIO_TIMEOUT, rx.recv()).await {
            // ---- Frame received ----
            Ok(Some(data)) => {
                audio_received = true;

                // Ignore frames while paused.
                if state.paused.load(Ordering::Relaxed) {
                    continue;
                }

                // ----------------------------------------------------------------
                // VAD STUB
                //
                // When `pipecat-audio` Silero VAD is wired in, this is where
                // it runs. It will:
                //   1. Call `vad_analyzer.analyze(&data.audio)` → VadState
                //   2. On QUIET → SPEAKING transition:
                //        push Frame::vad_user_started_speaking(start_secs, ts)
                //   3. On SPEAKING → QUIET transition:
                //        push Frame::vad_user_stopped_speaking(stop_secs, ts)
                //
                // For now we fall straight through to passthrough.
                // ----------------------------------------------------------------

                // Passthrough: push InputAudioRaw downstream so STT services
                // (and the future VAD processor) can consume it.
                if state.params.audio_in_passthrough {
                    let frame = Frame::input_audio_raw(data);
                    if let Err(e) = processor
                        .push_frame(frame, FrameDirection::Downstream)
                        .await
                    {
                        log::error!("BaseInputTransport: push_frame failed: {}", e);
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
                    // Client not yet connected — keep waiting silently.
                    continue;
                }

                // Audio was flowing and then stopped.
                // If the user was mid-utterance, push a forced stop so
                // downstream processors don't hang waiting for silence.
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