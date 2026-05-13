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

const AUDIO_TIMEOUT: Duration = Duration::from_millis(500);
const AUDIO_CHANNEL_CAP: usize = 100;

struct InputTransportState {
    params: TransportParams,
    sample_rate:      AtomicU32,
    paused:           AtomicBool,
    bot_speaking:     AtomicBool,
    user_speaking:    AtomicBool,
    emitted_speaking: AtomicBool,
    audio_tx:   mpsc::Sender<AudioRawData>,
    audio_rx:   std::sync::Mutex<Option<mpsc::Receiver<AudioRawData>>>,
    audio_task: std::sync::Mutex<Option<JoinHandle<()>>>,
    vad_machine: std::sync::Mutex<Option<StateMachine>>,
}

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

    fn on_start(&self, processor: &FrameProcessor) {
        self.state.paused.store(false, Ordering::Relaxed);
        self.state.user_speaking.store(false, Ordering::Relaxed);
        self.state.emitted_speaking.store(false, Ordering::Relaxed);

        let sr = self.state.params.audio_in_sample_rate.unwrap_or(16_000);
        self.state.sample_rate.store(sr, Ordering::Relaxed);

        if self.state.params.vad_analyzer.is_some() {
            let machine = StateMachine::new(sr, self.state.params.vad_params.clone());
            *self.state.vad_machine.lock().unwrap() = Some(machine);
            log::info!("BaseInputTransport: VAD state machine initialized (sr={})", sr);
        } else {
            log::warn!("BaseInputTransport: no VAD analyzer configured");
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

async fn run_audio_task(
    state: Arc<InputTransportState>,
    mut rx: mpsc::Receiver<AudioRawData>,
    processor: FrameProcessor,
) {
    let mut audio_received = false;
    let mut chunk_count = 0u64;
    log::info!("BaseInputTransport: audio task started");

    loop {
        match tokio::time::timeout(AUDIO_TIMEOUT, rx.recv()).await {
            Ok(Some(data)) => {
                audio_received = true;
                chunk_count += 1;

                if state.paused.load(Ordering::Relaxed) {
                    continue;
                }

                if state.params.audio_in_passthrough {
                    let frame = Frame::input_audio_raw(data.clone());
                    if let Err(e) = processor
                        .push_frame(frame, FrameDirection::Downstream)
                        .await
                    {
                        log::error!("BaseInputTransport: push_frame failed: {}", e);
                    }
                }

                if let Some(analyzer) = &state.params.vad_analyzer {
                    let window_opt = {
                        let mut machine = state.vad_machine.lock().unwrap();
                        machine.as_mut().and_then(|m| m.next_window(&data.audio))
                    };

                    if let Some(window) = window_opt {
                        let confidence = analyzer.voice_confidence(window.clone()).await;

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

                            match vad_state {
                                VadState::Speaking if !was_speaking => {
                                    log::info!("VAD: → Speaking (confidence={:.3})", confidence);
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
                                    log::info!("VAD: → Quiet (confidence={:.3})", confidence);
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

            Ok(None) => {
                log::info!("BaseInputTransport: audio channel closed — task exiting");
                break;
            }

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

    log::info!("BaseInputTransport: audio task exited (total chunks: {})", chunk_count);
}
