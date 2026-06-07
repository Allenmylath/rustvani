use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::error::Result;
use crate::frames::{
    ControlFrame, DataFrame, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor, SystemFrame,
};
use super::collector::AudioCaptureCollector;
use super::segment::PendingAudioSegment;

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

struct UserTurn {
    turn_id:    Uuid,
    pcm:        Vec<u8>,
    sample_rate: u32,
    channels:   u16,
    started_at: DateTime<Utc>,
}

struct BotTurn {
    turn_id:    Uuid,
    pcm:        Vec<u8>,
    sample_rate: u32,
    channels:   u16,
    started_at: DateTime<Utc>,
}

struct State {
    user_turn: Option<UserTurn>,
    bot_turn:  Option<BotTurn>,
}

// ---------------------------------------------------------------------------
// AudioCaptureProcessor
// ---------------------------------------------------------------------------

/// Records per-turn audio segments for both user and bot speaking turns.
///
/// Position this processor **after TTS and before the output transport** in
/// the pipeline. At that position it sees:
/// - `InputAudioRaw` (user PCM) travelling downstream
/// - `OutputAudioRaw` (bot PCM) travelling downstream (from TTS)
/// - `BotStartedSpeaking` / `BotStoppedSpeaking` travelling upstream (from transport)
/// - `VADUserStartedSpeaking` / `VADUserStoppedSpeaking` downstream
/// - `Interruption` in either direction
///
/// The `turn_id` UUID written to `active_user_turn_id` and `active_bot_turn_id`
/// must be the same cells passed to `LLMUserAggregator::with_billing` and
/// `LLMAssistantAggregator::with_billing` so that transcript entries and audio
/// segments share the same identifier.
pub struct AudioCaptureProcessor {
    collector:           Arc<dyn AudioCaptureCollector>,
    active_user_turn_id: Arc<Mutex<Option<Uuid>>>,
    active_bot_turn_id:  Arc<Mutex<Option<Uuid>>>,
    state:               Mutex<State>,
}

impl AudioCaptureProcessor {
    pub fn new(
        collector: Arc<dyn AudioCaptureCollector>,
        active_user_turn_id: Arc<Mutex<Option<Uuid>>>,
        active_bot_turn_id:  Arc<Mutex<Option<Uuid>>>,
    ) -> FrameProcessor {
        FrameProcessor::new(
            "AudioCaptureProcessor",
            Box::new(Self {
                collector,
                active_user_turn_id,
                active_bot_turn_id,
                state: Mutex::new(State { user_turn: None, bot_turn: None }),
            }),
            false,
        )
    }

    fn flush_user(&self, interrupted: bool) {
        let turn = self.state.lock().unwrap().user_turn.take();
        if let Some(t) = turn {
            if !t.pcm.is_empty() {
                self.collector.record_segment(PendingAudioSegment {
                    segment_id:   Uuid::new_v4(),
                    turn_id:      Some(t.turn_id),
                    speaker:      "user",
                    pcm:          t.pcm,
                    sample_rate:  t.sample_rate,
                    num_channels: t.channels,
                    started_at:   t.started_at,
                    interrupted,
                });
            }
            *self.active_user_turn_id.lock().unwrap() = None;
        }
    }

    fn flush_bot(&self, interrupted: bool) {
        let turn = self.state.lock().unwrap().bot_turn.take();
        if let Some(t) = turn {
            if !t.pcm.is_empty() {
                self.collector.record_segment(PendingAudioSegment {
                    segment_id:   Uuid::new_v4(),
                    turn_id:      Some(t.turn_id),
                    speaker:      "bot",
                    pcm:          t.pcm,
                    sample_rate:  t.sample_rate,
                    num_channels: t.channels,
                    started_at:   t.started_at,
                    interrupted,
                });
            }
            *self.active_bot_turn_id.lock().unwrap() = None;
        }
    }
}

#[async_trait]
impl FrameHandler for AudioCaptureProcessor {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match &frame.inner {
            // ---- User turn lifecycle ----
            FrameInner::System(SystemFrame::VADUserStartedSpeaking { .. }) => {
                // Start accumulating user audio for this turn.
                let turn_id = Uuid::new_v4();
                *self.active_user_turn_id.lock().unwrap() = Some(turn_id);
                self.state.lock().unwrap().user_turn = Some(UserTurn {
                    turn_id,
                    pcm: Vec::new(),
                    sample_rate: 0,
                    channels: 1,
                    started_at: Utc::now(),
                });
                processor.push_frame(frame, direction).await?;
            }

            FrameInner::System(SystemFrame::InputAudioRaw(audio)) => {
                {
                    let mut state = self.state.lock().unwrap();
                    if let Some(turn) = &mut state.user_turn {
                        turn.pcm.extend_from_slice(&audio.audio);
                        turn.sample_rate = audio.sample_rate;
                        turn.channels = audio.num_channels;
                    }
                }
                processor.push_frame(frame, direction).await?;
            }

            FrameInner::System(SystemFrame::VADUserStoppedSpeaking { .. }) => {
                self.flush_user(false);
                processor.push_frame(frame, direction).await?;
            }

            // ---- Bot turn lifecycle (BotStarted/Stopped broadcast from output transport) ----
            FrameInner::System(SystemFrame::BotStartedSpeaking) => {
                let turn_id = Uuid::new_v4();
                *self.active_bot_turn_id.lock().unwrap() = Some(turn_id);
                self.state.lock().unwrap().bot_turn = Some(BotTurn {
                    turn_id,
                    pcm: Vec::new(),
                    sample_rate: 0,
                    channels: 1,
                    started_at: Utc::now(),
                });
                processor.push_frame(frame, direction).await?;
            }

            FrameInner::Data(DataFrame::OutputAudioRaw(audio)) => {
                {
                    let mut state = self.state.lock().unwrap();
                    if let Some(turn) = &mut state.bot_turn {
                        turn.pcm.extend_from_slice(&audio.audio);
                        turn.sample_rate = audio.sample_rate;
                        turn.channels = audio.num_channels;
                    }
                }
                processor.push_frame(frame, direction).await?;
            }

            FrameInner::System(SystemFrame::BotStoppedSpeaking) => {
                self.flush_bot(false);
                processor.push_frame(frame, direction).await?;
            }

            // ---- Interruption — bot audio cut short by user speech ----
            FrameInner::System(SystemFrame::Interruption) => {
                self.flush_bot(true);
                processor.push_frame(frame, direction).await?;
            }

            // ---- Session end — flush anything still in progress ----
            FrameInner::System(SystemFrame::Stop { .. } | SystemFrame::Cancel { .. }) => {
                self.flush_bot(true);
                self.flush_user(true);
                processor.push_frame(frame, direction).await?;
            }
            FrameInner::Control(ControlFrame::End { .. }) => {
                self.flush_bot(true);
                self.flush_user(true);
                processor.push_frame(frame, direction).await?;
            }

            _ => {
                processor.push_frame(frame, direction).await?;
            }
        }

        Ok(())
    }
}
