//! LLM User Aggregator.
//!
//! Collects TranscriptionFrames during a user turn and triggers LLM inference.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use log;

use crate::context::LLMContext;
use crate::error::Result;
use crate::frames::{
    DataFrame, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor, SystemFrame,
};

struct State {
    aggregation: String,
    user_speaking: bool,
    waiting_for_transcription: bool,
}

/// Collects TranscriptionFrames during a user turn.
///
/// When to trigger the LLM:
///   A. VADUserStoppedSpeaking arrives and aggregation is non-empty
///      (transcript(s) already arrived before VAD stop — normal fast path).
///   B. TranscriptionFrame arrives while user_speaking == false
///      (transcript arrived after VAD stop — the race condition fix).
///
/// Interruption:
///   On VADUserStartedSpeaking, if interruptions are allowed, broadcasts
///   InterruptionFrame in both directions so the bot stops immediately.
///
/// TranscriptionFrames are consumed here — not forwarded downstream.
/// Everything else passes through unchanged.
pub struct LLMUserAggregator {
    context: Arc<Mutex<LLMContext>>,
    state: Mutex<State>,
}

impl LLMUserAggregator {
    pub fn new(context: Arc<Mutex<LLMContext>>) -> FrameProcessor {
        FrameProcessor::new(
            "LLMUserAggregator",
            Box::new(Self {
                context,
                state: Mutex::new(State {
                    aggregation: String::new(),
                    user_speaking: false,
                    waiting_for_transcription: false,
                }),
            }),
            false,
        )
    }

    /// Flush aggregation to context and trigger LLM.
    /// Returns true if LLM was triggered.
    async fn flush_and_trigger(&self, processor: &FrameProcessor) -> Result<bool> {
        let aggregation = {
            let mut state = self.state.lock().unwrap();
            let text = state.aggregation.trim().to_string();
            state.aggregation.clear();
            text
        };

        if aggregation.is_empty() {
            log::debug!("LLMUserAggregator: empty turn, skipping LLM trigger");
            return Ok(false);
        }

        log::info!("LLMUserAggregator: user said: '{}'", aggregation);

        self.context.lock().unwrap().add_user_message(&aggregation);

        processor
            .push_frame(
                Frame::llm_context(self.context.clone()),
                FrameDirection::Downstream,
            )
            .await?;

        Ok(true)
    }
}

#[async_trait]
impl FrameHandler for LLMUserAggregator {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match &frame.inner {
            FrameInner::System(SystemFrame::VADUserStartedSpeaking { .. }) => {
                {
                    let mut state = self.state.lock().unwrap();
                    state.user_speaking = true;
                    state.waiting_for_transcription = false;
                }
                processor.push_frame(frame, direction).await?;
                // Broadcast interruption so bot stops immediately
                if processor.interruptions_allowed() {
                    processor.broadcast_interruption().await?;
                }
            }

            FrameInner::System(SystemFrame::VADUserStoppedSpeaking { .. }) => {
                self.state.lock().unwrap().user_speaking = false;
                // Push VAD frame first so downstream sees it before LLMContextFrame
                processor.push_frame(frame, direction).await?;
                // Path A: transcripts already arrived — flush now
                let triggered = self.flush_and_trigger(processor).await?;
                if !triggered {
                    // VAD done but transcription not yet arrived — arm the race-condition fix
                    self.state.lock().unwrap().waiting_for_transcription = true;
                }
            }

            FrameInner::Data(DataFrame::Transcription(t)) => {
                let text = t.text.trim().to_string();
                if text.is_empty() {
                    return Ok(());
                }

                let should_trigger = {
                    let mut state = self.state.lock().unwrap();
                    if state.aggregation.is_empty() {
                        state.aggregation = text;
                    } else {
                        state.aggregation.push(' ');
                        state.aggregation.push_str(&text);
                    }
                    // Path B: only trigger if VAD already stopped with an empty buffer
                    if state.waiting_for_transcription {
                        state.waiting_for_transcription = false;
                        true
                    } else {
                        false
                    }
                };

                if should_trigger {
                    self.flush_and_trigger(processor).await?;
                }
                // Transcription consumed — not forwarded
            }

            _ => {
                processor.push_frame(frame, direction).await?;
            }
        }

        Ok(())
    }
}
