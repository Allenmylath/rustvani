//! LLM User Aggregator.
//!
//! Collects TranscriptionFrames during a user turn and triggers LLM inference.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::Utc;
use log;
use uuid::Uuid;

use crate::billing::collector::BillingCollector;
use crate::billing::events::{TranscriptEntry, TranscriptRole};
use crate::context::LLMContext;
use crate::error::Result;
use crate::frames::{
    DataFrame, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor, SystemFrame,
};

struct State {
    aggregation: String,
    user_speaking: bool,
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
    billing: Option<Arc<dyn BillingCollector>>,
    /// Shared with AudioCaptureProcessor — read here to link transcript to audio segment.
    active_user_turn_id: Arc<Mutex<Option<Uuid>>>,
    state: Mutex<State>,
}

impl LLMUserAggregator {
    pub fn new(context: Arc<Mutex<LLMContext>>) -> FrameProcessor {
        FrameProcessor::new(
            "LLMUserAggregator",
            Box::new(Self {
                context,
                billing: None,
                active_user_turn_id: Arc::new(Mutex::new(None)),
                state: Mutex::new(State {
                    aggregation: String::new(),
                    user_speaking: false,
                }),
            }),
            false,
        )
    }

    /// Create an aggregator that records transcript entries via the billing collector.
    /// Pass the same `active_user_turn_id` cell as `AudioCaptureProcessor` so that
    /// transcript entries share the same `turn_id` as the audio segment.
    pub fn with_billing(
        context: Arc<Mutex<LLMContext>>,
        billing: Arc<dyn BillingCollector>,
        active_user_turn_id: Arc<Mutex<Option<Uuid>>>,
    ) -> FrameProcessor {
        FrameProcessor::new(
            "LLMUserAggregator",
            Box::new(Self {
                context,
                billing: Some(billing),
                active_user_turn_id,
                state: Mutex::new(State {
                    aggregation: String::new(),
                    user_speaking: false,
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

        if let Some(billing) = &self.billing {
            let turn_id = self.active_user_turn_id
                .lock().unwrap()
                .unwrap_or_else(Uuid::new_v4);
            billing.record_transcript(TranscriptEntry {
                turn_id,
                session_id: billing.session_id(),
                role: TranscriptRole::User,
                text: aggregation.clone(),
                language: None,
                interrupted: false,
                occurred_at: Utc::now(),
            });
        }

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
                self.state.lock().unwrap().user_speaking = true;
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
                self.flush_and_trigger(processor).await?;
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
                    // Path B: transcript arrived after VAD stop
                    !state.user_speaking
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