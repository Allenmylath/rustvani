//! LLM Assistant Aggregator.
//!
//! Collects LLMTextFrames during a bot response and saves the complete
//! assistant message to the shared LLMContext.

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
    ControlFrame, DataFrame, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor,
    SystemFrame,
};

struct State {
    aggregation: String,
    in_response: bool,
}

/// Collects LLMTextFrames between LLMFullResponseStart and LLMFullResponseEnd,
/// then saves the complete assistant message to shared LLMContext.
///
/// LLMTextFrames pass through unchanged — TTS needs each chunk for streaming.
///
/// On interruption mid-response, the partial aggregation is discarded from
/// context but recorded to the transcript as `interrupted: true` — the user
/// heard those words even though the model did not "complete" its turn.
pub struct LLMAssistantAggregator {
    context: Arc<Mutex<LLMContext>>,
    billing: Option<Arc<dyn BillingCollector>>,
    /// Shared with AudioCaptureProcessor — read here to link transcript to audio segment.
    active_bot_turn_id: Arc<Mutex<Option<Uuid>>>,
    state: Mutex<State>,
}

impl LLMAssistantAggregator {
    pub fn new(context: Arc<Mutex<LLMContext>>) -> FrameProcessor {
        FrameProcessor::new(
            "LLMAssistantAggregator",
            Box::new(Self {
                context,
                billing: None,
                active_bot_turn_id: Arc::new(Mutex::new(None)),
                state: Mutex::new(State {
                    aggregation: String::new(),
                    in_response: false,
                }),
            }),
            false,
        )
    }

    /// Create an aggregator that records transcript entries via the billing collector.
    /// Pass the same `active_bot_turn_id` cell as `AudioCaptureProcessor` so that
    /// transcript entries share the same `turn_id` as the bot audio segment.
    pub fn with_billing(
        context: Arc<Mutex<LLMContext>>,
        billing: Arc<dyn BillingCollector>,
        active_bot_turn_id: Arc<Mutex<Option<Uuid>>>,
    ) -> FrameProcessor {
        FrameProcessor::new(
            "LLMAssistantAggregator",
            Box::new(Self {
                context,
                billing: Some(billing),
                active_bot_turn_id,
                state: Mutex::new(State {
                    aggregation: String::new(),
                    in_response: false,
                }),
            }),
            false,
        )
    }

    fn record_turn(&self, text: &str, interrupted: bool) {
        if let Some(billing) = &self.billing {
            let turn_id = self.active_bot_turn_id
                .lock().unwrap()
                .unwrap_or_else(Uuid::new_v4);
            billing.record_transcript(TranscriptEntry {
                turn_id,
                session_id: billing.session_id(),
                role: TranscriptRole::Assistant,
                text: text.to_string(),
                language: None,
                interrupted,
                occurred_at: Utc::now(),
            });
        }
    }
}

#[async_trait]
impl FrameHandler for LLMAssistantAggregator {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match &frame.inner {
            FrameInner::Control(ControlFrame::LLMFullResponseStart) => {
                {
                    let mut state = self.state.lock().unwrap();
                    state.in_response = true;
                    state.aggregation.clear();
                }
                processor.push_frame(frame, direction).await?;
            }

            FrameInner::Data(DataFrame::LLMText(text)) => {
                let text = text.clone();
                {
                    let mut state = self.state.lock().unwrap();
                    if state.in_response {
                        state.aggregation.push_str(&text);
                    }
                }
                // Always pass downstream — TTS needs each chunk
                processor.push_frame(frame, direction).await?;
            }

            FrameInner::Control(ControlFrame::LLMFullResponseEnd) => {
                let aggregation = {
                    let mut state = self.state.lock().unwrap();
                    state.in_response = false;
                    let text = state.aggregation.trim().to_string();
                    state.aggregation.clear();
                    text
                };

                if !aggregation.is_empty() {
                    log::info!(
                        "LLMAssistantAggregator: assistant said: '{}'",
                        aggregation
                    );
                    self.context
                        .lock()
                        .unwrap()
                        .add_assistant_message(&aggregation);
                    self.record_turn(&aggregation, false);
                }

                processor.push_frame(frame, direction).await?;
            }

            // Interrupted mid-response — discard partial from context, but record
            // to transcript as interrupted (the user heard those words).
            FrameInner::System(SystemFrame::Interruption) => {
                let partial = {
                    let mut state = self.state.lock().unwrap();
                    let text = state.aggregation.trim().to_string();
                    state.aggregation.clear();
                    let was_in = state.in_response;
                    state.in_response = false;
                    if was_in {
                        log::debug!(
                            "LLMAssistantAggregator: interrupted, discarding partial response"
                        );
                    }
                    if was_in { text } else { String::new() }
                };
                if !partial.is_empty() {
                    self.record_turn(&partial, true);
                }
                processor.push_frame(frame, direction).await?;
            }

            _ => {
                processor.push_frame(frame, direction).await?;
            }
        }

        Ok(())
    }
}
