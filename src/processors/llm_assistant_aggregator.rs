//! LLM Assistant Aggregator.
//!
//! Collects LLMTextFrames during a bot response and saves the complete
//! assistant message to the shared LLMContext.

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
    in_response: bool,
}

/// Collects LLMTextFrames between LLMFullResponseStart and LLMFullResponseEnd,
/// then saves the complete assistant message to shared LLMContext.
///
/// LLMTextFrames pass through unchanged — TTS needs each chunk for streaming.
///
/// On interruption mid-response, the partial aggregation is discarded and
/// not saved to context — the model never "said" the interrupted portion.
pub struct LLMAssistantAggregator {
    context: Arc<Mutex<LLMContext>>,
    state: Mutex<State>,
}

impl LLMAssistantAggregator {
    pub fn new(context: Arc<Mutex<LLMContext>>) -> FrameProcessor {
        FrameProcessor::new(
            "LLMAssistantAggregator",
            Box::new(Self {
                context,
                state: Mutex::new(State {
                    aggregation: String::new(),
                    in_response: false,
                }),
            }),
            false,
        )
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
            FrameInner::System(SystemFrame::LLMFullResponseStart) => {
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

            FrameInner::System(SystemFrame::LLMFullResponseEnd) => {
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
                }

                processor.push_frame(frame, direction).await?;
            }

            // Interrupted mid-response — discard partial, don't save to context
            FrameInner::System(SystemFrame::Interruption) => {
                {
                    let mut state = self.state.lock().unwrap();
                    if state.in_response {
                        log::debug!(
                            "LLMAssistantAggregator: interrupted, discarding partial response"
                        );
                    }
                    state.aggregation.clear();
                    state.in_response = false;
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
