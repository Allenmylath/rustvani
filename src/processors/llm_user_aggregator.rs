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
}

/// Collects TranscriptionFrames during a user turn.
///
/// When VADUserStoppedSpeaking arrives:
///   1. Appends aggregated text to shared LLMContext as a user message.
///   2. Pushes an LLMContextFrame downstream to trigger the LLM.
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
                }),
            }),
            false,
        )
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
            // Accumulate transcript text — consumed, not forwarded
            FrameInner::Data(DataFrame::Transcription(t)) => {
                let text = t.text.trim().to_string();
                if !text.is_empty() {
                    let mut state = self.state.lock().unwrap();
                    if !state.aggregation.is_empty() {
                        state.aggregation.push(' ');
                    }
                    state.aggregation.push_str(&text);
                    log::debug!("LLMUserAggregator: accumulated '{}'", text);
                }
            }

            // User turn ended — flush to context and trigger LLM
            FrameInner::System(SystemFrame::VADUserStoppedSpeaking { .. }) => {
                // Push VAD frame through first
                processor.push_frame(frame, direction).await?;

                let aggregation = {
                    let mut state = self.state.lock().unwrap();
                    let text = state.aggregation.trim().to_string();
                    state.aggregation.clear();
                    text
                };

                if aggregation.is_empty() {
                    log::debug!("LLMUserAggregator: empty turn, skipping LLM trigger");
                    return Ok(());
                }

                log::info!("LLMUserAggregator: user said: '{}'", aggregation);

                // Append user message to shared context
                self.context.lock().unwrap().add_message("user", &aggregation);

                // Trigger the LLM
                processor
                    .push_frame(
                        Frame::llm_context(self.context.clone()),
                        FrameDirection::Downstream,
                    )
                    .await?;
            }

            // Clear partial aggregation on interruption
            FrameInner::System(SystemFrame::Interruption) => {
                self.state.lock().unwrap().aggregation.clear();
                processor.push_frame(frame, direction).await?;
            }

            _ => {
                processor.push_frame(frame, direction).await?;
            }
        }

        Ok(())
    }
}
