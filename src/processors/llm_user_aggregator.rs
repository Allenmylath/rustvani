use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use log;

use crate::context::LLMContext;
use crate::error::Result;
use crate::frames::{
    DataFrame, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor, SystemFrame,
};

struct State {
    aggregation:   String,
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
/// This covers both orderings without any additional pending flag.
///
/// TranscriptionFrames are consumed here — not forwarded downstream.
/// Everything else passes through unchanged.
pub struct LLMUserAggregator {
    context: Arc<Mutex<LLMContext>>,
    state:   Mutex<State>,
}

impl LLMUserAggregator {
    pub fn new(context: Arc<Mutex<LLMContext>>) -> FrameProcessor {
        FrameProcessor::new(
            "LLMUserAggregator",
            Box::new(Self {
                context,
                state: Mutex::new(State {
                    aggregation:   String::new(),
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

        self.context.lock().unwrap().add_message("user", &aggregation);

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

                let user_speaking = {
                    let mut state = self.state.lock().unwrap();
                    if !state.aggregation.is_empty() {
                        state.aggregation.push(' ');
                    }
                    state.aggregation.push_str(&text);
                    log::debug!("LLMUserAggregator: accumulated '{}'", text);
                    state.user_speaking
                };

                if !user_speaking {
                    // Path B: VAD already stopped — trigger immediately
                    log::debug!("LLMUserAggregator: late transcript, triggering now");
                    self.flush_and_trigger(processor).await?;
                }
                // if user_speaking == true: just accumulated, VAD stop will flush
            }

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