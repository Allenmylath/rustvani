//! LLM User Aggregator.
//!
//! Collects TranscriptionFrames during a user turn and triggers LLM inference.
//!
//! ── Contract with the STT stage ────────────────────────────────────────────
//!
//! This aggregator assumes an upstream STT handler that enforces the TurnGate
//! invariants (see `services/stt/sarvam.rs`):
//!
//!   1. Transcript-before-stop: a turn's TranscriptionFrame(s) always reach
//!      this processor BEFORE the VADUserStoppedSpeaking that closes the
//!      turn (the gate stashes the stop and releases it after the
//!      transcript, or after a release timeout).
//!   2. Fatherhood by construction: with audio gating enabled, every
//!      transcript descends from a local-VAD-attested turn — Sarvam never
//!      receives inter-turn audio, so spurious transcripts cannot exist.
//!
//! Given those, the old "Path B" (trigger the LLM when a transcript arrives
//! after VAD stop) is removed entirely. There is exactly ONE trigger path:
//! VADUserStoppedSpeaking on an open turn. A transcript can never start an
//! LLM turn on its own.
//!
//! ── Turn semantics ──────────────────────────────────────────────────────────
//!
//! VADUserStartedSpeaking  → turn opens. The aggregation buffer is NOT
//!                           cleared: if a previous segment's text is still
//!                           buffered (barge-in / continuation), it merges
//!                           into this turn. Interruption is broadcast as
//!                           before.
//! TranscriptionFrame      → consumed (never forwarded).
//!                           turn open  → appended to the aggregation buffer.
//!                           turn closed→ disposition by `LateTranscriptPolicy`:
//!                             Defer (default): held in a deferred buffer and
//!                               prepended to the NEXT flush, so the user's
//!                               words are never lost and never trigger a
//!                               weird seconds-late bot reply on their own.
//!                             Discard: dropped with a warning. Use this if
//!                               audio gating is disabled upstream, where a
//!                               closed-turn transcript may be server-VAD
//!                               noise rather than real speech.
//! VADUserStoppedSpeaking  → forwarded downstream FIRST (so downstream sees
//!                           it before the LLMContextFrame), then the turn
//!                           closes and flushes: deferred + aggregation are
//!                           combined, written to the context, recorded for
//!                           billing, and the LLM is triggered. An empty
//!                           combined turn triggers nothing.
//!
//! A VADUserStoppedSpeaking arriving with no open turn is forwarded but
//! ignored (defensive: the gate's exactly-once release should make this
//! unreachable).

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

/// What to do with a transcript that arrives when no turn is open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LateTranscriptPolicy {
    /// Hold the text and prepend it to the next flushed turn (default).
    /// Safe when the upstream STT handler gates audio: a closed-turn
    /// transcript is then guaranteed to be real, just slow.
    Defer,
    /// Drop the text. Use when audio gating is disabled upstream and a
    /// closed-turn transcript may be hallucinated from inter-turn noise.
    Discard,
}

struct State {
    /// Text accumulated for the currently open / continuing utterance.
    aggregation: String,
    /// Text from transcripts that arrived after their turn closed; prepended
    /// to the next flush so it is answered without triggering on its own.
    deferred: String,
    /// True between VADUserStartedSpeaking and the (gated, hence
    /// transcript-ordered) VADUserStoppedSpeaking.
    turn_open: bool,
}

/// Collects TranscriptionFrames during a user turn.
pub struct LLMUserAggregator {
    context: Arc<Mutex<LLMContext>>,
    billing: Option<Arc<dyn BillingCollector>>,
    /// Shared with AudioCaptureProcessor — read here to link transcript to audio segment.
    active_user_turn_id: Arc<Mutex<Option<Uuid>>>,
    policy: LateTranscriptPolicy,
    state: Mutex<State>,
}

impl LLMUserAggregator {
    pub fn new(context: Arc<Mutex<LLMContext>>) -> FrameProcessor {
        Self::build(context, None, Arc::new(Mutex::new(None)), LateTranscriptPolicy::Defer)
    }

    pub fn new_with_policy(
        context: Arc<Mutex<LLMContext>>,
        policy: LateTranscriptPolicy,
    ) -> FrameProcessor {
        Self::build(context, None, Arc::new(Mutex::new(None)), policy)
    }

    /// Create an aggregator that records transcript entries via the billing collector.
    /// Pass the same `active_user_turn_id` cell as `AudioCaptureProcessor` so that
    /// transcript entries share the same `turn_id` as the audio segment.
    pub fn with_billing(
        context: Arc<Mutex<LLMContext>>,
        billing: Arc<dyn BillingCollector>,
        active_user_turn_id: Arc<Mutex<Option<Uuid>>>,
    ) -> FrameProcessor {
        Self::build(context, Some(billing), active_user_turn_id, LateTranscriptPolicy::Defer)
    }

    pub fn with_billing_and_policy(
        context: Arc<Mutex<LLMContext>>,
        billing: Arc<dyn BillingCollector>,
        active_user_turn_id: Arc<Mutex<Option<Uuid>>>,
        policy: LateTranscriptPolicy,
    ) -> FrameProcessor {
        Self::build(context, Some(billing), active_user_turn_id, policy)
    }

    fn build(
        context: Arc<Mutex<LLMContext>>,
        billing: Option<Arc<dyn BillingCollector>>,
        active_user_turn_id: Arc<Mutex<Option<Uuid>>>,
        policy: LateTranscriptPolicy,
    ) -> FrameProcessor {
        FrameProcessor::new(
            "LLMUserAggregator",
            Box::new(Self {
                context,
                billing,
                active_user_turn_id,
                policy,
                state: Mutex::new(State {
                    aggregation: String::new(),
                    deferred:    String::new(),
                    turn_open:   false,
                }),
            }),
            false,
        )
    }

    /// Combine deferred + aggregation, clear both, write the turn to the
    /// context, record billing, and trigger the LLM. No-op on empty text.
    /// Returns true if the LLM was triggered.
    async fn flush_and_trigger(&self, processor: &FrameProcessor) -> Result<bool> {
        let text = {
            let mut state = self.state.lock().unwrap();
            let combined = combine(&state.deferred, &state.aggregation);
            state.deferred.clear();
            state.aggregation.clear();
            combined
        };

        if text.is_empty() {
            log::debug!("LLMUserAggregator: empty turn, skipping LLM trigger");
            return Ok(false);
        }

        log::info!("LLMUserAggregator: user said: '{}'", text);

        self.context.lock().unwrap().add_user_message(&text);

        if let Some(billing) = &self.billing {
            let turn_id = self.active_user_turn_id
                .lock().unwrap()
                .unwrap_or_else(Uuid::new_v4);
            billing.record_transcript(TranscriptEntry {
                turn_id,
                session_id: billing.session_id(),
                role: TranscriptRole::User,
                text: text.clone(),
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

/// Join deferred and current-turn text into one user utterance.
fn combine(deferred: &str, aggregation: &str) -> String {
    let d = deferred.trim();
    let a = aggregation.trim();
    match (d.is_empty(), a.is_empty()) {
        (true,  true)  => String::new(),
        (false, true)  => d.to_string(),
        (true,  false) => a.to_string(),
        (false, false) => format!("{} {}", d, a),
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
                // Open (or continue) the turn. The aggregation buffer is
                // deliberately NOT cleared: under the gate's linearized
                // emission, a transcript that landed just before this start
                // belongs to the same continuing utterance (barge-in merge).
                self.state.lock().unwrap().turn_open = true;
                processor.push_frame(frame, direction).await?;
                // Broadcast interruption so bot stops immediately
                if processor.interruptions_allowed() {
                    processor.broadcast_interruption().await?;
                }
            }

            FrameInner::System(SystemFrame::VADUserStoppedSpeaking { .. }) => {
                let was_open = {
                    let mut state = self.state.lock().unwrap();
                    let was_open = state.turn_open;
                    state.turn_open = false;
                    was_open
                };

                // Push VAD frame first so downstream sees it before LLMContextFrame
                processor.push_frame(frame, direction).await?;

                if was_open {
                    // The ONLY trigger path. The gate guarantees this frame
                    // arrives after the turn's transcript(s).
                    self.flush_and_trigger(processor).await?;
                } else {
                    log::warn!(
                        "LLMUserAggregator: VadStop with no open turn — ignored \
                         (gate should make this unreachable)"
                    );
                }
            }

            FrameInner::Data(DataFrame::Transcription(t)) => {
                // Transcripts are consumed here — never forwarded, and never
                // a trigger on their own (Path B removed).
                let text = t.text.trim().to_string();
                if text.is_empty() {
                    return Ok(());
                }

                let mut state = self.state.lock().unwrap();
                if state.turn_open {
                    if state.aggregation.is_empty() {
                        state.aggregation = text;
                    } else {
                        state.aggregation.push(' ');
                        state.aggregation.push_str(&text);
                    }
                } else {
                    match self.policy {
                        LateTranscriptPolicy::Defer => {
                            log::info!(
                                "LLMUserAggregator: late transcript deferred to next \
                                 turn: '{}'",
                                text
                            );
                            if state.deferred.is_empty() {
                                state.deferred = text;
                            } else {
                                state.deferred.push(' ');
                                state.deferred.push_str(&text);
                            }
                        }
                        LateTranscriptPolicy::Discard => {
                            log::warn!(
                                "LLMUserAggregator: transcript with no open turn \
                                 discarded: '{}'",
                                text
                            );
                        }
                    }
                }
            }

            _ => {
                processor.push_frame(frame, direction).await?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combine_joins_deferred_and_current() {
        assert_eq!(combine("", ""), "");
        assert_eq!(combine("cancel cheyyanam", ""), "cancel cheyyanam");
        assert_eq!(combine("", "what is the status"), "what is the status");
        assert_eq!(
            combine("cancel cheyyanam", "of my order"),
            "cancel cheyyanam of my order"
        );
        assert_eq!(combine("  padded  ", "  text  "), "padded text");
    }
}
