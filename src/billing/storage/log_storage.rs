use async_trait::async_trait;

use crate::error::Result;
use super::super::events::{BillingEvent, SessionSummary};
use super::BillingStorage;

/// Writes billing data as structured JSON log lines at INFO level.
/// Zero-dependency fallback — always available when `DATABASE_URL` is unset.
pub struct LogBillingStorage;

#[async_trait]
impl BillingStorage for LogBillingStorage {
    async fn record_event(&self, event: &BillingEvent) -> Result<()> {
        log::info!(
            "billing_event: {}",
            serde_json::to_string(event).unwrap_or_default()
        );
        Ok(())
    }

    async fn finalize_session(&self, summary: &SessionSummary) -> Result<()> {
        log::info!(
            "billing_summary: {}",
            serde_json::to_string(summary).unwrap_or_default()
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::events::{TranscriptEntry, TranscriptRole};
    use chrono::Utc;
    use uuid::Uuid;

    #[tokio::test]
    async fn record_event_returns_ok_for_all_variants() {
        let storage = LogBillingStorage;
        let id = Uuid::new_v4();
        let now = Utc::now();
        let events = vec![
            BillingEvent::SessionStart { session_id: id, started_at: now, metadata: Default::default() },
            BillingEvent::SessionEnd   { session_id: id, ended_at: now, finish_reason: "end".into() },
            BillingEvent::LlmUsage     { session_id: id, provider: "openai".into(), model: "gpt-4o".into(),
                                          input_tokens: 100, output_tokens: 50, estimated: false, occurred_at: now },
            BillingEvent::TtsUsage     { session_id: id, provider: "deepgram".into(), voice: "v".into(),
                                          char_count: 80, occurred_at: now },
            BillingEvent::SttUsage     { session_id: id, provider: "gnani".into(), audio_duration_ms: 1500.0, occurred_at: now },
            BillingEvent::Transcript(TranscriptEntry {
                turn_id: Uuid::new_v4(), session_id: id,
                role: TranscriptRole::User, text: "hello".into(),
                language: Some("en-IN".into()), interrupted: false, occurred_at: now,
            }),
        ];
        for ev in &events {
            assert!(storage.record_event(ev).await.is_ok(), "record_event failed for {:?}", ev);
        }
    }

    #[tokio::test]
    async fn finalize_session_returns_ok() {
        let storage = LogBillingStorage;
        let summary = SessionSummary {
            session_id: Uuid::new_v4(),
            llm_input_tokens: 200,
            llm_output_tokens: 100,
            llm_calls: 2,
            tts_chars: 500,
            tts_calls: 3,
            stt_audio_ms: 7000.0,
            stt_calls: 4,
            ..Default::default()
        };
        assert!(storage.finalize_session(&summary).await.is_ok());
    }
}
