use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use tokio_postgres::Client;

use crate::error::{PipecatError, Result};
use super::super::events::{BillingEvent, SessionSummary, TranscriptEntry};
use super::BillingStorage;

/// Persists billing data to PostgreSQL.
///
/// Transcript entries are NOT inserted row-by-row. Instead the complete ordered
/// transcript is serialised as a single JSON array and written to
/// `billing_sessions.transcript_json` at session end. This gives you exactly
/// one transcript record per session.
///
/// Call `PostgresBillingStorage::run_migrations()` once at startup to create
/// or update the schema.
pub struct PostgresBillingStorage {
    client: Arc<Client>,
}

impl PostgresBillingStorage {
    pub fn new(client: Client) -> Self {
        Self { client: Arc::new(client) }
    }

    pub fn with_arc(client: Arc<Client>) -> Self {
        Self { client }
    }

    /// Creates / updates the billing schema. Safe to call repeatedly.
    pub async fn run_migrations(client: &Client) -> Result<()> {
        client.batch_execute(SCHEMA_SQL).await
            .map_err(|e| PipecatError::pipeline(format!("billing migration failed: {e}")))?;
        Ok(())
    }
}

#[async_trait]
impl BillingStorage for PostgresBillingStorage {
    async fn record_event(&self, event: &BillingEvent) -> Result<()> {
        let sid = event.session_id();

        // Ensure the session row exists before any FK references.
        self.client
            .execute(
                "INSERT INTO billing_sessions (session_id) VALUES ($1)
                 ON CONFLICT (session_id) DO NOTHING",
                &[&sid],
            )
            .await
            .map_err(|e| PipecatError::pipeline(format!("billing upsert session: {e}")))?;

        match event {
            BillingEvent::SessionStart { started_at, metadata, .. } => {
                let meta = serde_json::to_value(metadata)
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                self.client
                    .execute(
                        "UPDATE billing_sessions
                         SET started_at = $2, metadata = $3, updated_at = now()
                         WHERE session_id = $1",
                        &[&sid, started_at, &meta],
                    )
                    .await
                    .map_err(|e| PipecatError::pipeline(format!("billing session_start: {e}")))?;
            }

            BillingEvent::SessionEnd { ended_at, finish_reason, .. } => {
                self.client
                    .execute(
                        "UPDATE billing_sessions
                         SET ended_at = $2, finish_reason = $3,
                             duration_secs = EXTRACT(EPOCH FROM ($2 - started_at)),
                             updated_at = now()
                         WHERE session_id = $1",
                        &[&sid, ended_at, finish_reason],
                    )
                    .await
                    .map_err(|e| PipecatError::pipeline(format!("billing session_end: {e}")))?;
            }

            BillingEvent::LlmUsage {
                provider, model,
                input_tokens, output_tokens, estimated,
                occurred_at, ..
            } => {
                let raw = serde_json::to_value(event).unwrap_or_default();
                self.client
                    .execute(
                        "INSERT INTO billing_events
                         (session_id, event_type, provider, model,
                          input_tokens, output_tokens, estimated,
                          occurred_at, raw_json)
                         VALUES ($1,'llm',$2,$3,$4,$5,$6,$7,$8)",
                        &[
                            &sid, provider, model,
                            &(*input_tokens as i32), &(*output_tokens as i32), estimated,
                            occurred_at, &raw,
                        ],
                    )
                    .await
                    .map_err(|e| PipecatError::pipeline(format!("billing llm_event: {e}")))?;
            }

            BillingEvent::TtsUsage { provider, voice, char_count, occurred_at, .. } => {
                let raw = serde_json::to_value(event).unwrap_or_default();
                self.client
                    .execute(
                        "INSERT INTO billing_events
                         (session_id, event_type, provider, voice, char_count,
                          occurred_at, raw_json)
                         VALUES ($1,'tts',$2,$3,$4,$5,$6)",
                        &[&sid, provider, voice, &(*char_count as i32), occurred_at, &raw],
                    )
                    .await
                    .map_err(|e| PipecatError::pipeline(format!("billing tts_event: {e}")))?;
            }

            BillingEvent::SttUsage { provider, audio_duration_ms, occurred_at, .. } => {
                let raw = serde_json::to_value(event).unwrap_or_default();
                self.client
                    .execute(
                        "INSERT INTO billing_events
                         (session_id, event_type, provider, audio_duration_ms,
                          occurred_at, raw_json)
                         VALUES ($1,'stt',$2,$3,$4,$5)",
                        &[&sid, provider, audio_duration_ms, occurred_at, &raw],
                    )
                    .await
                    .map_err(|e| PipecatError::pipeline(format!("billing stt_event: {e}")))?;
            }

            // Transcripts are collected and written as a single JSON at session
            // end via finalize_session — do nothing here.
            BillingEvent::Transcript(_) => {}
        }

        Ok(())
    }

    async fn finalize_session(
        &self,
        s: &SessionSummary,
        transcripts: &[TranscriptEntry],
    ) -> Result<()> {
        let meta = serde_json::to_value(&s.metadata)
            .unwrap_or(serde_json::Value::Object(Default::default()));

        let transcript_json = serde_json::to_value(transcripts)
            .unwrap_or(serde_json::Value::Array(vec![]));

        let now = Utc::now();

        self.client
            .execute(
                "INSERT INTO billing_sessions
                 (session_id, started_at, ended_at, duration_secs, finish_reason,
                  llm_input_tokens, llm_output_tokens, llm_calls,
                  tts_chars, tts_calls,
                  stt_audio_ms, stt_calls,
                  metadata, transcript_json, created_at, updated_at)
                 VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$15)
                 ON CONFLICT (session_id) DO UPDATE SET
                   started_at        = COALESCE(EXCLUDED.started_at,        billing_sessions.started_at),
                   ended_at          = COALESCE(EXCLUDED.ended_at,          billing_sessions.ended_at),
                   duration_secs     = COALESCE(EXCLUDED.duration_secs,     billing_sessions.duration_secs),
                   finish_reason     = COALESCE(EXCLUDED.finish_reason,     billing_sessions.finish_reason),
                   llm_input_tokens  = EXCLUDED.llm_input_tokens,
                   llm_output_tokens = EXCLUDED.llm_output_tokens,
                   llm_calls         = EXCLUDED.llm_calls,
                   tts_chars         = EXCLUDED.tts_chars,
                   tts_calls         = EXCLUDED.tts_calls,
                   stt_audio_ms      = EXCLUDED.stt_audio_ms,
                   stt_calls         = EXCLUDED.stt_calls,
                   metadata          = EXCLUDED.metadata,
                   transcript_json   = EXCLUDED.transcript_json,
                   updated_at        = EXCLUDED.updated_at",
                &[
                    &s.session_id, &s.started_at, &s.ended_at,
                    &s.duration_secs, &s.finish_reason,
                    &(s.llm_input_tokens as i32), &(s.llm_output_tokens as i32),
                    &(s.llm_calls as i32),
                    &(s.tts_chars as i32), &(s.tts_calls as i32),
                    &s.stt_audio_ms, &(s.stt_calls as i32),
                    &meta, &transcript_json, &now,
                ],
            )
            .await
            .map_err(|e| PipecatError::pipeline(format!("billing finalize: {e}")))?;

        Ok(())
    }
}

const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS billing_sessions (
    session_id        UUID        PRIMARY KEY,
    started_at        TIMESTAMPTZ,
    ended_at          TIMESTAMPTZ,
    duration_secs     FLOAT8,
    finish_reason     TEXT,
    llm_input_tokens  INTEGER     NOT NULL DEFAULT 0,
    llm_output_tokens INTEGER     NOT NULL DEFAULT 0,
    llm_calls         INTEGER     NOT NULL DEFAULT 0,
    tts_chars         INTEGER     NOT NULL DEFAULT 0,
    tts_calls         INTEGER     NOT NULL DEFAULT 0,
    stt_audio_ms      FLOAT8      NOT NULL DEFAULT 0,
    stt_calls         INTEGER     NOT NULL DEFAULT 0,
    metadata          JSONB       NOT NULL DEFAULT '{}',
    transcript_json   JSONB       NOT NULL DEFAULT '[]',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Add transcript_json to existing deployments that pre-date this column.
ALTER TABLE billing_sessions
    ADD COLUMN IF NOT EXISTS transcript_json JSONB NOT NULL DEFAULT '[]';

CREATE TABLE IF NOT EXISTS billing_events (
    id                BIGSERIAL   PRIMARY KEY,
    session_id        UUID        NOT NULL REFERENCES billing_sessions(session_id) ON DELETE CASCADE,
    event_type        TEXT        NOT NULL,
    provider          TEXT,
    model             TEXT,
    input_tokens      INTEGER,
    output_tokens     INTEGER,
    estimated         BOOLEAN,
    char_count        INTEGER,
    voice             TEXT,
    audio_duration_ms FLOAT8,
    occurred_at       TIMESTAMPTZ NOT NULL,
    raw_json          JSONB
);

CREATE INDEX IF NOT EXISTS billing_sessions_started_at  ON billing_sessions (started_at);
CREATE INDEX IF NOT EXISTS billing_sessions_metadata    ON billing_sessions  USING GIN (metadata);
CREATE INDEX IF NOT EXISTS billing_events_session_id    ON billing_events    (session_id);
CREATE INDEX IF NOT EXISTS billing_events_occurred_at   ON billing_events    (occurred_at);
";
