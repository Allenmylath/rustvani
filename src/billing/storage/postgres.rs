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

    /// Appends one billable event to the ledger, idempotently. A retried
    /// checkpoint re-sends the same `event_id`, so `ON CONFLICT DO NOTHING`
    /// guarantees no double-insert. Non-billable events (session start/end,
    /// transcript) are not stored as ledger rows.
    async fn insert_event(&self, event_id: &uuid::Uuid, event: &BillingEvent) -> Result<()> {
        let sid = event.session_id();
        let raw = serde_json::to_value(event).unwrap_or_default();
        let res = match event {
            BillingEvent::LlmUsage {
                provider, model, input_tokens, output_tokens, estimated, occurred_at, ..
            } => {
                self.client.execute(
                    "INSERT INTO billing_events
                     (event_id, session_id, event_type, provider, model,
                      input_tokens, output_tokens, estimated, occurred_at, raw_json)
                     VALUES ($1,$2,'llm',$3,$4,$5,$6,$7,$8,$9)
                     ON CONFLICT (event_id) DO NOTHING",
                    &[
                        event_id, &sid, provider, model,
                        &(*input_tokens as i32), &(*output_tokens as i32), estimated,
                        occurred_at, &raw,
                    ],
                ).await
            }
            BillingEvent::TtsUsage { provider, voice, char_count, occurred_at, .. } => {
                self.client.execute(
                    "INSERT INTO billing_events
                     (event_id, session_id, event_type, provider, voice, char_count,
                      occurred_at, raw_json)
                     VALUES ($1,$2,'tts',$3,$4,$5,$6,$7)
                     ON CONFLICT (event_id) DO NOTHING",
                    &[event_id, &sid, provider, voice, &(*char_count as i32), occurred_at, &raw],
                ).await
            }
            BillingEvent::SttUsage { provider, audio_duration_ms, occurred_at, .. } => {
                self.client.execute(
                    "INSERT INTO billing_events
                     (event_id, session_id, event_type, provider, audio_duration_ms,
                      occurred_at, raw_json)
                     VALUES ($1,$2,'stt',$3,$4,$5,$6)
                     ON CONFLICT (event_id) DO NOTHING",
                    &[event_id, &sid, provider, audio_duration_ms, occurred_at, &raw],
                ).await
            }
            // Not stored as ledger rows.
            BillingEvent::SessionStart { .. }
            | BillingEvent::SessionEnd { .. }
            | BillingEvent::Transcript(_) => return Ok(()),
        };
        res.map_err(|e| PipecatError::pipeline(format!("billing insert_event: {e}")))?;
        Ok(())
    }
}

#[async_trait]
impl BillingStorage for PostgresBillingStorage {
    // No-op: durable writes are batched at `checkpoint`, not per event. Keeping
    // this off the per-event path means a slow/unreachable DB never backs up the
    // billing queue one event at a time.
    async fn record_event(&self, _event: &BillingEvent) -> Result<()> {
        Ok(())
    }

    async fn checkpoint(
        &self,
        s: &SessionSummary,
        new_events: &[(uuid::Uuid, BillingEvent)],
        transcripts: &[TranscriptEntry],
    ) -> Result<()> {
        let meta = serde_json::to_value(&s.metadata)
            .unwrap_or(serde_json::Value::Object(Default::default()));
        let transcript_json = serde_json::to_value(transcripts)
            .unwrap_or(serde_json::Value::Array(vec![]));

        // 1. Absolute snapshot of the running totals + transcript. Idempotent:
        //    totals are monotonic, so a retried/late checkpoint is last-writer-wins.
        //    `status` stays 'active' and `last_checkpoint_at` is the crash heartbeat.
        self.client
            .execute(
                "INSERT INTO billing_sessions
                 (session_id, started_at,
                  llm_input_tokens, llm_output_tokens, llm_calls,
                  tts_chars, tts_calls, stt_audio_ms, stt_calls,
                  metadata, transcript_json, status, last_checkpoint_at,
                  created_at, updated_at)
                 VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,'active',now(),now(),now())
                 ON CONFLICT (session_id) DO UPDATE SET
                   started_at        = COALESCE(billing_sessions.started_at, EXCLUDED.started_at),
                   llm_input_tokens  = EXCLUDED.llm_input_tokens,
                   llm_output_tokens = EXCLUDED.llm_output_tokens,
                   llm_calls         = EXCLUDED.llm_calls,
                   tts_chars         = EXCLUDED.tts_chars,
                   tts_calls         = EXCLUDED.tts_calls,
                   stt_audio_ms      = EXCLUDED.stt_audio_ms,
                   stt_calls         = EXCLUDED.stt_calls,
                   metadata          = EXCLUDED.metadata,
                   transcript_json   = EXCLUDED.transcript_json,
                   status            = 'active',
                   last_checkpoint_at= now(),
                   updated_at        = now()",
                &[
                    &s.session_id, &s.started_at,
                    &(s.llm_input_tokens as i32), &(s.llm_output_tokens as i32),
                    &(s.llm_calls as i32),
                    &(s.tts_chars as i32), &(s.tts_calls as i32),
                    &s.stt_audio_ms, &(s.stt_calls as i32),
                    &meta, &transcript_json,
                ],
            )
            .await
            .map_err(|e| PipecatError::pipeline(format!("billing checkpoint snapshot: {e}")))?;

        // 2. Append the ledger rows for events since the last checkpoint.
        for (event_id, event) in new_events {
            self.insert_event(event_id, event).await?;
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
                  metadata, transcript_json, status, created_at, updated_at)
                 VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,'complete',$15,$15)
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
                   status            = 'complete',
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
    -- Billing lifecycle: 'active' (in flight, not yet billable),
    -- 'complete' (clean end, exact), 'crashed' (settled at last checkpoint).
    status            TEXT        NOT NULL DEFAULT 'active',
    -- Heartbeat: bumped on every checkpoint; the sweeper uses it to detect
    -- sessions whose process died mid-conversation.
    last_checkpoint_at TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Add columns to existing deployments that pre-date them.
ALTER TABLE billing_sessions
    ADD COLUMN IF NOT EXISTS transcript_json    JSONB NOT NULL DEFAULT '[]';
ALTER TABLE billing_sessions
    ADD COLUMN IF NOT EXISTS status             TEXT  NOT NULL DEFAULT 'active';
ALTER TABLE billing_sessions
    ADD COLUMN IF NOT EXISTS last_checkpoint_at TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS billing_events (
    id                BIGSERIAL   PRIMARY KEY,
    -- Stable per-event id assigned by the drain task; makes the ledger append
    -- idempotent so a retried checkpoint never double-inserts.
    event_id          UUID,
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

-- Add event_id to existing deployments that pre-date it.
ALTER TABLE billing_events
    ADD COLUMN IF NOT EXISTS event_id UUID;

CREATE INDEX        IF NOT EXISTS billing_sessions_started_at  ON billing_sessions (started_at);
CREATE INDEX        IF NOT EXISTS billing_sessions_status      ON billing_sessions (status, last_checkpoint_at);
CREATE INDEX        IF NOT EXISTS billing_sessions_metadata    ON billing_sessions USING GIN (metadata);
CREATE UNIQUE INDEX IF NOT EXISTS billing_events_event_id      ON billing_events   (event_id);
CREATE INDEX        IF NOT EXISTS billing_events_session_id    ON billing_events   (session_id);
CREATE INDEX        IF NOT EXISTS billing_events_occurred_at   ON billing_events   (occurred_at);
";
