use std::sync::Arc;

use async_trait::async_trait;
use tokio_postgres::Client;
use uuid::Uuid;

use crate::error::{PipecatError, Result};
use super::super::segment::AudioSegmentMeta;
use super::AudioStorage;

/// Persists audio segment metadata to PostgreSQL.
///
/// Only handles `save_metadata` — binary storage is delegated to another
/// `AudioStorage` impl (e.g. `LocalAudioStorage`) which also implements
/// `store_segment`. Compose them or wrap in a combined impl as needed.
///
/// Call `run_migrations()` once at startup to create the table.
pub struct PostgresAudioMetaStorage {
    client: Arc<Client>,
}

impl PostgresAudioMetaStorage {
    pub fn new(client: Arc<Client>) -> Self {
        Self { client }
    }

    pub async fn run_migrations(client: &Client) -> Result<()> {
        client.batch_execute(SCHEMA_SQL).await
            .map_err(|e| PipecatError::pipeline(format!("audio migration failed: {e}")))?;
        Ok(())
    }
}

#[async_trait]
impl AudioStorage for PostgresAudioMetaStorage {
    async fn store_segment(
        &self,
        _session_id: Uuid,
        _segment_id: Uuid,
        _speaker: &str,
        _data: &[u8],
    ) -> Result<String> {
        // This impl only handles metadata — binary storage must come from a
        // wrapping impl. Return empty string; callers should not use this alone.
        Ok(String::new())
    }

    async fn save_metadata(&self, session_id: Uuid, meta: &AudioSegmentMeta) -> Result<()> {
        self.client
            .execute(
                "INSERT INTO session_audio_segments
                 (session_id, turn_id, speaker, audio_url, format,
                  sample_rate, num_channels, duration_ms, byte_size,
                  interrupted, started_at, ended_at)
                 VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)",
                &[
                    &session_id,
                    &meta.turn_id,
                    &meta.speaker,
                    &meta.audio_url,
                    &meta.format,
                    &(meta.sample_rate as i32),
                    &(meta.num_channels as i16),
                    &meta.duration_ms,
                    &(meta.byte_size as i64),
                    &meta.interrupted,
                    &meta.started_at,
                    &meta.ended_at,
                ],
            )
            .await
            .map_err(|e| PipecatError::pipeline(format!("audio save_metadata: {e}")))?;
        Ok(())
    }
}

const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS session_audio_segments (
    id           BIGSERIAL   PRIMARY KEY,
    session_id   UUID        NOT NULL,
    turn_id      UUID,
    speaker      TEXT        NOT NULL CHECK (speaker IN ('user', 'bot')),
    audio_url    TEXT        NOT NULL,
    format       TEXT        NOT NULL DEFAULT 'wav',
    sample_rate  INTEGER     NOT NULL,
    num_channels SMALLINT    NOT NULL,
    duration_ms  FLOAT8      NOT NULL,
    byte_size    BIGINT      NOT NULL,
    interrupted  BOOLEAN     NOT NULL DEFAULT FALSE,
    started_at   TIMESTAMPTZ NOT NULL,
    ended_at     TIMESTAMPTZ NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audio_segments_session_id ON session_audio_segments (session_id);
CREATE INDEX IF NOT EXISTS idx_audio_segments_turn_id    ON session_audio_segments (turn_id);
";
