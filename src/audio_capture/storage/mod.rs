use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::error::Result;
use super::segment::AudioSegmentMeta;

/// One captured speaking turn with its wall-clock start time.
/// Passed to `finalize_recording` so storage implementations can place
/// each turn at its correct position in the final mixed recording.
pub struct RecordedSegment {
    /// "user" | "bot"
    pub speaker:     &'static str,
    /// Raw PCM (16-bit signed little-endian).
    pub pcm:         Vec<u8>,
    pub sample_rate:  u32,
    pub num_channels: u16,
    /// When this turn started (wall clock) — used to compute position offset.
    pub started_at:   DateTime<Utc>,
}

/// Pluggable back-end for audio segment storage.
#[async_trait]
pub trait AudioStorage: Send + Sync {
    /// Write encoded audio bytes and return a stable URL or file path.
    async fn store_segment(
        &self,
        session_id: Uuid,
        segment_id: Uuid,
        speaker: &str,
        data: &[u8],
    ) -> Result<String>;

    /// Persist the segment metadata (including the URL from `store_segment`).
    async fn save_metadata(&self, session_id: Uuid, meta: &AudioSegmentMeta) -> Result<()>;

    /// Called once at session end with every recorded segment and its timestamp.
    /// The default is a no-op — override to produce a single mixed recording.
    async fn finalize_recording(
        &self,
        _session_id: Uuid,
        _segments: &[RecordedSegment],
    ) -> Result<()> {
        Ok(())
    }
}

pub mod local;

#[cfg(feature = "db-postgres")]
pub mod postgres;
