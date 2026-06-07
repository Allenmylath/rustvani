use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use super::segment::AudioSegmentMeta;

/// Pluggable back-end for audio segment storage.
///
/// Two-step contract: store the encoded bytes (returns a URL/path), then
/// persist the metadata row. Implementations may combine both into one call.
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
}

pub mod local;

#[cfg(feature = "db-postgres")]
pub mod postgres;
