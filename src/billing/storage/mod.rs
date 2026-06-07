use async_trait::async_trait;

use crate::error::Result;
use super::events::{BillingEvent, SessionSummary, TranscriptEntry};

/// Pluggable persistence back-end for billing data.
#[async_trait]
pub trait BillingStorage: Send + Sync {
    /// Called for every event as it arrives. Implementations may buffer.
    async fn record_event(&self, event: &BillingEvent) -> Result<()>;

    /// Called once at session end with the aggregated summary and the complete
    /// ordered transcript for the session. Implementations should persist both
    /// together so there is exactly one transcript record per session.
    async fn finalize_session(
        &self,
        summary: &SessionSummary,
        transcripts: &[TranscriptEntry],
    ) -> Result<()>;
}

pub mod log_storage;

#[cfg(feature = "db-postgres")]
pub mod postgres;
