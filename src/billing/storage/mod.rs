use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use super::events::{BillingEvent, SessionSummary, TranscriptEntry};

/// Pluggable persistence back-end for billing data.
#[async_trait]
pub trait BillingStorage: Send + Sync {
    /// Per-event observability hook, called for every event as it arrives on the
    /// drain task (never on the pipeline's hot path). Durable persistence is done
    /// in [`checkpoint`](Self::checkpoint), not here — the default DB back-end
    /// treats this as a no-op and batches writes at checkpoint time instead.
    async fn record_event(&self, _event: &BillingEvent) -> Result<()> {
        Ok(())
    }

    /// Periodic durable checkpoint (≈ once per minute) plus a final flush at
    /// session end. Persists an **absolute snapshot** of the running totals and
    /// transcript (idempotent: totals are monotonic) and appends `new_events` —
    /// the billable events accumulated since the last successful checkpoint —
    /// to the ledger. Each event carries a stable id so a retried checkpoint
    /// never double-inserts.
    ///
    /// On crash, the last successful checkpoint is the billable amount, so the
    /// provider's exposure is bounded to one checkpoint interval and the client
    /// is never billed for the unsettled tail. Default is a no-op.
    async fn checkpoint(
        &self,
        _summary: &SessionSummary,
        _new_events: &[(Uuid, BillingEvent)],
        _transcripts: &[TranscriptEntry],
    ) -> Result<()> {
        Ok(())
    }

    /// Called once at clean session end with the aggregated summary and the
    /// complete ordered transcript. Marks the session settled (`complete`) and
    /// persists both together so there is exactly one transcript record per
    /// session. After this, the session's totals are final and durable —
    /// correctness does not depend on this call succeeding, since the last
    /// checkpoint already holds the billable amount.
    async fn finalize_session(
        &self,
        summary: &SessionSummary,
        transcripts: &[TranscriptEntry],
    ) -> Result<()>;
}

pub mod log_storage;

#[cfg(feature = "db-postgres")]
pub mod postgres;
