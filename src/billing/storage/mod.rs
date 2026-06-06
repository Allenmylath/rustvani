use async_trait::async_trait;

use crate::error::Result;
use super::events::{BillingEvent, SessionSummary};

/// Pluggable persistence back-end for billing data.
#[async_trait]
pub trait BillingStorage: Send + Sync {
    /// Called for every event as it arrives. Implementations may buffer.
    async fn record_event(&self, event: &BillingEvent) -> Result<()>;

    /// Called once when the session ends with the fully-aggregated summary.
    async fn finalize_session(&self, summary: &SessionSummary) -> Result<()>;
}

pub mod log_storage;

#[cfg(feature = "db-postgres")]
pub mod postgres;
