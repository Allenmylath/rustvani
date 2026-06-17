//! Billing reconciliation sweeper.
//!
//! Settles sessions whose process crashed mid-conversation: any session still
//! marked `active` whose heartbeat (`last_checkpoint_at`) is older than the
//! threshold is moved to `crashed`, with its last checkpoint value becoming the
//! billable amount. This bounds the provider's loss to one checkpoint interval
//! and never bills the client for the unsettled tail.
//!
//! The operation is a single idempotent UPDATE — safe to run on any schedule
//! (cron, k8s CronJob, Neon scheduled query) and harmless if it runs twice or
//! is skipped (the next run catches everything). Run once and exit:
//!
//!   cargo run --bin billing_sweeper
//!
//! Env:
//!   DATABASE_URL                  Postgres connection string (required)
//!   BILLING_CRASH_THRESHOLD_SECS  staleness cutoff in seconds (default 180)

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let url = std::env::var("DATABASE_URL").map_err(|_| "DATABASE_URL not set")?;
    let threshold_secs: f64 = std::env::var("BILLING_CRASH_THRESHOLD_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(180.0);

    let connector = native_tls::TlsConnector::builder().build()?;
    let tls = postgres_native_tls::MakeTlsConnector::new(connector);
    let (client, connection) = tokio_postgres::connect(&url, tls).await?;
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            log::error!("sweeper: connection dropped: {e}");
        }
    });

    // A session stuck 'active' with a stale (or missing) heartbeat = its process
    // died. Settle it at its last checkpoint; carry last_checkpoint_at into
    // ended_at when the session never recorded a clean end.
    let swept = client
        .execute(
            "UPDATE billing_sessions
             SET status   = 'crashed',
                 ended_at = COALESCE(ended_at, last_checkpoint_at),
                 updated_at = now()
             WHERE status = 'active'
               AND COALESCE(last_checkpoint_at, created_at) < now() - ($1 * interval '1 second')",
            &[&threshold_secs],
        )
        .await?;

    println!(
        "billing_sweeper: settled {swept} crashed session(s) (threshold {threshold_secs}s)"
    );
    Ok(())
}
