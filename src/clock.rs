use std::time::{SystemTime, UNIX_EPOCH};

/// Provides a monotonic or wall-clock timestamp (seconds since epoch).
pub trait BaseClock: Send + Sync {
    fn get_time(&self) -> f64;
}

/// Default wall-clock implementation using `std::time::SystemTime`.
pub struct SystemClock;

impl BaseClock for SystemClock {
    fn get_time(&self) -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
    }
}
