use std::sync::Arc;

pub trait BaseClock: Send + Sync {
    fn get_time(&self) -> f64;
}

pub struct SystemClock;

impl BaseClock for SystemClock {
    fn get_time(&self) -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
    }
}

pub fn system_clock() -> Arc<dyn BaseClock> {
    Arc::new(SystemClock)
}
