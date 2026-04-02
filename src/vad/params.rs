//! VAD configuration parameters.

pub const VAD_CONFIDENCE: f32 = 0.1;
pub const VAD_START_SECS: f32 = 0.4;
pub const VAD_STOP_SECS:  f32 = 0.4;
pub const VAD_MIN_VOLUME: f32 = 0.3;

#[derive(Debug, Clone)]
pub struct VadParams {
    pub confidence: f32,
    pub start_secs: f32,
    pub stop_secs:  f32,
    pub min_volume: f32,
}

impl Default for VadParams {
    fn default() -> Self {
        Self {
            confidence: VAD_CONFIDENCE,
            start_secs: VAD_START_SECS,
            stop_secs:  VAD_STOP_SECS,
            min_volume: VAD_MIN_VOLUME,
        }
    }
}