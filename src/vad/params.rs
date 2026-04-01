//! VAD configuration parameters.
//!
//! Constants and defaults copied directly from Pipecat's `vad_analyzer.py`.

// ---------------------------------------------------------------------------
// Defaults — exact values from Python
// ---------------------------------------------------------------------------

pub const VAD_CONFIDENCE: f32 = 0.1;
pub const VAD_START_SECS: f32 = 0.4;
pub const VAD_STOP_SECS:  f32 = 0.4;
pub const VAD_MIN_VOLUME: f32 = 0.01;

/// Configuration parameters for Voice Activity Detection.
///
/// Mirrors Python's `VADParams` exactly.
#[derive(Debug, Clone)]
pub struct VadParams {
    /// Minimum model confidence to consider audio as speech.
    pub confidence: f32,

    /// How long speech must be detected before transitioning STARTING → SPEAKING.
    pub start_secs: f32,

    /// How long silence must persist before transitioning STOPPING → QUIET.
    pub stop_secs: f32,

    /// Minimum RMS volume (0–1) below which audio is treated as silence
    /// regardless of model confidence.
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