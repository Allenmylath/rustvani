//! VAD configuration parameters.
//!
//! # Calibration notes
//!
//! These constants are calibrated for the Silero VAD v4.0 ONNX model
//! (input: "input"/"sr"/"h"/"c", output: "output"/"hn"/"cn").
//!
//! ## VAD_CONFIDENCE
//! The v4.0 model returns confidence in roughly 0.0–0.2 range for speech
//! (verified against official test.wav). Python defaults (0.7) were calibrated
//! for an older model version. Set to 0.1 — above noise floor, below speech peaks.
//!
//! ## VAD_MIN_VOLUME
//! Python uses EBU R128 loudness normalised to 0–1 (pyloudnorm).
//! We approximate with dBFS: (dBFS + 60) / 60, clamped to 0–1.
//! On this scale, normal speech (~-20 to -10 dBFS) maps to ~0.67–0.83.
//! Set to 0.3 — catches speech well above silence without false triggers.
//!
//! ## VAD_START_SECS / VAD_STOP_SECS
//! Increased from Python's 0.2s to 0.4s to reduce rapid oscillation
//! given the lower confidence values from this model version.

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

/// Minimum model confidence to consider audio as speech.
/// Calibrated for Silero v4.0 ONNX (confidence range ~0.0–0.2 for speech).
pub const VAD_CONFIDENCE: f32 = 0.1;

/// How long speech must be detected before transitioning STARTING → SPEAKING.
pub const VAD_START_SECS: f32 = 0.4;

/// How long silence must persist before transitioning STOPPING → QUIET.
pub const VAD_STOP_SECS: f32 = 0.4;

/// Minimum volume (dBFS-normalised 0–1) below which audio is treated as silence.
/// Uses dBFS scale: (dBFS + 60) / 60. Normal speech → ~0.5–0.8 on this scale.
pub const VAD_MIN_VOLUME: f32 = 0.3;

// ---------------------------------------------------------------------------
// VadParams
// ---------------------------------------------------------------------------

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

    /// Minimum volume (dBFS-normalised 0–1) below which audio is treated as silence.
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