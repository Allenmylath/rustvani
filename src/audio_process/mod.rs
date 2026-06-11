//! Audio processing utilities.
//!
//! - `resamplers` — streaming sample-rate conversion (pure Rust, via `rubato`)
//! - `noisefilter` — RNNoise-based noise suppression (pure Rust, via `nnnoiseless`)
//! - `agc` — high-pass filter, automatic gain control, and soft limiter

pub mod agc;
pub mod noisefilter;
pub mod resamplers;
