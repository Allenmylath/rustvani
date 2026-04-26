//! Audio processing utilities.
//!
//! - `resamplers` — streaming sample-rate conversion (pure Rust, via `rubato`)
//! - `noisefilter` — RNNoise-based noise suppression (pure Rust, via `nnnoiseless`)

pub mod noisefilter;
pub mod resamplers;
