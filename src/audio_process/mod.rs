//! Audio processing utilities.
//!
//! - `resamplers` — streaming sample-rate conversion (pure Rust, via `rubato`)
//! - `noisefilter` — RNNoise-based noise suppression (pure Rust, via `nnnoiseless`)
//! - `hushfilter` — DeepFilterNet3-style suppression (pure Rust, via `hush-vani`)
//! - `agc` — high-pass filter, automatic gain control, and soft limiter

pub mod agc;
pub mod hushfilter;
pub mod noisefilter;
pub mod resamplers;

/// A streaming noise-suppression filter over PCM i16 audio.
///
/// Both [`RNNoiseFilter`](noisefilter::RNNoiseFilter) and
/// [`HushVaniFilter`](hushfilter::HushVaniFilter) implement this, so the STT
/// path can select a backend at runtime behind a single `Box<dyn>`.
///
/// Contract: over a full utterance the total output length is approximately the
/// total input length. Any individual [`filter`](Self::filter) call may return
/// fewer (or zero) samples because of internal frame buffering / algorithmic
/// latency. Call [`flush`](Self::flush) at end-of-utterance to drain the tail,
/// and [`reset`](Self::reset) to clear state before the next utterance.
pub trait StreamingDenoiser: Send {
    /// Denoise a chunk of PCM i16 audio (may return fewer samples than given).
    fn filter(&mut self, audio: &[i16]) -> Vec<i16>;
    /// Flush any buffered/lagging tail at end of utterance.
    fn flush(&mut self) -> Vec<i16>;
    /// Discard buffered audio and reset internal state between utterances.
    fn reset(&mut self);
}

impl StreamingDenoiser for noisefilter::RNNoiseFilter {
    fn filter(&mut self, audio: &[i16]) -> Vec<i16> {
        RNNoiseFilter::filter(self, audio)
    }
    fn flush(&mut self) -> Vec<i16> {
        RNNoiseFilter::flush(self)
    }
    fn reset(&mut self) {
        RNNoiseFilter::reset(self)
    }
}

use noisefilter::RNNoiseFilter;
