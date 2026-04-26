//! RNNoise noise suppression filter (pure Rust via `nnnoiseless`).
//!
//! Buffers incoming audio to 480-sample frames at 48 kHz, runs each frame
//! through the RNNoise denoiser, and returns cleaned audio at the original
//! sample rate.  If the source rate isn't 48 kHz, transparent resampling
//! is applied using [`StreamResampler`](super::resamplers::StreamResampler).
//!
//! ```ignore
//! let mut nf = RNNoiseFilter::new(16_000);
//! let clean = nf.filter(&noisy_pcm_i16);   // may return fewer samples (buffering)
//! let tail  = nf.flush();                   // zero-pads + drains at end of utterance
//! nf.reset();                               // clear state between utterances
//! ```
//!
//! **Important**: `nnnoiseless` operates in i16-range floats (−32 768 … 32 767),
//! *not* normalised (−1 … 1).  We convert i16 → f32 without scaling, pass
//! through the denoiser, then clamp back to i16.  Resampling is scale-
//! invariant so the same range is used throughout.

use log;
use nnnoiseless::DenoiseState;

use super::resamplers::{ResamplerQuality, StreamResampler};

/// Required sample rate for the RNNoise model.
const RNNOISE_SAMPLE_RATE: u32 = 48_000;

/// Samples per RNNoise frame (480 @ 48 kHz = 10 ms).
const FRAME_SIZE: usize = DenoiseState::FRAME_SIZE;

// ---------------------------------------------------------------------------
// RNNoiseFilter
// ---------------------------------------------------------------------------

/// Real-time noise suppression filter backed by RNNoise.
///
/// Accepts audio at any sample rate; internally resamples to 48 kHz if
/// necessary.  All public methods work with PCM i16 samples.
pub struct RNNoiseFilter {
    denoiser: Box<DenoiseState>,
    /// Buffered samples at 48 kHz, in i16-range f32.
    buffer: Vec<f32>,
    enabled: bool,
    sample_rate: u32,
    resampler_in: Option<StreamResampler>,
    resampler_out: Option<StreamResampler>,
}

impl RNNoiseFilter {
    /// Create a new noise filter for the given source sample rate.
    ///
    /// If `sample_rate` ≠ 48 000 Hz, internal resamplers are created
    /// automatically (using [`ResamplerQuality::Quick`] for lowest latency).
    pub fn new(sample_rate: u32) -> Self {
        Self::with_quality(sample_rate, ResamplerQuality::Quick)
    }

    /// Create a noise filter with explicit resampler quality.
    pub fn with_quality(sample_rate: u32, quality: ResamplerQuality) -> Self {
        let (resampler_in, resampler_out) = if sample_rate != RNNOISE_SAMPLE_RATE {
            log::info!(
                "RNNoiseFilter: enabling resampling {} ↔ {} Hz",
                sample_rate,
                RNNOISE_SAMPLE_RATE
            );
            (
                Some(StreamResampler::new(sample_rate, RNNOISE_SAMPLE_RATE, quality)),
                Some(StreamResampler::new(RNNOISE_SAMPLE_RATE, sample_rate, quality)),
            )
        } else {
            (None, None)
        };

        Self {
            denoiser: DenoiseState::new(),
            buffer: Vec::with_capacity(FRAME_SIZE * 4),
            enabled: true,
            sample_rate,
            resampler_in,
            resampler_out,
        }
    }

    /// Enable or disable noise suppression.
    ///
    /// When disabled, [`filter`](Self::filter) passes audio through unchanged.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    // -----------------------------------------------------------------------
    // Core API
    // -----------------------------------------------------------------------

    /// Denoise a chunk of PCM i16 audio.
    ///
    /// Returns denoised samples.  The output length may differ from the input
    /// length on any individual call because of internal buffering to 480-
    /// sample frames.  Over the course of an utterance, total output ≈ total
    /// input.
    ///
    /// Returns an empty vec when the buffer hasn't yet accumulated a full
    /// 480-sample frame.
    pub fn filter(&mut self, audio: &[i16]) -> Vec<i16> {
        if !self.enabled || audio.is_empty() {
            return audio.to_vec();
        }

        // i16 → f32 (keep in i16 range for nnnoiseless)
        let samples_f32: Vec<f32> = audio.iter().map(|&s| s as f32).collect();

        // Resample to 48 kHz if needed
        let at_48k = match self.resampler_in {
            Some(ref mut r) => r.process(&samples_f32),
            None => samples_f32,
        };

        // Append to frame buffer
        self.buffer.extend_from_slice(&at_48k);

        // Process every complete 480-sample frame
        let mut denoised_48k = Vec::new();
        while self.buffer.len() >= FRAME_SIZE {
            let frame: Vec<f32> = self.buffer.drain(..FRAME_SIZE).collect();
            let mut out = vec![0.0f32; FRAME_SIZE];
            let _vad_prob = self.denoiser.process_frame(&mut out, &frame);
            denoised_48k.extend_from_slice(&out);
        }

        if denoised_48k.is_empty() {
            return Vec::new();
        }

        // Resample back to source rate
        let final_f32 = match self.resampler_out {
            Some(ref mut r) => r.process(&denoised_48k),
            None => denoised_48k,
        };

        f32_to_i16(&final_f32)
    }

    /// Flush any remaining buffered audio.
    ///
    /// Zero-pads the internal buffer to a full 480-sample frame, processes it,
    /// and returns the result.  Also flushes internal resamplers.
    ///
    /// Call this when the speaker stops (VAD end-of-speech) to avoid losing
    /// the tail of the utterance.
    pub fn flush(&mut self) -> Vec<i16> {
        // 1. Flush input resampler → may produce additional 48 kHz samples
        if let Some(ref mut r) = self.resampler_in {
            let tail = r.flush();
            self.buffer.extend_from_slice(&tail);
        }

        if self.buffer.is_empty() {
            // Still flush output resampler in case it has leftovers
            if let Some(ref mut r) = self.resampler_out {
                let tail = r.flush();
                if !tail.is_empty() {
                    return f32_to_i16(&tail);
                }
            }
            return Vec::new();
        }

        // 2. Zero-pad to frame boundary and denoise
        let mut denoised_48k = Vec::new();

        // Process any full frames first
        while self.buffer.len() >= FRAME_SIZE {
            let frame: Vec<f32> = self.buffer.drain(..FRAME_SIZE).collect();
            let mut out = vec![0.0f32; FRAME_SIZE];
            let _vad_prob = self.denoiser.process_frame(&mut out, &frame);
            denoised_48k.extend_from_slice(&out);
        }

        // Zero-pad the remainder
        if !self.buffer.is_empty() {
            self.buffer.resize(FRAME_SIZE, 0.0);
            let frame: Vec<f32> = self.buffer.drain(..).collect();
            let mut out = vec![0.0f32; FRAME_SIZE];
            let _vad_prob = self.denoiser.process_frame(&mut out, &frame);
            denoised_48k.extend_from_slice(&out);
        }

        // 3. Resample back + flush output resampler
        let final_f32 = match self.resampler_out {
            Some(ref mut r) => {
                let mut resampled = r.process(&denoised_48k);
                resampled.extend_from_slice(&r.flush());
                resampled
            }
            None => denoised_48k,
        };

        f32_to_i16(&final_f32)
    }

    /// Discard all buffered audio and reset internal state.
    ///
    /// Call this when a transcript arrives (especially if local VAD didn't
    /// fire) to ensure the next utterance starts with a clean slate.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.denoiser = DenoiseState::new();
        if let Some(ref mut r) = self.resampler_in {
            r.reset();
        }
        if let Some(ref mut r) = self.resampler_out {
            r.reset();
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Clamp f32 (in i16 range) to i16.
fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&s| s.clamp(-32_768.0, 32_767.0) as i16)
        .collect()
}
