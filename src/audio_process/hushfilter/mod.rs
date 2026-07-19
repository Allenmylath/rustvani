//! DeepFilterNet3-style noise suppression via the `hush-vani` crate.
//!
//! `hush-vani` exposes a **batch** API: [`Hush::enhance`] processes a whole
//! signal and its GRUs start from zero on every call (no streaming API is
//! exposed by the crate). To use it on the live STT stream — where audio
//! arrives in ~20 ms chunks and is forwarded to Sarvam incrementally — this
//! wrapper runs a **sliding window with lookback context**:
//!
//! * All input for the current utterance is buffered at 16 kHz (`hush-vani`'s
//!   native rate). On each [`filter`](HushVaniFilter::filter) call we re-run
//!   `enhance` over a window that includes up to [`CONTEXT_SAMPLES`] of *prior*
//!   audio plus the newly-arrived audio.
//! * The context region re-primes the GRUs to roughly the state they'd hold in
//!   a true streaming decode, so the new region is enhanced with real context.
//!   The context region of the output is discarded; only the new samples are
//!   emitted. Re-computing the context each call is cheap (`hush-vani` runs at
//!   ~100× real-time).
//!
//! `enhance` returns `len / 160 * 160` samples that **lag the input by
//! `LATENCY_SAMPLES` (160)**. So for an absolute input index `m` inside a window
//! starting at absolute index `win_start`:
//!
//! ```text
//! clean(input[m]) == out[m - win_start + LATENCY]
//! ```
//!
//! We therefore hold back the last frame of each call (its lagged output isn't
//! available yet) and drain it in [`flush`](HushVaniFilter::flush) via zero
//! padding. Across an utterance total output length ≈ total input length.
//!
//! Sample rates other than 16 kHz are resampled transparently in/out using
//! [`StreamResampler`](super::resamplers::StreamResampler). Samples are
//! normalised to `f32` in `[-1, 1]` for `hush-vani` and denormalised back to
//! i16 on the way out.

use hush_vani::Hush;
use log;

use super::resamplers::{ResamplerQuality, StreamResampler};

/// `hush-vani`'s required sample rate.
const HUSH_SAMPLE_RATE: u32 = 16_000;

/// Samples per 10 ms frame at 16 kHz (`hush-vani`'s frame unit).
const FRAME: usize = 160;

/// Algorithmic latency of `enhance` (documented as `LATENCY_SAMPLES` = 160).
const LATENCY: usize = 160;

/// Lookback context re-fed each call to re-prime the GRUs (200 ms). Must be a
/// multiple of [`FRAME`].
const CONTEXT_SAMPLES: usize = 20 * FRAME;

// ---------------------------------------------------------------------------
// HushVaniFilter
// ---------------------------------------------------------------------------

/// Streaming adapter around the batch `hush-vani` denoiser.
///
/// Accepts PCM i16 at any sample rate (resampled to 16 kHz internally) and
/// exposes the same streaming shape as
/// [`RNNoiseFilter`](super::noisefilter::RNNoiseFilter):
/// `filter` / `flush` / `reset`.
pub struct HushVaniFilter {
    hush: Hush,
    sample_rate: u32,
    /// Utterance audio at 16 kHz, normalised to `[-1, 1]`. Element 0 is at
    /// absolute index [`Self::base`]; the front is trimmed as it is consumed.
    input16k: Vec<f32>,
    /// Absolute index (in 16 kHz samples since the last reset) of `input16k[0]`.
    base: usize,
    /// Absolute index up to which output has already been emitted.
    emitted: usize,
    resampler_in: Option<StreamResampler>,
    resampler_out: Option<StreamResampler>,
}

impl HushVaniFilter {
    /// Create a filter for the given source sample rate.
    ///
    /// Loads the embedded `hush-vani` weights (no files/network). Returns an
    /// error only if the model fails to initialise.
    pub fn new(sample_rate: u32) -> Result<Self, hush_vani::Error> {
        let hush = Hush::new()?;

        let (resampler_in, resampler_out) = if sample_rate != HUSH_SAMPLE_RATE {
            log::info!(
                "HushVaniFilter: enabling resampling {} ↔ {} Hz",
                sample_rate,
                HUSH_SAMPLE_RATE
            );
            (
                Some(StreamResampler::new(
                    sample_rate,
                    HUSH_SAMPLE_RATE,
                    ResamplerQuality::Quick,
                )),
                Some(StreamResampler::new(
                    HUSH_SAMPLE_RATE,
                    sample_rate,
                    ResamplerQuality::Quick,
                )),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            hush,
            sample_rate,
            input16k: Vec::with_capacity(CONTEXT_SAMPLES + FRAME * 8),
            base: 0,
            emitted: 0,
            resampler_in,
            resampler_out,
        })
    }

    /// Convert incoming i16 → normalised f32 and resample to 16 kHz.
    fn ingest(&mut self, audio: &[i16]) {
        let norm: Vec<f32> = audio.iter().map(|&s| s as f32 / 32_768.0).collect();
        let at_16k = match self.resampler_in {
            Some(ref mut r) => r.process(&norm),
            None => norm,
        };
        self.input16k.extend_from_slice(&at_16k);
    }

    /// Resample a 16 kHz normalised block back to the source rate and clamp to
    /// i16. Set `flush_out` to also drain the output resampler tail.
    fn egress(&mut self, block16k: &[f32], flush_out: bool) -> Vec<i16> {
        let at_src = match self.resampler_out {
            Some(ref mut r) => {
                let mut out = r.process(block16k);
                if flush_out {
                    out.extend_from_slice(&r.flush());
                }
                out
            }
            None => block16k.to_vec(),
        };
        f32_to_i16(&at_src)
    }

    /// Denoise a chunk of PCM i16 audio.
    ///
    /// Returns the newly-finalised denoised samples (empty while the wrapper is
    /// still accumulating a frame of lookahead). Output at the source rate.
    pub fn filter(&mut self, audio: &[i16]) -> Vec<i16> {
        if audio.is_empty() {
            return Vec::new();
        }
        self.ingest(audio);

        let total_abs = self.base + self.input16k.len();
        let total_frames = total_abs / FRAME;
        // Hold back the final frame: its lagged output is not available yet.
        let target = total_frames.saturating_sub(1) * FRAME;
        if target <= self.emitted {
            return Vec::new();
        }

        // Window: up to CONTEXT_SAMPLES before `emitted`, through the last whole
        // frame. Frame-aligned so `enhance` returns the full window length.
        let win_start = (self.emitted.saturating_sub(CONTEXT_SAMPLES)) / FRAME * FRAME;
        let frame_end = total_frames * FRAME;
        let window: Vec<f32> = self.input16k[(win_start - self.base)..(frame_end - self.base)].to_vec();

        let out = match self.hush.enhance(&window) {
            Ok(o) => o,
            Err(e) => {
                // Passthrough this block on failure rather than dropping audio.
                log::warn!("HushVaniFilter: enhance failed ({e}); passing through");
                let block: Vec<f32> =
                    self.input16k[(self.emitted - self.base)..(target - self.base)].to_vec();
                self.emitted = target;
                self.trim(target);
                return self.egress(&block, false);
            }
        };

        // clean(input[m]) == out[m - win_start + LATENCY]
        let lo = self.emitted - win_start + LATENCY;
        let hi = (target - win_start + LATENCY).min(out.len());
        let block = if lo < hi { out[lo..hi].to_vec() } else { Vec::new() };

        self.emitted = target;
        self.trim(target);
        self.egress(&block, false)
    }

    /// Drain the buffered/lagging tail at end of utterance.
    ///
    /// Emits denoised output for every remaining input sample by zero-padding
    /// the window past the [`LATENCY`] lag, then clears utterance state.
    pub fn flush(&mut self) -> Vec<i16> {
        // Pull any tail out of the input resampler first.
        if let Some(ref mut r) = self.resampler_in {
            let tail = r.flush();
            self.input16k.extend_from_slice(&tail);
        }

        let total_abs = self.base + self.input16k.len();
        if total_abs <= self.emitted {
            // Nothing new to enhance; still drain the output resampler.
            self.reset_buffers();
            let empty: Vec<f32> = Vec::new();
            return self.egress(&empty, true);
        }

        let win_start = (self.emitted.saturating_sub(CONTEXT_SAMPLES)) / FRAME * FRAME;
        // Window = remaining audio (context + unemitted), zero-padded to a frame
        // boundary, plus one extra frame to flush the LATENCY lag.
        let mut window: Vec<f32> = self.input16k[(win_start - self.base)..].to_vec();
        while window.len() % FRAME != 0 {
            window.push(0.0);
        }
        window.extend(std::iter::repeat(0.0).take(FRAME));

        let block = match self.hush.enhance(&window) {
            Ok(out) => {
                let lo = self.emitted - win_start + LATENCY;
                let hi = (total_abs - win_start + LATENCY).min(out.len());
                if lo < hi {
                    out[lo..hi].to_vec()
                } else {
                    Vec::new()
                }
            }
            Err(e) => {
                log::warn!("HushVaniFilter: flush enhance failed ({e}); passing through");
                self.input16k[(self.emitted - self.base)..].to_vec()
            }
        };

        self.reset_buffers();
        self.egress(&block, true)
    }

    /// Discard buffered audio and reset state between utterances.
    pub fn reset(&mut self) {
        self.reset_buffers();
        if let Some(ref mut r) = self.resampler_in {
            r.reset();
        }
        if let Some(ref mut r) = self.resampler_out {
            r.reset();
        }
        // `Hush` holds no cross-call state (GRUs reset per `enhance`).
    }

    fn reset_buffers(&mut self) {
        self.input16k.clear();
        self.base = 0;
        self.emitted = 0;
    }

    /// Drop the front of `input16k` we will never need again (anything before
    /// the next call's lookback window), advancing `base`.
    fn trim(&mut self, emitted: usize) {
        let keep_from = emitted.saturating_sub(CONTEXT_SAMPLES) / FRAME * FRAME;
        if keep_from > self.base {
            let drop = keep_from - self.base;
            self.input16k.drain(..drop);
            self.base = keep_from;
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

impl super::StreamingDenoiser for HushVaniFilter {
    fn filter(&mut self, audio: &[i16]) -> Vec<i16> {
        HushVaniFilter::filter(self, audio)
    }
    fn flush(&mut self) -> Vec<i16> {
        HushVaniFilter::flush(self)
    }
    fn reset(&mut self) {
        HushVaniFilter::reset(self)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Denormalise `[-1, 1]` f32 back to clamped i16.
fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&s| (s * 32_768.0).clamp(-32_768.0, 32_767.0) as i16)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// At 16 kHz (no resampling) the wrapper is exactly length-preserving over
    /// an utterance: everything fed in comes back out across filter + flush.
    #[test]
    fn length_preserving_at_16k() {
        let mut nf = HushVaniFilter::new(16_000).expect("init");
        let total = 16_000; // 1 s
        let mut produced = 0usize;
        // Feed in 320-sample (20 ms) chunks, like the live stream.
        let chunk: Vec<i16> = (0..320).map(|i| ((i % 100) as i16 - 50) * 100).collect();
        let n_chunks = total / chunk.len();
        for _ in 0..n_chunks {
            produced += nf.filter(&chunk).len();
        }
        produced += nf.flush().len();
        assert_eq!(produced, n_chunks * chunk.len(), "output length must match input");
    }

    /// Streaming in tiny sub-frame chunks must not panic and stays contiguous.
    #[test]
    fn subframe_chunks_and_reset() {
        let mut nf = HushVaniFilter::new(16_000).expect("init");
        let chunk: Vec<i16> = vec![1_000; 37]; // not a frame multiple
        let mut produced = 0usize;
        for _ in 0..200 {
            produced += nf.filter(&chunk).len();
        }
        produced += nf.flush().len();
        assert_eq!(produced, 200 * 37);

        // After reset the next utterance is independent.
        nf.reset();
        let tail = nf.flush();
        assert!(tail.is_empty());
    }
}
