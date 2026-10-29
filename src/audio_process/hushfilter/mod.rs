//! DeepFilterNet3-style noise suppression via the `hush-vani` crate.
//!
//! [`hush_vani::HushStream`] runs the model one 10 ms frame at a time with its GRUs and its
//! STFT normalisation carried from frame to frame, matching a batch `Hush::enhance` over the same
//! audio. (Until 0.4.0-dev.19 this wrapper re-ran the batch `enhance` over a 200 ms lookback
//! window per chunk. Each call restarted the spectral normalisation, whose time constant is
//! 1 s, so the model never saw settled features and suppressed speech as noise: transcripts
//! went missing after the first few utterances.)
//!
//! The stream's output lags its input by `Hush::LATENCY_SAMPLES` (160 samples, 10 ms); this
//! wrapper removes the lag, so over an utterance (`filter` calls, then `flush`) the output is the
//! input's length and lines up with it. `flush` drains the lag with zero padding.
//!
//! [`reset`](HushVaniFilter::reset) between utterances restarts that bookkeeping but keeps the
//! model's state: the same speaker in the same room continues, and a cold start would cost the
//! next utterance its first second while the normalisation settles again.
//!
//! Sample rates other than 16 kHz are resampled transparently in/out using
//! [`StreamResampler`](super::resamplers::StreamResampler). Samples are normalised to `f32` in
//! `[-1, 1]` for `hush-vani` and denormalised back to i16 on the way out.

use std::sync::Arc;

use hush_vani::{Hush, HushStream};
use log;

use super::resamplers::{ResamplerQuality, StreamResampler};

/// `hush-vani`'s required sample rate.
const HUSH_SAMPLE_RATE: u32 = 16_000;

/// Samples per 10 ms frame at 16 kHz (`hush-vani`'s frame unit).
const FRAME: usize = Hush::FRAME_SAMPLES;

/// The stream's output lag, in samples.
const LATENCY: usize = Hush::LATENCY_SAMPLES;

// ---------------------------------------------------------------------------
// HushVaniFilter
// ---------------------------------------------------------------------------

/// Streaming `hush-vani` denoiser.
///
/// Accepts PCM i16 at any sample rate (resampled to 16 kHz internally) and
/// exposes the same streaming shape as
/// [`RNNoiseFilter`](super::noisefilter::RNNoiseFilter):
/// `filter` / `flush` / `reset`.
pub struct HushVaniFilter {
    stream: HushStream,
    sample_rate: u32,
    /// 16 kHz samples fed in this utterance, and given out (aligned, lag removed).
    fed: usize,
    emitted: usize,
    /// Output samples still to drop at the start of the utterance: the lag.
    skip: usize,
    resampler_in: Option<StreamResampler>,
    resampler_out: Option<StreamResampler>,
}

impl HushVaniFilter {
    /// Create a filter for the given source sample rate.
    ///
    /// Loads the embedded `hush-vani` weights (no files/network). Returns an
    /// error only if the model fails to initialise.
    pub fn new(sample_rate: u32) -> Result<Self, hush_vani::Error> {
        let stream = HushStream::new(Arc::new(Hush::new()?));

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
            stream,
            sample_rate,
            fed: 0,
            emitted: 0,
            skip: LATENCY,
            resampler_in,
            resampler_out,
        })
    }

    /// Convert incoming i16 → normalised f32 at 16 kHz.
    fn ingest(&mut self, audio: &[i16]) -> Vec<f32> {
        let norm: Vec<f32> = audio.iter().map(|&s| s as f32 / 32_768.0).collect();
        match self.resampler_in {
            Some(ref mut r) => r.process(&norm),
            None => norm,
        }
    }

    /// Run 16 kHz audio through the model; returns the aligned output it completes.
    fn enhance(&mut self, at_16k: &[f32]) -> Vec<f32> {
        self.fed += at_16k.len();
        let mut out = self.stream.process(at_16k);
        let drop = self.skip.min(out.len());
        out.drain(..drop);
        self.skip -= drop;
        self.emitted += out.len();
        out
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
    /// Returns the newly-finalised denoised samples (fewer than given while the
    /// model's 10 ms framing and lag are being filled). Output at the source rate.
    pub fn filter(&mut self, audio: &[i16]) -> Vec<i16> {
        if audio.is_empty() {
            return Vec::new();
        }
        let at_16k = self.ingest(audio);
        let block = self.enhance(&at_16k);
        self.egress(&block, false)
    }

    /// Drain the buffered/lagging tail at end of utterance: every remaining
    /// input sample comes out (the model is fed zeros to push it through).
    pub fn flush(&mut self) -> Vec<i16> {
        let tail = match self.resampler_in {
            Some(ref mut r) => r.flush(),
            None => Vec::new(),
        };
        let mut block = self.enhance(&tail);
        let owed = self.fed.saturating_sub(self.emitted);
        if owed > 0 {
            // Zeros through the partial frame and the lag, then keep only what is owed.
            let mut out = self.stream.process(&vec![0.0; owed + FRAME + LATENCY]);
            let drop = self.skip.min(out.len());
            out.drain(..drop);
            out.truncate(owed);
            block.extend_from_slice(&out);
        }
        let out = self.egress(&block, true);
        self.start_utterance();
        out
    }

    /// Discard what is buffered for this utterance and start the next one (the
    /// model's state is kept: see the module docs).
    pub fn reset(&mut self) {
        // Drop a partial frame by completing it with zeros, unread.
        let partial = self.fed % FRAME;
        if partial != 0 {
            let _ = self.stream.process(&vec![0.0; FRAME - partial]);
        }
        if let Some(ref mut r) = self.resampler_in {
            r.reset();
        }
        if let Some(ref mut r) = self.resampler_out {
            r.reset();
        }
        self.start_utterance();
    }

    fn start_utterance(&mut self) {
        self.fed = 0;
        self.emitted = 0;
        self.skip = LATENCY;
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

    /// Voiced-ish speech (harmonics, gated like syllables) under broadband noise, at 16 kHz.
    fn noisy_speech(n: usize, seed: u32) -> (Vec<i16>, Vec<f32>) {
        let mut s = seed;
        let mut clean = Vec::with_capacity(n);
        let noisy = (0..n)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                let f0 = 140.0 + 30.0 * (2.0 * std::f32::consts::PI * 3.0 * t).sin();
                let gate = if (t * 4.0) as u32 % 3 == 2 { 0.0 } else { 1.0 };
                let voice: f32 = (1..8).map(|k| (2.0 * std::f32::consts::PI * f0 * k as f32 * t).sin() / k as f32).sum();
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
                clean.push(0.2 * gate * voice);
                ((0.2 * gate * voice + 0.05 * noise) * 32_767.0) as i16
            })
            .collect();
        (noisy, clean)
    }

    fn rms(x: impl Iterator<Item = f32>) -> f32 {
        let (s, n) = x.fold((0.0f32, 0usize), |(s, n), v| (s + v * v, n + 1));
        (s / n.max(1) as f32).sqrt()
    }

    /// Speech keeps the level the model gives it, utterance after utterance (the windowed
    /// wrapper lost it after two or three), and the output is the input's length.
    #[test]
    fn speech_survives_many_utterances() {
        let hush = Hush::new().unwrap();
        let mut nf = HushVaniFilter::new(16_000).expect("init");
        for utt in 0..8 {
            let (noisy, clean) = noisy_speech(16_000 * 2, 7 + utt);
            let mut out = Vec::new();
            for c in noisy.chunks(320) {
                out.extend(nf.filter(c));
            }
            out.extend(nf.flush());
            nf.reset();
            assert_eq!(out.len(), noisy.len());
            // Over the voiced parts.
            let voiced = |x: &[f32]| rms(x.iter().zip(&clean).filter(|(_, c)| c.abs() > 0.0).map(|(v, _)| *v));
            let got: Vec<f32> = out.iter().map(|&s| s as f32 / 32_768.0).collect();
            let ratio = voiced(&got) / voiced(&clean);
            // The batch model over this utterance alone, as the bar: the stream keeps as much.
            let batch = hush.enhance(&noisy.iter().map(|&v| v as f32 / 32_768.0).collect::<Vec<_>>()).unwrap();
            let batch_ratio = voiced(&batch[LATENCY..]) / voiced(&clean[..batch.len() - LATENCY]);
            assert!(
                ratio > batch_ratio * 0.9,
                "utterance {utt}: speech at {ratio:.2} of its level, the batch model keeps {batch_ratio:.2}"
            );
        }
    }
}
