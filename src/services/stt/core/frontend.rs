//! The STT input audio chain, as one reusable struct.
//!
//! ```text
//! transport PCM ─► resample ─► high-pass ─► denoise ─► AGC + limiter ─► provider
//!                  (if rate    (DC/rumble)  (RNNoise |  (level + clip
//!                   differs)                 hush-vani)  protection)
//! ```
//!
//! This composes existing primitives rather than reimplementing them:
//! [`StreamResampler`](crate::audio_process::resamplers::StreamResampler),
//! [`AudioEnhancer`](crate::audio_process::agc::AudioEnhancer), and the
//! [`StreamingDenoiser`](crate::audio_process::StreamingDenoiser) backends.
//!
//! **Latency contract**, inherited from `StreamingDenoiser`: [`process`] may
//! return fewer samples than it was given — or none at all — because the
//! denoiser and the resampler both buffer internally. Over a full utterance
//! output length ≈ input length × rate ratio. Call [`flush`] at end of turn to
//! drain the tail and [`reset`] before the next one.
//!
//! [`process`]: AudioFrontend::process
//! [`flush`]: AudioFrontend::flush
//! [`reset`]: AudioFrontend::reset

use crate::audio_process::agc::{AgcConfig, AudioEnhancer};
use crate::audio_process::hushfilter::HushVaniFilter;
use crate::audio_process::noisefilter::RNNoiseFilter;
use crate::audio_process::resamplers::{ResamplerQuality, StreamResampler};
use crate::audio_process::StreamingDenoiser;

/// Noise-suppression backend applied on the STT input path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NoiseBackend {
    /// RNNoise (`nnnoiseless`) — true streaming filter. The default.
    #[default]
    Rnnoise,
    /// DeepFilterNet3-style (`hush-vani`) — stronger suppression, run via a
    /// sliding-window wrapper over its batch API. Opt-in.
    HushVani,
}

/// How to build an [`AudioFrontend`].
#[derive(Debug, Clone)]
pub struct FrontendConfig {
    /// Rate the provider wants. Incoming audio at any other rate is resampled.
    pub target_sample_rate: u32,
    /// Run noise suppression.
    pub noise_reduction: bool,
    /// Which suppressor, when `noise_reduction` is on.
    pub noise_backend: NoiseBackend,
    /// Run the high-pass / AGC / limiter chain.
    pub agc: bool,
    /// AGC tuning. Ignored when `agc` is false.
    pub agc_config: AgcConfig,
}

impl Default for FrontendConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: 16_000,
            noise_reduction: true,
            noise_backend: NoiseBackend::default(),
            agc: true,
            agc_config: AgcConfig::default(),
        }
    }
}

/// The assembled chain. Not `Sync` — the driver keeps it behind a mutex and
/// shares that with the receive task (which resets it between utterances).
pub struct AudioFrontend {
    target_sample_rate: u32,
    /// Rate of the audio currently being fed in; drives resampler construction.
    source_sample_rate: Option<u32>,
    resampler: Option<StreamResampler>,
    denoiser: Option<Box<dyn StreamingDenoiser>>,
    enhancer: Option<AudioEnhancer>,
}

impl AudioFrontend {
    pub fn new(config: &FrontendConfig) -> Self {
        let rate = config.target_sample_rate;

        let denoiser: Option<Box<dyn StreamingDenoiser>> = if config.noise_reduction {
            Some(match config.noise_backend {
                NoiseBackend::Rnnoise => {
                    log::info!("AudioFrontend: noise reduction — RNNoise (sample_rate={rate})");
                    Box::new(RNNoiseFilter::new(rate))
                }
                NoiseBackend::HushVani => match HushVaniFilter::new(rate) {
                    Ok(f) => {
                        log::info!(
                            "AudioFrontend: noise reduction — hush-vani (sample_rate={rate})"
                        );
                        Box::new(f) as Box<dyn StreamingDenoiser>
                    }
                    Err(e) => {
                        log::error!(
                            "AudioFrontend: hush-vani init failed ({e}); falling back to RNNoise"
                        );
                        Box::new(RNNoiseFilter::new(rate))
                    }
                },
            })
        } else {
            None
        };

        let enhancer = if config.agc {
            log::info!("AudioFrontend: speech enhancement — HPF + AGC + limiter (sample_rate={rate})");
            Some(AudioEnhancer::with_config(rate, config.agc_config.clone()))
        } else {
            None
        };

        Self {
            target_sample_rate: rate,
            source_sample_rate: None,
            resampler: None,
            denoiser,
            enhancer,
        }
    }

    /// Rate the chain emits at — what the provider will receive.
    pub fn target_sample_rate(&self) -> u32 {
        self.target_sample_rate
    }

    /// Run one chunk of PCM through the chain.
    ///
    /// `source_rate` is the rate of `pcm`, taken from the incoming
    /// `AudioRawData`. When it differs from the target rate a resampler is
    /// built on first use and reused; a mid-stream rate change rebuilds it.
    ///
    /// May return fewer samples than given (or none) while the denoiser and
    /// resampler buffer.
    pub fn process(&mut self, pcm: &[i16], source_rate: u32) -> Vec<i16> {
        let pcm = self.resample(pcm, source_rate);
        if pcm.is_empty() {
            return Vec::new();
        }

        // 1. DC removal + high-pass, before the denoiser.
        let mut out = match &mut self.enhancer {
            Some(enh) => enh.pre_filter(&pcm),
            None => pcm,
        };

        // 2. Noise suppression (may buffer — empty output is normal here).
        if let Some(d) = &mut self.denoiser {
            out = d.filter(&out);
        }

        // 3. AGC + soft limiter, after the denoiser so silence isn't amplified
        //    before it is suppressed.
        if !out.is_empty() {
            if let Some(enh) = &mut self.enhancer {
                out = enh.post_filter(&out);
            }
        }

        out
    }

    /// Drain the tail at end of utterance: resampler tail, then denoiser tail,
    /// then level it like any other output.
    ///
    /// When resampling is active the tail can overshoot the ideal length by up
    /// to one output chunk (60 ms at 16 kHz) of trailing near-silence, because
    /// [`StreamResampler::flush`] zero-pads its partial buffer to a chunk
    /// boundary. That is harmless here — the tail is trailing silence sent just
    /// before the finalize signal, and providers that report their own audio
    /// duration bill from that, not from what we sent.
    pub fn flush(&mut self) -> Vec<i16> {
        let mut tail: Vec<i16> = Vec::new();

        if let Some(r) = &mut self.resampler {
            let resampled = f32_to_i16(&r.flush());
            if !resampled.is_empty() {
                let pre = match &mut self.enhancer {
                    Some(enh) => enh.pre_filter(&resampled),
                    None => resampled,
                };
                match &mut self.denoiser {
                    Some(d) => tail.extend(d.filter(&pre)),
                    None => tail.extend(pre),
                }
            }
        }

        if let Some(d) = &mut self.denoiser {
            tail.extend(d.flush());
        }

        if !tail.is_empty() {
            if let Some(enh) = &mut self.enhancer {
                tail = enh.post_filter(&tail);
            }
        }

        tail
    }

    /// Clear buffered audio between utterances.
    ///
    /// The enhancer's *adapted AGC gain* is deliberately retained — the same
    /// speaker is likely to continue at the same level; only its filter state
    /// is cleared (see [`AudioEnhancer::reset`]).
    pub fn reset(&mut self) {
        if let Some(r) = &mut self.resampler {
            r.reset();
        }
        if let Some(d) = &mut self.denoiser {
            d.reset();
        }
        if let Some(enh) = &mut self.enhancer {
            enh.reset();
        }
    }

    /// Convert to the target rate, building (or rebuilding) the resampler as
    /// the source rate demands. A matching rate is a straight copy.
    fn resample(&mut self, pcm: &[i16], source_rate: u32) -> Vec<i16> {
        if source_rate == self.target_sample_rate {
            // A previously-built resampler is now moot; drop it so a later
            // return to the old rate starts clean.
            if self.resampler.is_some() {
                log::info!(
                    "AudioFrontend: input rate now matches target {}Hz — resampler dropped",
                    self.target_sample_rate
                );
                self.resampler = None;
            }
            self.source_sample_rate = Some(source_rate);
            return pcm.to_vec();
        }

        if self.source_sample_rate != Some(source_rate) || self.resampler.is_none() {
            log::info!(
                "AudioFrontend: resampling input {}Hz → {}Hz",
                source_rate, self.target_sample_rate
            );
            self.resampler = Some(StreamResampler::new(
                source_rate,
                self.target_sample_rate,
                ResamplerQuality::Quick,
            ));
            self.source_sample_rate = Some(source_rate);
        }

        let input: Vec<f32> = pcm.iter().map(|&s| s as f32).collect();
        let resampled = self
            .resampler
            .as_mut()
            .expect("resampler built above")
            .process(&input);
        f32_to_i16(&resampled)
    }
}

/// i16-range floats → clamped i16. (The DSP chain works in i16-range floats,
/// not normalised ones — see the note in `audio_process::noisefilter`.)
fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&s| s.clamp(i16::MIN as f32, i16::MAX as f32) as i16)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 300 Hz tone at `rate`, `ms` long — speech-band, so the high-pass
    /// leaves it alone.
    fn tone(rate: u32, ms: u32) -> Vec<i16> {
        let n = (rate as u64 * ms as u64 / 1000) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / rate as f32;
                (8000.0 * (2.0 * std::f32::consts::PI * 300.0 * t).sin()) as i16
            })
            .collect()
    }

    fn plain(target: u32) -> FrontendConfig {
        FrontendConfig {
            target_sample_rate: target,
            noise_reduction: false,
            noise_backend: NoiseBackend::Rnnoise,
            agc: false,
            agc_config: AgcConfig::default(),
        }
    }

    #[test]
    fn passthrough_when_everything_disabled() {
        let mut fe = AudioFrontend::new(&plain(16_000));
        let input = tone(16_000, 20);
        assert_eq!(fe.process(&input, 16_000), input);
        assert!(fe.flush().is_empty());
    }

    /// The resampler emits in 480-sample input chunks and `flush` zero-pads a
    /// partial buffer up to that boundary, so a stream can end up to one output
    /// chunk long. Assert the real contract: never short, never more than one
    /// chunk of padding over.
    fn assert_resampled_len(out: usize, expected: usize, target_rate: u32, from_rate: u32) {
        let max_pad = (480.0 * target_rate as f64 / from_rate as f64).ceil() as usize;
        assert!(
            out + expected / 20 >= expected,
            "resampling must not lose audio: expected ~{expected}, got {out}"
        );
        assert!(
            out <= expected + max_pad,
            "flush padding must stay under one output chunk ({max_pad}): \
             expected ~{expected}, got {out}"
        );
    }

    #[test]
    fn upsamples_8k_to_16k_at_roughly_double_length() {
        let mut fe = AudioFrontend::new(&plain(16_000));

        let mut out = 0usize;
        // 10 × 20ms chunks at 8kHz = 1600 samples in → ~3200 out at 16kHz.
        for _ in 0..10 {
            out += fe.process(&tone(8_000, 20), 8_000).len();
        }
        out += fe.flush().len();

        assert_resampled_len(out, 3_200, 16_000, 8_000);
    }

    #[test]
    fn downsamples_48k_to_16k_at_roughly_a_third() {
        let mut fe = AudioFrontend::new(&plain(16_000));

        let mut out = 0usize;
        for _ in 0..10 {
            out += fe.process(&tone(48_000, 20), 48_000).len();
        }
        out += fe.flush().len();

        assert_resampled_len(out, 3_200, 16_000, 48_000);
    }

    #[test]
    fn matching_rate_is_not_resampled() {
        let mut fe = AudioFrontend::new(&plain(16_000));
        fe.process(&tone(16_000, 20), 16_000);
        assert!(fe.resampler.is_none(), "no resampler should be built for a matching rate");
    }

    #[test]
    fn rate_change_mid_stream_rebuilds_the_resampler() {
        let mut fe = AudioFrontend::new(&plain(16_000));

        fe.process(&tone(8_000, 20), 8_000);
        assert_eq!(fe.source_sample_rate, Some(8_000));
        assert!(fe.resampler.is_some());

        fe.process(&tone(48_000, 20), 48_000);
        assert_eq!(fe.source_sample_rate, Some(48_000));
        assert!(fe.resampler.is_some());

        // Falling back to the target rate drops the resampler entirely.
        fe.process(&tone(16_000, 20), 16_000);
        assert!(fe.resampler.is_none());
    }

    #[test]
    fn denoise_chain_preserves_length_over_an_utterance() {
        let mut fe = AudioFrontend::new(&FrontendConfig {
            target_sample_rate: 16_000,
            noise_reduction: true,
            noise_backend: NoiseBackend::Rnnoise,
            agc: true,
            agc_config: AgcConfig::default(),
        });

        let chunk = tone(16_000, 20); // 320 samples
        let mut sent = 0usize;
        let mut out = 0usize;
        for _ in 0..25 {
            sent += chunk.len();
            out += fe.process(&chunk, 16_000).len();
        }
        out += fe.flush().len();

        let ratio = out as f64 / sent as f64;
        assert!(
            (0.9..=1.1).contains(&ratio),
            "denoise chain must roughly preserve length: sent {sent}, got {out} (ratio {ratio:.3})"
        );
    }

    #[test]
    fn reset_clears_buffers_without_panicking() {
        let mut fe = AudioFrontend::new(&FrontendConfig {
            target_sample_rate: 16_000,
            noise_reduction: true,
            ..Default::default()
        });
        fe.process(&tone(8_000, 20), 8_000);
        fe.reset();
        // Still usable afterwards.
        fe.process(&tone(8_000, 20), 8_000);
        fe.flush();
    }
}
