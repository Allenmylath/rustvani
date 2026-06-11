//! Speech enhancement chain: high-pass filter, AGC, and soft limiter.
//!
//! Complements [`RNNoiseFilter`](super::noisefilter::RNNoiseFilter) so STT
//! receives clean, consistently-levelled audio:
//!
//! - **High-pass filter** (`pre_filter`) — removes DC offset, rumble, and
//!   handling noise below ~90 Hz. Apply *before* the denoiser.
//! - **AGC + soft limiter** (`post_filter`) — normalises speech toward a
//!   target RMS level, then soft-clips peaks so the final i16 conversion
//!   never hard-clips. Apply *after* the denoiser (so silence/noise isn't
//!   amplified before suppression).
//!
//! ```ignore
//! let mut enh = AudioEnhancer::new(16_000);
//! let pcm  = enh.pre_filter(&raw_pcm_i16);   // HPF → feed to RNNoise
//! let out  = enh.post_filter(&denoised);      // AGC + limiter → feed to STT
//! enh.reset();                                // between utterances
//! ```
//!
//! Like the noise filter, the whole chain operates in i16-range floats
//! (−32 768 … 32 767). All stages are zero-latency (output length equals
//! input length), so there is nothing to flush.

use log;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Tuning parameters for [`AudioEnhancer`].
#[derive(Debug, Clone)]
pub struct AgcConfig {
    /// High-pass cutoff in Hz. Speech fundamentals start ~85 Hz; anything
    /// below is rumble/DC. Default: 90.
    pub highpass_hz: f32,

    /// Target speech RMS in i16 range. Default: 3277 (≈ −20 dBFS), a
    /// comfortable level for STT models.
    pub target_rms: f32,

    /// RMS below which a chunk is treated as silence and gain adaptation
    /// is held (prevents pumping the noise floor up between words).
    /// Default: 165 (≈ −46 dBFS).
    pub noise_gate_rms: f32,

    /// Maximum gain the AGC may apply (linear). Default: 31.6 (+30 dB).
    pub max_gain: f32,

    /// Minimum gain the AGC may apply (linear). Default: 0.125 (−18 dB).
    pub min_gain: f32,

    /// Attack time constant in ms — how fast gain *drops* when input gets
    /// loud. Fast, to catch sudden shouts. Default: 10.
    pub attack_ms: f32,

    /// Release time constant in ms — how fast gain *rises* for quiet
    /// speakers. Slow, to avoid breathing artefacts. Default: 400.
    pub release_ms: f32,

    /// Soft-limiter knee in i16 range — samples above this are compressed
    /// smoothly toward full scale instead of hard-clipping.
    /// Default: 22937 (≈ −3 dBFS).
    pub limiter_knee: f32,
}

impl Default for AgcConfig {
    fn default() -> Self {
        Self {
            highpass_hz:    90.0,
            target_rms:     3_277.0,
            noise_gate_rms: 165.0,
            max_gain:       31.6,
            min_gain:       0.125,
            attack_ms:      10.0,
            release_ms:     400.0,
            limiter_knee:   22_937.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Biquad high-pass (RBJ cookbook, Butterworth Q)
// ---------------------------------------------------------------------------

/// Second-order Butterworth high-pass. Removes DC by construction.
struct BiquadHighPass {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl BiquadHighPass {
    fn new(cutoff_hz: f32, sample_rate: u32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * cutoff_hz / sample_rate as f32;
        let cos_w0 = w0.cos();
        let q = std::f32::consts::FRAC_1_SQRT_2; // Butterworth
        let alpha = w0.sin() / (2.0 * q);
        let a0 = 1.0 + alpha;

        Self {
            b0: ((1.0 + cos_w0) / 2.0) / a0,
            b1: (-(1.0 + cos_w0)) / a0,
            b2: ((1.0 + cos_w0) / 2.0) / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha) / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// AudioEnhancer
// ---------------------------------------------------------------------------

/// High-pass + AGC + soft-limiter chain for speech.
///
/// All methods are zero-latency: output length always equals input length.
pub struct AudioEnhancer {
    config: AgcConfig,
    highpass: BiquadHighPass,
    /// Current AGC gain (linear), smoothed per sample.
    gain: f32,
    /// Per-sample smoothing coefficients derived from attack/release times.
    attack_coef: f32,
    release_coef: f32,
    enabled: bool,
}

impl AudioEnhancer {
    /// Create an enhancer with default tuning for the given sample rate.
    pub fn new(sample_rate: u32) -> Self {
        Self::with_config(sample_rate, AgcConfig::default())
    }

    /// Create an enhancer with explicit tuning.
    pub fn with_config(sample_rate: u32, config: AgcConfig) -> Self {
        let sr = sample_rate as f32;
        // One-pole smoothing: coef = e^(−1 / (τ·sr)); gain moves ~63 % of
        // the way to its target within one time constant.
        let attack_coef = (-1.0 / (config.attack_ms / 1_000.0 * sr)).exp();
        let release_coef = (-1.0 / (config.release_ms / 1_000.0 * sr)).exp();

        log::info!(
            "AudioEnhancer: highpass={}Hz target_rms={:.0} max_gain={:+.1}dB",
            config.highpass_hz,
            config.target_rms,
            20.0 * config.max_gain.log10(),
        );

        Self {
            highpass: BiquadHighPass::new(config.highpass_hz, sample_rate),
            gain: 1.0,
            attack_coef,
            release_coef,
            config,
            enabled: true,
        }
    }

    /// Enable or disable the whole chain.
    ///
    /// When disabled, both filters pass audio through unchanged.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    // -----------------------------------------------------------------------
    // Core API
    // -----------------------------------------------------------------------

    /// DC removal + high-pass. Run on raw input *before* the noise filter.
    pub fn pre_filter(&mut self, audio: &[i16]) -> Vec<i16> {
        if !self.enabled || audio.is_empty() {
            return audio.to_vec();
        }
        audio
            .iter()
            .map(|&s| clamp_i16(self.highpass.process(s as f32)))
            .collect()
    }

    /// AGC + soft limiter. Run on denoised audio *after* the noise filter.
    pub fn post_filter(&mut self, audio: &[i16]) -> Vec<i16> {
        if !self.enabled || audio.is_empty() {
            return audio.to_vec();
        }

        // Chunk RMS decides the gain target; the gate holds adaptation
        // during silence so the noise floor isn't pumped up.
        let rms = chunk_rms(audio);
        let desired = if rms > self.config.noise_gate_rms {
            (self.config.target_rms / rms)
                .clamp(self.config.min_gain, self.config.max_gain)
        } else {
            self.gain
        };

        let knee = self.config.limiter_knee;
        let headroom = 32_767.0 - knee;

        audio
            .iter()
            .map(|&s| {
                // Smooth gain per sample: fast when reducing (attack),
                // slow when increasing (release).
                let coef = if desired < self.gain {
                    self.attack_coef
                } else {
                    self.release_coef
                };
                self.gain = coef * self.gain + (1.0 - coef) * desired;

                let x = s as f32 * self.gain;

                // Soft limiter: linear below the knee, tanh-compressed above,
                // asymptotically bounded by full scale.
                let y = if x.abs() <= knee {
                    x
                } else {
                    x.signum() * (knee + headroom * ((x.abs() - knee) / headroom).tanh())
                };

                clamp_i16(y)
            })
            .collect()
    }

    /// Clear filter state between utterances.
    ///
    /// The adapted AGC gain is intentionally *kept* — the same speaker is
    /// likely to continue at the same level, so re-learning from 0 dB every
    /// utterance would clip or duck the first words.
    pub fn reset(&mut self) {
        self.highpass.reset();
    }

    /// Current AGC gain in dB (for diagnostics).
    pub fn gain_db(&self) -> f32 {
        20.0 * self.gain.log10()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn chunk_rms(audio: &[i16]) -> f32 {
    let sum_sq: f64 = audio.iter().map(|&s| (s as f64) * (s as f64)).sum();
    ((sum_sq / audio.len() as f64) as f32).sqrt()
}

#[inline]
fn clamp_i16(s: f32) -> i16 {
    s.clamp(-32_768.0, 32_767.0) as i16
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a sine wave at `freq` Hz with the given peak amplitude.
    fn sine(freq: f32, peak: f32, sample_rate: u32, n: usize) -> Vec<i16> {
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (peak * (2.0 * std::f32::consts::PI * freq * t).sin()) as i16
            })
            .collect()
    }

    #[test]
    fn output_length_matches_input() {
        let mut enh = AudioEnhancer::new(16_000);
        let input = sine(300.0, 8_000.0, 16_000, 1_234);
        assert_eq!(enh.pre_filter(&input).len(), input.len());
        assert_eq!(enh.post_filter(&input).len(), input.len());
    }

    #[test]
    fn highpass_removes_dc_offset() {
        let mut enh = AudioEnhancer::new(16_000);
        let input = vec![5_000i16; 16_000]; // pure DC, 1 s
        let out = enh.pre_filter(&input);
        // After settling, DC must be gone.
        let tail_mean: f64 =
            out[8_000..].iter().map(|&s| s as f64).sum::<f64>() / 8_000.0;
        assert!(tail_mean.abs() < 50.0, "residual DC: {tail_mean}");
    }

    #[test]
    fn highpass_passes_speech_band() {
        let mut enh = AudioEnhancer::new(16_000);
        let input = sine(300.0, 8_000.0, 16_000, 16_000);
        let out = enh.pre_filter(&input);
        let in_rms = chunk_rms(&input);
        let out_rms = chunk_rms(&out[8_000..]);
        // 300 Hz is well above the 90 Hz cutoff — attenuation < 1 dB.
        assert!(out_rms > in_rms * 0.89, "in={in_rms} out={out_rms}");
    }

    #[test]
    fn agc_boosts_quiet_audio_toward_target() {
        let mut enh = AudioEnhancer::new(16_000);
        // Quiet speech: RMS ≈ 707, well below the 3 277 target.
        let input = sine(300.0, 1_000.0, 16_000, 16_000 * 4);
        let out = enh.post_filter(&input);
        let out_rms = chunk_rms(&out[out.len() / 2..]);
        assert!(
            out_rms > 2_500.0 && out_rms < 4_500.0,
            "rms after AGC: {out_rms}"
        );
    }

    #[test]
    fn agc_reduces_loud_audio_toward_target() {
        let mut enh = AudioEnhancer::new(16_000);
        // Loud speech: RMS ≈ 19 800, well above target.
        let input = sine(300.0, 28_000.0, 16_000, 16_000 * 4);
        let out = enh.post_filter(&input);
        let out_rms = chunk_rms(&out[out.len() / 2..]);
        assert!(
            out_rms > 2_500.0 && out_rms < 4_500.0,
            "rms after AGC: {out_rms}"
        );
    }

    #[test]
    fn agc_holds_gain_during_silence() {
        let mut enh = AudioEnhancer::new(16_000);
        // Learn a boost from quiet speech…
        let speech = sine(300.0, 1_000.0, 16_000, 16_000 * 4);
        enh.post_filter(&speech);
        let learned = enh.gain_db();
        assert!(learned > 6.0, "expected boost, got {learned} dB");
        // …then feed near-silence: gain must not change.
        let silence = vec![10i16; 16_000];
        enh.post_filter(&silence);
        assert!((enh.gain_db() - learned).abs() < 0.5);
    }

    #[test]
    fn limiter_prevents_hard_clipping() {
        let mut enh = AudioEnhancer::new(16_000);
        // Force max gain on already-loud audio so raw gain would clip.
        let input = sine(300.0, 30_000.0, 16_000, 16_000);
        let out = enh.post_filter(&input);
        assert!(out.iter().all(|&s| s > i16::MIN));
        // No flat-topped runs of identical extreme samples (hard clip
        // signature) — the limiter rounds peaks instead.
        let max = out.iter().map(|&s| s.unsigned_abs()).max().unwrap();
        let at_max = out.iter().filter(|&&s| s.unsigned_abs() == max).count();
        assert!(at_max < 20, "{at_max} samples pinned at peak {max}");
    }

    #[test]
    fn disabled_passes_through() {
        let mut enh = AudioEnhancer::new(16_000);
        enh.set_enabled(false);
        let input = sine(300.0, 1_000.0, 16_000, 480);
        assert_eq!(enh.pre_filter(&input), input);
        assert_eq!(enh.post_filter(&input), input);
    }

    #[test]
    fn reset_keeps_learned_gain() {
        let mut enh = AudioEnhancer::new(16_000);
        let speech = sine(300.0, 1_000.0, 16_000, 16_000 * 4);
        enh.post_filter(&speech);
        let learned = enh.gain_db();
        enh.reset();
        assert!((enh.gain_db() - learned).abs() < 0.01);
    }
}
