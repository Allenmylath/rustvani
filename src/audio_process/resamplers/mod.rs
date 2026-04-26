//! Streaming sample-rate converter using `rubato` (pure Rust).
//!
//! Wraps a sinc interpolation resampler with internal buffering so callers
//! can push arbitrary-length audio and pull whatever output is ready.
//!
//! ```ignore
//! let mut r = StreamResampler::new(16_000, 48_000, ResamplerQuality::Quick);
//! let out = r.process(&input_samples);   // may be empty if buffering
//! let tail = r.flush();                  // zero-pads remainder
//! ```

use log;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

// ---------------------------------------------------------------------------
// Quality presets — mirrors Python pipecat's SOXRStreamAudioResampler
// ---------------------------------------------------------------------------

/// Resampler quality preset.
///
/// Higher quality → longer sinc kernel → more latency.
/// For real-time voice pipelines, `Quick` or `Low` is usually fine.
#[derive(Debug, Clone, Copy)]
pub enum ResamplerQuality {
    /// Lowest latency, adequate for voice.
    Quick,
    /// Slight improvement over Quick.
    Low,
    /// Balanced.
    Medium,
    /// High quality, more CPU.
    High,
    /// Best quality, most CPU/latency.
    VeryHigh,
}

impl ResamplerQuality {
    fn sinc_params(self) -> SincInterpolationParameters {
        match self {
            Self::Quick => SincInterpolationParameters {
                sinc_len: 32,
                f_cutoff: 0.85,
                interpolation: SincInterpolationType::Nearest,
                oversampling_factor: 64,
                window: WindowFunction::Hann,
            },
            Self::Low => SincInterpolationParameters {
                sinc_len: 64,
                f_cutoff: 0.90,
                interpolation: SincInterpolationType::Nearest,
                oversampling_factor: 128,
                window: WindowFunction::Hann,
            },
            Self::Medium => SincInterpolationParameters {
                sinc_len: 128,
                f_cutoff: 0.92,
                interpolation: SincInterpolationType::Linear,
                oversampling_factor: 160,
                window: WindowFunction::BlackmanHarris,
            },
            Self::High => SincInterpolationParameters {
                sinc_len: 192,
                f_cutoff: 0.94,
                interpolation: SincInterpolationType::Cubic,
                oversampling_factor: 192,
                window: WindowFunction::BlackmanHarris2,
            },
            Self::VeryHigh => SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Cubic,
                oversampling_factor: 256,
                window: WindowFunction::BlackmanHarris2,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// StreamResampler
// ---------------------------------------------------------------------------

/// Streaming mono resampler with internal buffering.
///
/// Buffers input until a full chunk is available, resamples, and returns
/// the output.  Callers should call [`flush`](Self::flush) at the end of
/// a speech segment to drain any remaining buffered samples.
pub struct StreamResampler {
    inner: SincFixedIn<f32>,
    buffer: Vec<f32>,
    chunk_size: usize,
}

impl StreamResampler {
    /// Create a new streaming resampler.
    ///
    /// # Arguments
    /// * `from_rate` — input sample rate in Hz
    /// * `to_rate`   — output sample rate in Hz
    /// * `quality`   — interpolation quality preset
    ///
    /// # Panics
    /// Panics if `rubato` cannot create the resampler (e.g. invalid rates).
    pub fn new(from_rate: u32, to_rate: u32, quality: ResamplerQuality) -> Self {
        let ratio = to_rate as f64 / from_rate as f64;
        // Use 480 as chunk size — aligns with RNNoise frame length at 48 kHz
        // and keeps latency ≤ 30 ms even at 16 kHz.
        let chunk_size: usize = 480;

        let inner = SincFixedIn::<f32>::new(
            ratio,
            2.0, // max relative ratio (headroom)
            quality.sinc_params(),
            chunk_size,
            1, // mono
        )
        .expect("failed to create rubato resampler");

        Self {
            inner,
            buffer: Vec::with_capacity(chunk_size * 2),
            chunk_size,
        }
    }

    /// Push samples in, get resampled samples out.
    ///
    /// Returns an empty vec if the internal buffer hasn't accumulated a full
    /// chunk yet.  Over time, total output samples ≈ total input × ratio.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        self.buffer.extend_from_slice(input);

        let mut output = Vec::new();

        while self.buffer.len() >= self.chunk_size {
            let chunk: Vec<f32> = self.buffer.drain(..self.chunk_size).collect();
            match self.inner.process(&[chunk], None) {
                Ok(result) => {
                    if let Some(ch) = result.first() {
                        output.extend_from_slice(ch);
                    }
                }
                Err(e) => {
                    log::warn!("StreamResampler: process error: {}", e);
                    break;
                }
            }
        }

        output
    }

    /// Flush remaining buffered samples by zero-padding to a full chunk.
    ///
    /// Returns the resampled tail (may include a few trailing zeros).
    /// After flushing, the internal buffer is empty.
    pub fn flush(&mut self) -> Vec<f32> {
        if self.buffer.is_empty() {
            return Vec::new();
        }

        // Zero-pad to chunk boundary
        self.buffer.resize(self.chunk_size, 0.0);
        let chunk: Vec<f32> = self.buffer.drain(..).collect();

        match self.inner.process(&[chunk], None) {
            Ok(result) => result.into_iter().next().unwrap_or_default(),
            Err(e) => {
                log::warn!("StreamResampler: flush error: {}", e);
                Vec::new()
            }
        }
    }

    /// Discard all buffered samples and reset the resampler state.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.inner.reset();
    }
}
