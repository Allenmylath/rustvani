//! Whisper-compatible log-mel spectrogram feature extraction.
//!
//! Pipeline: normalize → reflect-pad → STFT (Hann, n_fft=400, hop=160) → power
//!         → mel filterbank → log10 → drop last frame → clamp → scale

use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 80;
const NUM_FREQ_BINS: usize = N_FFT / 2 + 1;
pub(crate) const N_SAMPLES: usize = 16_000 * 8; // 128_000
const PAD_SIZE: usize = N_FFT / 2;
const PADDED_LENGTH: usize = N_SAMPLES + N_FFT;
const NUM_FRAMES: usize = 1 + (PADDED_LENGTH - N_FFT) / HOP_LENGTH; // 801
const OUTPUT_FRAMES: usize = NUM_FRAMES - 1; // 800

const MEL_FILTERS_BYTES: &[u8] = include_bytes!("mel_filters_80x201_f32.bin");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision { F32, F64 }

pub(crate) struct WhisperFeatureExtractor { inner: InnerState }

impl WhisperFeatureExtractor {
    pub fn new(precision: Precision) -> Self {
        Self { inner: match precision {
            Precision::F32 => InnerState::F32(F32State::new()),
            Precision::F64 => InnerState::F64(F64State::new()),
        }}
    }

    /// Extract [80 × 800] features from 128,000 samples at 16 kHz.
    pub fn extract(&mut self, audio: &[f32]) -> Vec<f32> {
        assert_eq!(audio.len(), N_SAMPLES);
        match &mut self.inner {
            InnerState::F32(s) => s.extract(audio),
            InnerState::F64(s) => s.extract(audio),
        }
    }

    pub fn precision(&self) -> Precision {
        match &self.inner {
            InnerState::F32(_) => Precision::F32,
            InnerState::F64(_) => Precision::F64,
        }
    }
}

enum InnerState { F32(F32State), F64(F64State) }

fn load_mel_filters_f32() -> Vec<f32> {
    let f: Vec<f32> = MEL_FILTERS_BYTES.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    assert_eq!(f.len(), NUM_FREQ_BINS * N_MELS);
    f
}

fn load_mel_filters_f64() -> Vec<f64> {
    load_mel_filters_f32().iter().map(|&v| v as f64).collect()
}

macro_rules! impl_extractor {
    ($name:ident, $float:ty, $mel_loader:ident, $mel_floor:expr) => {
        struct $name {
            hann_window: [$float; N_FFT],
            mel_filters: Vec<$float>,
            fft: Arc<dyn Fft<$float>>,
            padded: Vec<$float>,
            fft_buffer: Vec<Complex<$float>>,
            mel_spec: Vec<$float>,
        }

        impl $name {
            fn new() -> Self {
                let mut hann_window = [0.0 as $float; N_FFT];
                for i in 0..N_FFT {
                    hann_window[i] = (0.5
                        * (1.0
                            - (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()))
                        as $float;
                }
                let mel_filters = $mel_loader();
                let mut planner = FftPlanner::<$float>::new();
                let fft = planner.plan_fft_forward(N_FFT);

                Self {
                    hann_window, mel_filters, fft,
                    padded: vec![0.0 as $float; PADDED_LENGTH],
                    fft_buffer: vec![Complex::zero(); N_FFT],
                    mel_spec: vec![0.0 as $float; N_MELS * NUM_FRAMES],
                }
            }

            fn extract(&mut self, audio: &[f32]) -> Vec<f32> {
                // Step 1+2: Normalize + reflect-pad
                let mean: $float =
                    audio.iter().map(|&x| x as $float).sum::<$float>() / audio.len() as $float;
                let variance: $float = audio
                    .iter()
                    .map(|&x| { let d = x as $float - mean; d * d })
                    .sum::<$float>()
                    / audio.len() as $float;
                let inv_std: $float = 1.0 as $float / (variance.sqrt() + 1e-7 as $float);

                for i in 0..PAD_SIZE {
                    self.padded[i] = (audio[PAD_SIZE - i] as $float - mean) * inv_std;
                }
                for i in 0..N_SAMPLES {
                    self.padded[PAD_SIZE + i] = (audio[i] as $float - mean) * inv_std;
                }
                for i in 0..PAD_SIZE {
                    self.padded[PAD_SIZE + N_SAMPLES + i] =
                        (audio[N_SAMPLES - 2 - i] as $float - mean) * inv_std;
                }

                // Step 3+4: STFT → power → mel (streamed)
                self.mel_spec.fill(0.0 as $float);

                for frame_idx in 0..NUM_FRAMES {
                    let start = frame_idx * HOP_LENGTH;
                    for i in 0..N_FFT {
                        self.fft_buffer[i] = Complex::new(
                            self.padded[start + i] * self.hann_window[i],
                            0.0 as $float,
                        );
                    }
                    self.fft.process(&mut self.fft_buffer);

                    for freq in 0..NUM_FREQ_BINS {
                        let re = self.fft_buffer[freq].re;
                        let im = self.fft_buffer[freq].im;
                        let power = re * re + im * im;
                        if power == 0.0 as $float { continue; }
                        let filter_row = freq * N_MELS;
                        for mel in 0..N_MELS {
                            let w = self.mel_filters[filter_row + mel];
                            if w != 0.0 as $float {
                                self.mel_spec[mel * NUM_FRAMES + frame_idx] += w * power;
                            }
                        }
                    }
                }

                // Step 5: Floor + Log10
                let mel_floor: $float = $mel_floor;
                for v in self.mel_spec.iter_mut() {
                    if *v < mel_floor { *v = mel_floor; }
                    *v = v.log10();
                }

                // Step 6: Drop last frame → [80, 800]
                let mut features = vec![0.0f32; N_MELS * OUTPUT_FRAMES];
                for mel in 0..N_MELS {
                    let src = mel * NUM_FRAMES;
                    let dst = mel * OUTPUT_FRAMES;
                    for frame in 0..OUTPUT_FRAMES {
                        features[dst + frame] = self.mel_spec[src + frame] as f32;
                    }
                }

                // Step 7: Clamp to max - 8.0
                let global_max = features.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let floor = global_max - 8.0;
                for v in features.iter_mut() { *v = v.max(floor); }

                // Step 8: Scale
                for v in features.iter_mut() { *v = (*v + 4.0) / 4.0; }

                features
            }
        }
    };
}

impl_extractor!(F32State, f32, load_mel_filters_f32, 1e-10f32);
impl_extractor!(F64State, f64, load_mel_filters_f64, 1e-10f64);
