//! Audio conditioning for VAD: whatever arrives → mono PCM at the model's rate.
//!
//! Silero runs at a fixed rate (16 kHz native, 8 or 16 kHz on ort) and reads
//! its window as mono i16 LE. Transports don't always deliver that: a WebRTC
//! leg decodes Opus at 48 kHz, an SFU tap may be stereo, a caller may switch
//! codecs mid-call. `VadInput` takes each [`AudioRawData`] at the rate and
//! channel count it carries and hands the VAD (and SmartTurn) audio in the
//! format the model was built for.
//!
//! Audio already in that format is borrowed, not copied, so existing 16 kHz
//! mono pipelines pay nothing.
//!
//! Resampling uses [`StreamResampler`], which works in 480-sample chunks:
//! 10 ms of delay at 48 kHz, 30 ms at 16 kHz, 60 ms at 8 kHz. That delay
//! only applies to the VAD's view of the audio; the frame passed downstream
//! is untouched.

use std::borrow::Cow;

use crate::audio_process::downmix_to_mono;
use crate::audio_process::resamplers::{ResamplerQuality, StreamResampler};
use crate::frames::AudioRawData;

pub struct VadInput {
    target_rate: u32,
    source_rate: Option<u32>,
    resampler: Option<StreamResampler>,
}

impl VadInput {
    /// `target_rate` is the VAD model's rate ([`VadAnalyzer::sample_rate`](super::VadAnalyzer::sample_rate)).
    pub fn new(target_rate: u32) -> Self {
        Self { target_rate, source_rate: None, resampler: None }
    }

    pub fn target_rate(&self) -> u32 {
        self.target_rate
    }

    /// Mono i16 LE at `target_rate`. Empty while the resampler fills its
    /// first chunk. A rate of 0 is read as "already at the target rate".
    pub fn convert<'a>(&mut self, audio: &'a AudioRawData) -> Cow<'a, [u8]> {
        let rate = if audio.sample_rate == 0 { self.target_rate } else { audio.sample_rate };
        let channels = audio.num_channels.max(1) as usize;

        if rate == self.target_rate && channels == 1 {
            self.note_rate(rate);
            return Cow::Borrowed(&audio.audio[..]);
        }

        let mono = downmix(&audio.audio, channels);
        if rate == self.target_rate {
            self.note_rate(rate);
            return Cow::Owned(to_bytes(mono.iter().map(|&s| s as f32)));
        }

        if self.source_rate != Some(rate) || self.resampler.is_none() {
            log::info!("VadInput: resampling {}Hz → {}Hz for VAD", rate, self.target_rate);
            self.resampler = Some(StreamResampler::new(rate, self.target_rate, ResamplerQuality::Quick));
            self.source_rate = Some(rate);
        }
        let input: Vec<f32> = mono.iter().map(|&s| s as f32).collect();
        let out = self.resampler.as_mut().expect("built above").process(&input);
        Cow::Owned(to_bytes(out.into_iter()))
    }

    /// Drop buffered samples (between sessions).
    pub fn reset(&mut self) {
        if let Some(r) = &mut self.resampler {
            r.reset();
        }
    }

    /// Back at the target rate: a later return to the old rate starts clean.
    fn note_rate(&mut self, rate: u32) {
        if self.resampler.is_some() {
            log::info!("VadInput: input now at {}Hz — resampler dropped", rate);
            self.resampler = None;
        }
        self.source_rate = Some(rate);
    }
}

/// Interleaved i16 LE bytes → mono samples.
fn downmix(bytes: &[u8], channels: usize) -> Vec<i16> {
    let samples: Vec<i16> =
        bytes.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect();
    downmix_to_mono(&samples, channels)
}

fn to_bytes(samples: impl Iterator<Item = f32>) -> Vec<u8> {
    samples
        .flat_map(|s| (s.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16).to_le_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pcm(samples: &[i16]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    fn samples(bytes: &[u8]) -> Vec<i16> {
        bytes.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect()
    }

    #[test]
    fn matching_mono_audio_is_borrowed() {
        let mut input = VadInput::new(16_000);
        let data = AudioRawData::new(pcm(&[1, 2, 3, 4]), 16_000, 1);
        let out = input.convert(&data);
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(&out[..], &data.audio[..]);
    }

    #[test]
    fn stereo_is_averaged_to_mono() {
        let mut input = VadInput::new(16_000);
        let data = AudioRawData::new(pcm(&[100, 300, -50, -150]), 16_000, 2);
        assert_eq!(samples(&input.convert(&data)), vec![200, -100]);
    }

    #[test]
    fn zero_rate_is_read_as_the_target() {
        let mut input = VadInput::new(16_000);
        let data = AudioRawData::new(pcm(&[7, 8]), 0, 1);
        assert!(matches!(input.convert(&data), Cow::Borrowed(_)));
    }

    #[test]
    fn higher_rates_come_out_at_the_target_rate() {
        // One second at 48 kHz stereo → about one second at 16 kHz mono.
        let mut input = VadInput::new(16_000);
        let mut total = 0;
        for _ in 0..100 {
            let chunk = AudioRawData::new(pcm(&[1000i16; 480 * 2]), 48_000, 2);
            total += input.convert(&chunk).len() / 2;
        }
        assert!((15_000..=16_000).contains(&total), "got {total} samples");
    }

    #[test]
    fn a_rate_change_rebuilds_the_resampler() {
        let mut input = VadInput::new(16_000);
        input.convert(&AudioRawData::new(pcm(&[0; 960]), 48_000, 1));
        assert_eq!(input.source_rate, Some(48_000));
        input.convert(&AudioRawData::new(pcm(&[0; 160]), 8_000, 1));
        assert_eq!(input.source_rate, Some(8_000));
        input.convert(&AudioRawData::new(pcm(&[0; 320]), 16_000, 1));
        assert!(input.resampler.is_none());
    }
}
