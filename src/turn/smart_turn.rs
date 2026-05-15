//! Smart turn analyzer — pure Rust, no external dependencies at runtime.

use std::time::Instant;

use crate::audio_process::resamplers::{ResamplerQuality, StreamResampler};
use super::engine::SmartTurnEngine;
use super::whisper_features::{Precision, WhisperFeatureExtractor, N_SAMPLES as MODEL_SAMPLES};

const DEFAULT_STOP_SECS: f32 = 3.0;
const DEFAULT_PRE_SPEECH_MS: f32 = 500.0;
const DEFAULT_MAX_DURATION_SECS: f32 = 8.0;
const MODEL_SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndOfTurnState {
    Complete,
    Incomplete,
}

#[derive(Clone)]
pub struct SmartTurnConfig {
    pub stop_secs: f32,
    pub pre_speech_ms: f32,
    pub max_duration_secs: f32,
    pub precision: Precision,
    pub resampler_quality: ResamplerQuality,
    /// Path to the smart-turn weights file (`smart_turn_weights.bin.gz`).
    /// If `None`, defaults to the file in the rustvani cache directory
    /// (`~/.rustvani/cache/` on Unix, `%LOCALAPPDATA%\rustvani\cache` on Windows).
    pub weights_path: Option<String>,
}

impl Default for SmartTurnConfig {
    fn default() -> Self {
        Self {
            stop_secs: DEFAULT_STOP_SECS,
            pre_speech_ms: DEFAULT_PRE_SPEECH_MS,
            max_duration_secs: DEFAULT_MAX_DURATION_SECS,
            precision: Precision::F32,
            resampler_quality: ResamplerQuality::Quick,
            weights_path: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TurnMetrics {
    pub is_complete: bool,
    pub probability: f32,
    pub e2e_processing_time_ms: f64,
}

struct AudioChunk {
    timestamp: f64,
    samples: Vec<f32>,
}

pub struct SmartTurnAnalyzer {
    stop_ms: f64,
    pre_speech_ms: f64,
    max_duration_secs: f64,
    resampler_quality: ResamplerQuality,
    sample_rate: u32,
    feature_extractor: WhisperFeatureExtractor,
    engine: SmartTurnEngine,
    epoch: Instant,
    audio_buffer: Vec<AudioChunk>,
    speech_triggered: bool,
    silence_ms: f64,
    speech_start_time: f64,
    vad_start_secs: f64,
}

impl SmartTurnAnalyzer {
    pub fn new(config: &SmartTurnConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let engine = SmartTurnEngine::new(config.weights_path.as_deref())
            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
        let feature_extractor = WhisperFeatureExtractor::new(config.precision);

        Ok(Self {
            stop_ms: config.stop_secs as f64 * 1000.0,
            pre_speech_ms: config.pre_speech_ms as f64,
            max_duration_secs: config.max_duration_secs as f64,
            resampler_quality: config.resampler_quality,
            sample_rate: 0,
            feature_extractor,
            engine,
            epoch: Instant::now(),
            audio_buffer: Vec::new(),
            speech_triggered: false,
            silence_ms: 0.0,
            speech_start_time: 0.0,
            vad_start_secs: 0.0,
        })
    }

    pub fn set_sample_rate(&mut self, sample_rate: u32) {
        self.sample_rate = sample_rate;
    }

    pub fn speech_triggered(&self) -> bool {
        self.speech_triggered
    }

    pub fn update_vad_start_secs(&mut self, vad_start_secs: f64) {
        self.vad_start_secs = vad_start_secs;
    }

    pub fn append_audio(&mut self, buffer: &[u8], is_speech: bool) -> EndOfTurnState {
        let now = self.epoch.elapsed().as_secs_f64();

        let audio_f32: Vec<f32> = buffer
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect();
        let num_samples = audio_f32.len();

        self.audio_buffer.push(AudioChunk {
            timestamp: now,
            samples: audio_f32,
        });

        let mut state = EndOfTurnState::Incomplete;

        if is_speech {
            self.silence_ms = 0.0;
            self.speech_triggered = true;
            if self.speech_start_time == 0.0 {
                self.speech_start_time = now;
            }
        } else if self.speech_triggered {
            let chunk_duration_ms = num_samples as f64 / (self.sample_rate as f64 / 1000.0);
            self.silence_ms += chunk_duration_ms;
            if self.silence_ms >= self.stop_ms {
                log::debug!(
                    "SmartTurn: end of turn due to stop_secs. Silence: {:.0}ms",
                    self.silence_ms
                );
                state = EndOfTurnState::Complete;
                self.clear_state(state);
            }
        } else {
            let max_buffer_secs =
                (self.pre_speech_ms / 1000.0) + (self.stop_ms / 1000.0) + self.max_duration_secs;
            let cutoff = now - max_buffer_secs;
            while let Some(first) = self.audio_buffer.first() {
                if first.timestamp < cutoff {
                    self.audio_buffer.remove(0);
                } else {
                    break;
                }
            }
        }

        state
    }

    pub fn analyze_end_of_turn(&mut self) -> (EndOfTurnState, Option<TurnMetrics>) {
        let (state, metrics) = self.process_speech_segment();
        if state == EndOfTurnState::Complete {
            self.clear_state(state);
        }
        log::debug!("SmartTurn: analyze result: {:?}", state);
        (state, metrics)
    }

    pub fn clear(&mut self) {
        self.clear_state(EndOfTurnState::Complete);
    }

    fn clear_state(&mut self, turn_state: EndOfTurnState) {
        self.speech_triggered = turn_state == EndOfTurnState::Incomplete;
        self.audio_buffer.clear();
        self.speech_start_time = 0.0;
        self.silence_ms = 0.0;
    }

    fn process_speech_segment(&mut self) -> (EndOfTurnState, Option<TurnMetrics>) {
        if self.audio_buffer.is_empty() {
            return (EndOfTurnState::Incomplete, None);
        }

        let effective_pre_speech_ms = self.pre_speech_ms + (self.vad_start_secs * 1000.0);
        let start_time = self.speech_start_time - (effective_pre_speech_ms / 1000.0);

        let start_index = self
            .audio_buffer
            .iter()
            .position(|c| c.timestamp >= start_time)
            .unwrap_or(0);

        let total_samples: usize = self.audio_buffer[start_index..]
            .iter()
            .map(|c| c.samples.len())
            .sum();

        let mut segment = Vec::with_capacity(total_samples);
        for chunk in &self.audio_buffer[start_index..] {
            segment.extend_from_slice(&chunk.samples);
        }

        let max_samples = (self.max_duration_secs * self.sample_rate as f64) as usize;
        if segment.len() > max_samples {
            let start = segment.len() - max_samples;
            segment = segment[start..].to_vec();
        }

        if segment.is_empty() {
            return (EndOfTurnState::Incomplete, None);
        }

        let start = Instant::now();

        match self.predict_endpoint(&segment) {
            Ok((prediction, probability)) => {
                let e2e_ms = start.elapsed().as_secs_f64() * 1000.0;
                let is_complete = prediction == 1;
                let state = if is_complete {
                    EndOfTurnState::Complete
                } else {
                    EndOfTurnState::Incomplete
                };

                log::trace!(
                    "SmartTurn: prob={:.4} complete={} time={:.1}ms",
                    probability, is_complete, e2e_ms
                );

                (state, Some(TurnMetrics {
                    is_complete,
                    probability,
                    e2e_processing_time_ms: e2e_ms,
                }))
            }
            Err(e) => {
                log::warn!("SmartTurn: prediction failed: {}", e);
                (EndOfTurnState::Incomplete, None)
            }
        }
    }

    fn predict_endpoint(
        &mut self,
        segment: &[f32],
    ) -> Result<(u8, f32), Box<dyn std::error::Error>> {
        let audio_16k = if self.sample_rate == MODEL_SAMPLE_RATE {
            segment.to_vec()
        } else {
            let mut resampler = StreamResampler::new(
                self.sample_rate,
                MODEL_SAMPLE_RATE,
                self.resampler_quality,
            );
            let mut resampled = resampler.process(segment);
            resampled.extend(resampler.flush());
            resampled
        };

        let audio_8s = if audio_16k.len() > MODEL_SAMPLES {
            audio_16k[audio_16k.len() - MODEL_SAMPLES..].to_vec()
        } else if audio_16k.len() < MODEL_SAMPLES {
            let mut padded = vec![0.0f32; MODEL_SAMPLES - audio_16k.len()];
            padded.extend_from_slice(&audio_16k);
            padded
        } else {
            audio_16k
        };

        let features = self.feature_extractor.extract(&audio_8s);
        let probability = self.engine.infer(&features);
        let prediction = if probability > 0.5 { 1u8 } else { 0u8 };

        Ok((prediction, probability))
    }
}
