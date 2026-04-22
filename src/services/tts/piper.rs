//! Piper TTS — local ONNX-based text-to-speech (US English).
//!
//! Runs inference entirely on-device via `ort`. No network calls.
//!
//! Design choices for scalability / memory:
//!   - `Arc<Mutex<PiperModel>>` — the model is behind a std::sync::Mutex
//!     (same pattern as SileroVad) allowing safe `&mut self` access for
//!     session.run(). Multiple pipeline instances can share one Arc to
//!     avoid duplicating model weights in memory.
//!   - Inference runs on `spawn_blocking` — never blocks tokio workers.
//!   - Output is chunked into 20 ms frames for smooth streaming playback.
//!
//! Timing prints (same convention as sarvam.rs):
//!   [tts:piper] phonemize   — espeak-ng subprocess duration
//!   [tts:piper] inference    — ONNX session.run() duration
//!   [tts:piper] first_chunk  — first 20 ms chunk pushed downstream

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use log;
use ort::session::builder::SessionBuilder;
use ort::session::Session;
use ort::value::Value;
use serde::Deserialize;

use crate::error::{PipecatError, Result};
use crate::frames::{
    AudioRawData, ControlFrame, DataFrame, Frame, FrameDirection, FrameHandler, FrameInner,
    FrameProcessor, SystemFrame,
};
use crate::utils::sentence_splitter::{extract_sentences, find_sentence_end};
use crate::utils::text_preprocessor::preprocess_for_tts;

// ---------------------------------------------------------------------------
// Timing helper (same as sarvam.rs)
// ---------------------------------------------------------------------------

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

// ---------------------------------------------------------------------------
// Quality / voice presets
// ---------------------------------------------------------------------------

/// Voice quality — determines which Piper ONNX model is used.
///
/// Memory footprint & inference speed scale with quality:
///   Low    ≈ 15 MB, ~80–150 ms/sentence
///   Medium ≈ 60 MB, ~150–300 ms/sentence
///   High   ≈ 65 MB, ~200–400 ms/sentence
/// (times on a 4-core server CPU, single-sentence, vary with length)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PiperQuality {
    Low,
    Medium,
    High,
}

impl PiperQuality {
    /// Default Piper model name for US English at this quality level.
    pub fn default_model_name(&self) -> &'static str {
        match self {
            Self::Low    => "en_US-lessac-low",
            Self::Medium => "en_US-lessac-medium",
            Self::High   => "en_US-lessac-high",
        }
    }

    /// Recommended intra-op thread count — lower for light models to save
    /// memory, slightly higher for heavy models to keep latency down.
    pub fn default_threads(&self) -> usize {
        match self {
            Self::Low    => 1,
            Self::Medium => 2,
            Self::High   => 2,
        }
    }
}

impl std::fmt::Display for PiperQuality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low    => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High   => write!(f, "high"),
        }
    }
}

// ---------------------------------------------------------------------------
// Piper JSON config (subset we need)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct PiperModelConfig {
    audio: PiperAudioConfig,
    #[serde(default)]
    phoneme_id_map: HashMap<String, Vec<i64>>,
    #[serde(default)]
    inference: PiperInferenceConfig,
    #[allow(dead_code)]
    #[serde(default)]
    num_speakers: usize,
}

#[derive(Debug, Deserialize)]
struct PiperAudioConfig {
    sample_rate: u32,
}

#[derive(Debug, Deserialize, Default)]
struct PiperInferenceConfig {
    #[serde(default = "default_noise_scale")]
    noise_scale: f32,
    #[serde(default = "default_length_scale")]
    length_scale: f32,
    #[serde(default = "default_noise_w")]
    noise_w: f32,
}

fn default_noise_scale() -> f32 { 0.667 }
fn default_length_scale() -> f32 { 1.0 }
fn default_noise_w() -> f32 { 0.8 }

// ---------------------------------------------------------------------------
// Config (user-facing)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PiperTtsConfig {
    /// Quality level — selects model if model_path is not set.
    pub quality: PiperQuality,
    /// Explicit path to `.onnx` model file. If `None`, derived from
    /// `model_dir` + quality.
    pub model_path: Option<PathBuf>,
    /// Explicit path to `.onnx.json` config file. If `None`, derived from
    /// `model_dir` + quality.
    pub config_path: Option<PathBuf>,
    /// Directory containing Piper model files. Defaults to `./piper-models`.
    pub model_dir: PathBuf,
    /// Speaker ID for multi-speaker models. `None` for single-speaker.
    pub speaker_id: Option<i64>,
    /// Speed: < 1.0 = faster, > 1.0 = slower. Overrides model default.
    pub length_scale: Option<f32>,
    /// Phoneme variation. Overrides model default.
    pub noise_scale: Option<f32>,
    /// Phoneme width variation. Overrides model default.
    pub noise_w: Option<f32>,
    /// ONNX intra-op parallelism threads. `None` = use quality default.
    /// Keep low (1–2) to save memory under concurrency.
    pub num_threads: Option<usize>,
    /// Sentence buffer — minimum chars before attempting sentence split.
    pub min_buffer_size: usize,
    /// Maximum chars per TTS chunk.
    pub max_chunk_length: usize,
}

impl Default for PiperTtsConfig {
    fn default() -> Self {
        Self {
            quality:          PiperQuality::Medium,
            model_path:       None,
            config_path:      None,
            model_dir:        PathBuf::from("./piper-models"),
            speaker_id:       None,
            length_scale:     None,
            noise_scale:      None,
            noise_w:          None,
            num_threads:      None,
            min_buffer_size:  50,
            max_chunk_length: 150,
        }
    }
}

impl PiperTtsConfig {
    fn resolved_model_path(&self) -> PathBuf {
        self.model_path.clone().unwrap_or_else(|| {
            self.model_dir
                .join(self.quality.default_model_name())
                .with_extension("onnx")
        })
    }

    fn resolved_config_path(&self) -> PathBuf {
        self.config_path.clone().unwrap_or_else(|| {
            // Piper convention: model.onnx → model.onnx.json
            let mut p = self.resolved_model_path();
            let name = format!(
                "{}.json",
                p.file_name().unwrap_or_default().to_string_lossy()
            );
            p.set_file_name(name);
            p
        })
    }
}

// ---------------------------------------------------------------------------
// PiperModel — mutable model state behind Arc<Mutex<>>
// ---------------------------------------------------------------------------

/// Model state. Wrapped in `Arc<Mutex<>>` because `session.run()` requires
/// `&mut self` in ort v2. Same pattern as `SileroVadInner`.
pub struct PiperModel {
    session:        Session,
    phoneme_id_map: HashMap<String, Vec<i64>>,
    sample_rate:    u32,
    noise_scale:    f32,
    length_scale:   f32,
    noise_w:        f32,
    speaker_id:     Option<i64>,
    pad_id:         i64,
    bos_id:         i64,
    eos_id:         i64,
}

impl PiperModel {
    pub fn load(config: &PiperTtsConfig) -> std::result::Result<Self, String> {
        let model_path  = config.resolved_model_path();
        let config_path = config.resolved_config_path();

        // ---- Load JSON config ----
        let json_bytes = std::fs::read(&config_path)
            .map_err(|e| format!("Failed to read Piper config {}: {}", config_path.display(), e))?;
        let model_cfg: PiperModelConfig = serde_json::from_slice(&json_bytes)
            .map_err(|e| format!("Failed to parse Piper config: {}", e))?;

        // ---- Build ONNX session (same pattern as silero.rs) ----
        let session = SessionBuilder::new()
            .map_err(|e| format!("SessionBuilder error: {}", e))?
            .commit_from_file(&model_path)
            .map_err(|e| format!("Failed to load ONNX model {}: {}", model_path.display(), e))?;

        log::info!(
            "PiperTts: loaded {} ({} quality, {} Hz)",
            model_path.display(),
            config.quality,
            model_cfg.audio.sample_rate,
        );

        // ---- Resolve scales (user override > model default) ----
        let noise_scale  = config.noise_scale.unwrap_or(model_cfg.inference.noise_scale);
        let length_scale = config.length_scale.unwrap_or(model_cfg.inference.length_scale);
        let noise_w      = config.noise_w.unwrap_or(model_cfg.inference.noise_w);

        // ---- Standard Piper special token IDs ----
        // "_" = pad (inserted between every phoneme)
        // "^" = BOS, "$" = EOS
        let pad_id = model_cfg.phoneme_id_map.get("_")
            .and_then(|v| v.first().copied())
            .unwrap_or(0);
        let bos_id = model_cfg.phoneme_id_map.get("^")
            .and_then(|v| v.first().copied())
            .unwrap_or(0);
        let eos_id = model_cfg.phoneme_id_map.get("$")
            .and_then(|v| v.first().copied())
            .unwrap_or(0);

        Ok(Self {
            session,
            phoneme_id_map: model_cfg.phoneme_id_map,
            sample_rate:    model_cfg.audio.sample_rate,
            noise_scale,
            length_scale,
            noise_w,
            speaker_id:     config.speaker_id,
            pad_id,
            bos_id,
            eos_id,
        })
    }

    /// Convert IPA string from espeak-ng into Piper phoneme IDs.
    fn phonemes_to_ids(&self, ipa: &str) -> Vec<i64> {
        let mut ids = Vec::with_capacity(ipa.len() * 3);
        ids.push(self.bos_id);
        ids.push(self.pad_id);

        for ch in ipa.chars() {
            let key = ch.to_string();
            if let Some(mapped) = self.phoneme_id_map.get(&key) {
                for &id in mapped {
                    ids.push(id);
                    ids.push(self.pad_id);
                }
            }
            // Unknown phonemes silently skipped (Piper convention).
        }

        ids.push(self.eos_id);
        ids
    }

    /// Run phonemization + ONNX inference. **Blocking** — call from
    /// `spawn_blocking` only. Requires `&mut self` because ort's
    /// `session.run()` takes `&mut Session`.
    pub fn synthesize(&mut self, text: &str) -> std::result::Result<Vec<u8>, String> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        // ---- 1. Phonemize via espeak-ng ----
        let t0 = now();
        let ipa = phonemize_espeak(text)?;
        let t1 = now();
        println!(
            "[{:.3}] [tts:piper] phonemize  {:.1}ms  ({} chars → {} IPA chars)",
            t1, (t1 - t0) * 1000.0, text.len(), ipa.len()
        );

        if ipa.trim().is_empty() {
            return Ok(Vec::new());
        }

        // ---- 2. Phoneme IDs ----
        let phoneme_ids = self.phonemes_to_ids(&ipa);
        let id_count = phoneme_ids.len();

        // ---- 3. Build input tensors (same Value::from_array pattern as silero.rs) ----
        let input_val = Value::from_array(
            ([1usize, id_count], phoneme_ids),
        ).map_err(|e| format!("Input tensor error: {}", e))?;

        let lengths_val = Value::from_array(
            ([1usize], vec![id_count as i64]),
        ).map_err(|e| format!("Lengths tensor error: {}", e))?;

        let scales_val = Value::from_array(
            ([3usize], vec![self.noise_scale, self.length_scale, self.noise_w]),
        ).map_err(|e| format!("Scales tensor error: {}", e))?;

        // ---- 4. Run inference ----
        let t2 = now();

        let outputs = if let Some(sid) = self.speaker_id {
            let sid_val = Value::from_array(
                ([1usize], vec![sid]),
            ).map_err(|e| format!("SID tensor error: {}", e))?;

            self.session.run(ort::inputs![
                "input"         => input_val,
                "input_lengths" => lengths_val,
                "scales"        => scales_val,
                "sid"           => sid_val
            ]).map_err(|e| format!("Inference error: {}", e))?
        } else {
            self.session.run(ort::inputs![
                "input"         => input_val,
                "input_lengths" => lengths_val,
                "scales"        => scales_val
            ]).map_err(|e| format!("Inference error: {}", e))?
        };

        let t3 = now();
        println!(
            "[{:.3}] [tts:piper] inference  {:.1}ms",
            t3, (t3 - t2) * 1000.0
        );

        // ---- 5. Extract audio: output shape is [1, 1, samples] as f32 ----
        let audio_f32: Vec<f32> = outputs["output"]
            .try_extract_array::<f32>()
            .map_err(|e| format!("Audio extract error: {}", e))?
            .iter()
            .copied()
            .collect();

        drop(outputs);

        // ---- 6. Convert f32 → 16-bit PCM (little-endian) ----
        let mut pcm = Vec::with_capacity(audio_f32.len() * 2);
        for sample in &audio_f32 {
            let clamped = sample.clamp(-1.0, 1.0);
            let s16 = (clamped * 32767.0) as i16;
            pcm.extend_from_slice(&s16.to_le_bytes());
        }

        Ok(pcm)
    }
}

// ---------------------------------------------------------------------------
// espeak-ng phonemization (subprocess)
// ---------------------------------------------------------------------------

/// Call espeak-ng to convert English text → IPA. **Blocking.**
fn phonemize_espeak(text: &str) -> std::result::Result<String, String> {
    let output = std::process::Command::new("espeak-ng")
        .args([
            "--ipa", "-q",
            "--sep= ",
            "-v", "en-us",
            text,
        ])
        .output()
        .map_err(|e| format!(
            "espeak-ng not found or failed to execute: {}. \
             Install with: apt-get install espeak-ng / dnf install espeak-ng",
            e
        ))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("espeak-ng error: {}", stderr));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

// ---------------------------------------------------------------------------
// Internal handler state
// ---------------------------------------------------------------------------

struct PiperState {
    text_buffer:  String,
    bot_speaking: bool,
}

impl PiperState {
    fn new() -> Self {
        Self {
            text_buffer:  String::new(),
            bot_speaking: false,
        }
    }
}

// ---------------------------------------------------------------------------
// PiperTtsHandler
// ---------------------------------------------------------------------------

pub struct PiperTtsHandler {
    config: PiperTtsConfig,
    /// Shared model — `Arc<Mutex<>>` because `session.run()` needs `&mut`.
    /// Same pattern as `SileroVad` wrapping `SileroVadInner`.
    model: Arc<Mutex<PiperModel>>,
    state: tokio::sync::Mutex<PiperState>,
}

impl PiperTtsHandler {
    pub fn new(config: PiperTtsConfig) -> std::result::Result<Self, String> {
        let model = PiperModel::load(&config)?;
        Ok(Self {
            config,
            model: Arc::new(Mutex::new(model)),
            state: tokio::sync::Mutex::new(PiperState::new()),
        })
    }

    /// Construct from a pre-loaded, shared model. Use this when running
    /// multiple pipelines to avoid duplicating model weights in memory.
    pub fn with_shared_model(
        config: PiperTtsConfig,
        model: Arc<Mutex<PiperModel>>,
    ) -> Self {
        Self {
            config,
            model,
            state: tokio::sync::Mutex::new(PiperState::new()),
        }
    }

    pub fn into_processor(self) -> FrameProcessor {
        FrameProcessor::new("PiperTts", Box::new(self), false)
    }

    /// Expose the inner Arc so callers can share it across
    /// multiple handler instances.
    pub fn shared_model(&self) -> Arc<Mutex<PiperModel>> {
        self.model.clone()
    }

    /// Read sample_rate from model (lock briefly, no .await while held).
    fn sample_rate(&self) -> u32 {
        self.model.lock().unwrap().sample_rate
    }

    // ---- Core: synthesize + chunk + push ----

    async fn synthesize_and_push(
        &self,
        text: &str,
        processor: &FrameProcessor,
    ) -> Result<()> {
        let text = preprocess_for_tts(text);
        if text.trim().is_empty() {
            return Ok(());
        }

        let model = self.model.clone();
        let text_owned = text.to_string();

        // Offload CPU-heavy work to blocking thread pool.
        // Lock is acquired inside spawn_blocking — never held across .await.
        let pcm = tokio::task::spawn_blocking(move || {
            let mut guard = model.lock().unwrap();
            guard.synthesize(&text_owned)
        })
        .await
        .map_err(|e| PipecatError::Pipeline(format!("spawn_blocking join: {}", e)))?
        .map_err(|e| PipecatError::Pipeline(e))?;

        if pcm.is_empty() {
            return Ok(());
        }

        // ---- Chunk into 20 ms frames ----
        let sample_rate = self.sample_rate();
        let bytes_per_20ms = chunk_size_bytes(sample_rate, 20);
        let mut first = true;

        for chunk in pcm.chunks(bytes_per_20ms) {
            if first {
                let ts = now();
                println!("[{:.3}] [tts:piper] first_chunk  {} bytes", ts, chunk.len());
                first = false;
            }

            let frame = Frame::output_audio_raw(
                AudioRawData::new(chunk.to_vec(), sample_rate, 1),
            );
            processor
                .push_frame(frame, FrameDirection::Downstream)
                .await?;
        }

        Ok(())
    }
}

/// 20 ms of mono 16-bit PCM at the given sample rate, in bytes.
fn chunk_size_bytes(sample_rate: u32, ms: u32) -> usize {
    let samples = (sample_rate * ms) / 1000;
    (samples as usize) * 2 // 16-bit = 2 bytes per sample
}

// ---------------------------------------------------------------------------
// FrameHandler impl
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameHandler for PiperTtsHandler {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match &frame.inner {
            // ---- Lifecycle ----
            FrameInner::System(SystemFrame::Start(_)) => {
                processor.push_frame(frame, direction).await?;
                log::info!(
                    "PiperTts: ready ({} quality, {} Hz)",
                    self.config.quality,
                    self.sample_rate(),
                );
            }

            // ---- LLM text streaming: buffer + sentence-split ----
            FrameInner::Control(ControlFrame::LLMFullResponseStart) => {
                processor.push_frame(frame, direction).await?;
            }

            FrameInner::Data(DataFrame::LLMText(text)) => {
                let text = text.clone();
                let min_buf = self.config.min_buffer_size;
                let max_chunk = self.config.max_chunk_length;

                let sentences = {
                    let mut state = self.state.lock().await;
                    state.text_buffer.push_str(&text);

                    if state.text_buffer.len() < min_buf
                        && find_sentence_end(&state.text_buffer).is_none()
                    {
                        vec![]
                    } else {
                        extract_sentences(&mut state.text_buffer, max_chunk)
                    }
                };

                for sentence in sentences {
                    self.synthesize_and_push(&sentence, processor).await?;
                }

                // Forward the LLMText frame (downstream aggregators may need it)
                processor.push_frame(frame, direction).await?;
            }

            // ---- LLM done: flush remaining buffer ----
            FrameInner::Control(ControlFrame::LLMFullResponseEnd) => {
                let remaining = {
                    let mut state = self.state.lock().await;
                    let tail = state.text_buffer.trim().to_string();
                    state.text_buffer.clear();
                    tail
                };

                if !remaining.is_empty() {
                    self.synthesize_and_push(&remaining, processor).await?;
                }

                processor.push_frame(frame, direction).await?;
            }

            // ---- Speaking state tracking ----
            FrameInner::System(SystemFrame::BotStartedSpeaking) => {
                self.state.lock().await.bot_speaking = true;
                processor.push_frame(frame, direction).await?;
            }

            FrameInner::System(SystemFrame::BotStoppedSpeaking) => {
                self.state.lock().await.bot_speaking = false;
                processor.push_frame(frame, direction).await?;
            }

            // ---- Interruption: clear buffer, no reconnect needed ----
            FrameInner::System(SystemFrame::Interruption) => {
                {
                    let mut state = self.state.lock().await;
                    state.text_buffer.clear();
                    // No reconnect needed — local inference, no connection.
                    // In-flight spawn_blocking completes but output frames are
                    // drained by pipeline interruption (drain_keep_uninterruptible).
                }
                processor.push_frame(frame, direction).await?;
            }

            // ---- Shutdown ----
            FrameInner::Control(ControlFrame::End { .. })
            | FrameInner::System(SystemFrame::Cancel { .. }) => {
                processor.push_frame(frame, direction).await?;
            }

            // ---- Everything else: passthrough ----
            _ => {
                processor.push_frame(frame, direction).await?;
            }
        }

        Ok(())
    }
}