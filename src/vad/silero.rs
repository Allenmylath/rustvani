//! Silero VAD model wrapper using ort (ONNX Runtime).
//!
//! Model loaded from file at startup — no compile-time embedding.
//! Default path: `silero_vad.onnx` (relative to working dir).
//!
//! Inference runs on `tokio::task::spawn_blocking` — never blocks the executor.
//!
//! Model API (verified via introspection):
//!   Inputs:  "input" [1, num_samples], "sr" [1], "h" [2,1,128], "c" [2,1,128]
//!   Outputs: "output" (confidence f32), "hn" [2,1,128], "cn" [2,1,128]

use std::sync::{Arc, Mutex};
use std::time::Instant;

use ort::session::{Session, builder::SessionBuilder};
use ort::value::Value;

/// How often to reset model internal state (seconds).
const MODEL_RESET_SECS: f64 = 5.0;

/// Default model path relative to the working directory.
pub const DEFAULT_MODEL_PATH: &str = "silero_vad.onnx";

// ---------------------------------------------------------------------------
// SileroVadInner
// ---------------------------------------------------------------------------

struct SileroVadInner {
    session:     Session,
    /// LSTM hidden state — shape [2, 1, 64] = 128 f32 elements.
    /// Model input name: "h", output name: "hn".
    h:           Vec<f32>,
    /// LSTM cell state — shape [2, 1, 64] = 128 f32 elements.
    /// Model input name: "c", output name: "cn".
    c:           Vec<f32>,
    num_samples: usize,
    sample_rate: i64,
    last_reset:  Instant,
}

impl SileroVadInner {
    fn build(sample_rate: u32, model_path: &str) -> Result<Self, String> {
        // Silero ONNX expects exactly 512 samples at 16kHz, 256 at 8kHz.
        let num_samples: usize = if sample_rate == 16000 { 512 } else { 256 };

        let session = SessionBuilder::new()
            .map_err(|e| format!("SessionBuilder error: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to load model from {}: {}", model_path, e))?;

        Ok(Self {
            session,
            h:           vec![0.0f32; 2 * 1 * 64],
            c:           vec![0.0f32; 2 * 1 * 64],
            num_samples,
            sample_rate: sample_rate as i64,
            last_reset:  Instant::now(),
        })
    }

    fn reset_states(&mut self) {
        self.h.fill(0.0);
        self.c.fill(0.0);
    }

    pub fn infer(&mut self, audio_bytes: &[u8]) -> Result<f32, String> {
        // i16 LE PCM → f32 normalised — shape [1, num_samples]
        let audio_f32: Vec<f32> = audio_bytes
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect();

        // Build ORT input values using tuple (shape, data) API.
        let audio_val = Value::from_array(([1usize, self.num_samples], audio_f32))
            .map_err(|e| format!("Input tensor error: {}", e))?;

        let sr_val = Value::from_array(([1usize], vec![self.sample_rate]))
            .map_err(|e| format!("SR tensor error: {}", e))?;

        let h_val = Value::from_array(([2usize, 1usize, 64usize], self.h.clone()))
            .map_err(|e| format!("h tensor error: {}", e))?;

        let c_val = Value::from_array(([2usize, 1usize, 64usize], self.c.clone()))
            .map_err(|e| format!("c tensor error: {}", e))?;

        // Named inputs — exact names verified via model introspection.
        let outputs = self.session
            .run(ort::inputs![
                "input" => audio_val,
                "sr"    => sr_val,
                "h"     => h_val,
                "c"     => c_val
            ])
            .map_err(|e| format!("Inference error: {}", e))?;

        // "output" — confidence scalar
        let confidence = {
            outputs["output"]
                .try_extract_array::<f32>()
                .map_err(|e| format!("Confidence extract error: {}", e))?
                .iter()
                .next()
                .copied()
                .unwrap_or(0.0)
        };

        // "hn" — updated hidden state
        let new_h: Vec<f32> = {
            outputs["hn"]
                .try_extract_array::<f32>()
                .map_err(|e| format!("hn extract error: {}", e))?
                .iter()
                .copied()
                .collect()
        };

        // "cn" — updated cell state
        let new_c: Vec<f32> = {
            outputs["cn"]
                .try_extract_array::<f32>()
                .map_err(|e| format!("cn extract error: {}", e))?
                .iter()
                .copied()
                .collect()
        };

        // Drop outputs — releases borrow on self.session.
        drop(outputs);

        // Persist LSTM states for temporal continuity across calls.
        self.h.copy_from_slice(&new_h);
        self.c.copy_from_slice(&new_c);

        // Periodic state reset — mirrors Python's MODEL_RESET_SECS logic.
        if self.last_reset.elapsed().as_secs_f64() >= MODEL_RESET_SECS {
            self.reset_states();
            self.last_reset = Instant::now();
        }

        Ok(confidence)
    }
}

// ---------------------------------------------------------------------------
// SileroVad — public API
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct SileroVad {
    inner: Arc<Mutex<SileroVadInner>>,
}

impl SileroVad {
    pub fn new(sample_rate: u32) -> Result<Self, String> {
        Self::from_path(sample_rate, DEFAULT_MODEL_PATH)
    }

    pub fn from_path(sample_rate: u32, path: &str) -> Result<Self, String> {
        if sample_rate != 8000 && sample_rate != 16000 {
            return Err(format!(
                "Silero VAD only supports 8000 or 16000 Hz, got {}",
                sample_rate
            ));
        }
        let inner = SileroVadInner::build(sample_rate, path)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    /// Run inference asynchronously on a blocking thread.
    ///
    /// `audio_bytes` must be exactly `num_samples * 2` bytes of i16 LE PCM.
    pub async fn infer_async(&self, audio_bytes: Vec<u8>) -> Result<f32, String> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut guard = inner.lock().unwrap();
            guard.infer(&audio_bytes)
        })
        .await
        .map_err(|e| format!("spawn_blocking error: {}", e))?
    }
}