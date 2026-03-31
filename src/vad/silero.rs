//! Silero VAD model wrapper using ort (ONNX Runtime).
//!
//! Model loaded from file at startup — no compile-time embedding.
//! Default path: `src/vad/data/silero_vad.onnx` (relative to working dir).
//!
//! Inference runs on `tokio::task::spawn_blocking` — never blocks the executor.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use ndarray::{Array1, Array2, Array3};
use ort::session::{Session, builder::SessionBuilder};
use ort::value::Tensor;

/// How often to reset model internal state (seconds).
const MODEL_RESET_SECS: f64 = 5.0;

/// Default model path relative to the crate root.
pub const DEFAULT_MODEL_PATH: &str = "src/vad/data/silero_vad.onnx";

// ---------------------------------------------------------------------------
// SileroVadInner
// ---------------------------------------------------------------------------

struct SileroVadInner {
    session:      Session,
    h_state:      Vec<f32>,  // [2, 1, 128]
    context:      Vec<f32>,  // [1, context_size]
    context_size: usize,
    num_samples:  usize,
    sample_rate:  i64,
    last_reset:   Instant,
}

impl SileroVadInner {
    fn build(sample_rate: u32, model_path: &str) -> Result<Self, String> {
        let num_samples:  usize = if sample_rate == 16000 { 512 } else { 256 };
        let context_size: usize = if sample_rate == 16000 { 64  } else { 32  };

        let session = SessionBuilder::new()
            .map_err(|e| format!("SessionBuilder error: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to load model from {}: {}", model_path, e))?;

        Ok(Self {
            session,
            h_state:      vec![0.0f32; 2 * 1 * 128],
            context:      vec![0.0f32; context_size],
            context_size,
            num_samples,
            sample_rate:  sample_rate as i64,
            last_reset:   Instant::now(),
        })
    }

    fn reset_states(&mut self) {
        self.h_state.fill(0.0);
        self.context.fill(0.0);
    }

    pub fn infer(&mut self, audio_bytes: &[u8]) -> Result<f32, String> {
        // i16 LE PCM → f32 normalised
        let audio_f32: Vec<f32> = audio_bytes
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect();

        let context_size = self.context_size;
        let num_samples  = self.num_samples;

        // Input: concat context + audio → [1, context_size + num_samples]
        let full_len = context_size + num_samples;
        let mut full_input = Vec::with_capacity(full_len);
        full_input.extend_from_slice(&self.context);
        full_input.extend_from_slice(&audio_f32);

        let input_array = Array2::from_shape_vec((1, full_len), full_input.clone())
            .map_err(|e| format!("Input shape error: {}", e))?;

        let state_array = Array3::from_shape_vec((2, 1, 128), self.h_state.clone())
            .map_err(|e| format!("State shape error: {}", e))?;

        let sr_array = Array1::from_vec(vec![self.sample_rate]);

        let input_tensor = Tensor::from_array(input_array)
            .map_err(|e| format!("Input tensor error: {}", e))?;
        let state_tensor = Tensor::from_array(state_array)
            .map_err(|e| format!("State tensor error: {}", e))?;
        let sr_tensor = Tensor::from_array(sr_array)
            .map_err(|e| format!("SR tensor error: {}", e))?;

        let outputs = self.session
            .run(ort::inputs![input_tensor, state_tensor, sr_tensor]
)
            .map_err(|e| format!("Inference error: {}", e))?;

        // Extract outputs eagerly so SessionOutputs is dropped before reset_states()
        let confidence = {
            outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| format!("Confidence extract error: {}", e))?
                .iter()
                .next()
                .copied()
                .unwrap_or(0.0)
        };

        let new_state: Vec<f32> = {
            outputs[1]
                .try_extract_array::<f32>()
                .map_err(|e| format!("State extract error: {}", e))?
                .iter()
                .copied()
                .collect()
        };

        // Drop outputs here — releases borrow on self.session
        drop(outputs);

        self.h_state.copy_from_slice(&new_state);

        // Update context
        let ctx_start = full_input.len().saturating_sub(context_size);
        self.context.copy_from_slice(&full_input[ctx_start..]);

        // Periodic reset
        let needs_reset = self.last_reset.elapsed().as_secs_f64() >= MODEL_RESET_SECS;
        if needs_reset {
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
