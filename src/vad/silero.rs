//! Silero VAD model wrapper using ort (ONNX Runtime).
//!
//! Model loaded from file at startup — no compile-time embedding.
//! Default path: `silero.onnx` (relative to working dir).
//!
//! Inference runs on `tokio::task::spawn_blocking` — never blocks the executor.
//!
//! Actual model API:
//!   Inputs (positional):
//!     [0] input:  [1, context_size + num_samples] f32
//!     [1] state:  [2, 1, 128] f32
//!     [2] sr:     [1] i64
//!   Outputs:
//!     "output"  — confidence f32
//!     "stateN"  — updated state [2, 1, 128] f32
//!
//! Context size: 64 samples @ 16kHz, 32 samples @ 8kHz.
//! State shape: [2, 1, 128].

use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ort::session::{Session, builder::SessionBuilder};
use ort::value::Value;

use super::analyzer::VadAnalyzer;

pub const DEFAULT_MODEL_PATH: &str = "silero.onnx";

// ---------------------------------------------------------------------------
// SileroVadInner
// ---------------------------------------------------------------------------

struct SileroVadInner {
    session:      Session,
    state:        ArrayD<f32>,
    context:      Array1<f32>,
    context_size: usize,
    num_samples:  usize,
    sample_rate:  i64,
}

impl SileroVadInner {
    fn build(sample_rate: u32, model_path: &str) -> Result<Self, String> {
        let session = SessionBuilder::new()
            .map_err(|e| format!("SessionBuilder error: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to load model from {}: {}", model_path, e))?;

        let sr = sample_rate as i64;
        let num_samples   = if sr == 16000 { 512 } else { 256 };
        let context_size  = if sr == 16000 { 64  } else { 32  };

        let state   = ArrayD::<f32>::zeros(IxDyn(&[2, 1, 128]));
        let context = Array1::<f32>::zeros(context_size);

        log::info!(
            "SileroVad: loaded model (sr={}, num_samples={}, context_size={})",
            sr, num_samples, context_size
        );

        Ok(Self { session, state, context, context_size, num_samples, sample_rate: sr })
    }

    pub fn infer(&mut self, audio_bytes: &[u8]) -> Result<f32, String> {
        let expected_bytes = self.num_samples * 2;
        if audio_bytes.len() != expected_bytes {
            return Err(format!(
                "Audio length mismatch: expected {} bytes, got {}",
                expected_bytes, audio_bytes.len()
            ));
        }

        // i16 LE → f32 normalised
        let audio_f32: Vec<f32> = audio_bytes
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect();

        // Prepend context
        let mut input_with_context = Vec::with_capacity(self.context_size + audio_f32.len());
        input_with_context.extend_from_slice(self.context.as_slice().unwrap());
        input_with_context.extend_from_slice(&audio_f32);

        let frame_len = input_with_context.len();
        let frame = Array2::<f32>::from_shape_vec([1, frame_len], input_with_context)
            .map_err(|e| format!("Frame shape error: {}", e))?;

        let frame_val = Value::from_array(frame)
            .map_err(|e| format!("Frame tensor error: {}", e))?;
        let state_val = Value::from_array(self.state.clone())
            .map_err(|e| format!("State tensor error: {}", e))?;
        let sr_val = Value::from_array(ndarray::array![self.sample_rate])
            .map_err(|e| format!("SR tensor error: {}", e))?;

        // Positional inputs — NOT named
        let outputs = self.session
            .run([
                (&frame_val).into(),
                (&state_val).into(),
                (&sr_val).into(),
            ])
            .map_err(|e| format!("Inference error: {}", e))?;

        // Update state
        let (shape, state_data) = outputs["stateN"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("stateN extract error: {}", e))?;
        let shape_usize: Vec<usize> = shape.as_ref().iter().map(|&d| d as usize).collect();
        self.state = ArrayD::from_shape_vec(
            IxDyn(shape_usize.as_slice()),
            state_data.to_vec(),
        )
        .map_err(|e| format!("State reshape error: {}", e))?;

        // Extract confidence
        let confidence = *outputs["output"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Output extract error: {}", e))?
            .1
            .first()
            .ok_or_else(|| "Empty output tensor".to_string())?;

        // Update context
        if audio_f32.len() >= self.context_size {
            self.context = Array1::from_vec(
                audio_f32[audio_f32.len() - self.context_size..].to_vec(),
            );
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
        Ok(Self { inner: Arc::new(Mutex::new(SileroVadInner::build(sample_rate, path)?)) })
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

// ---------------------------------------------------------------------------
// VadAnalyzer impl
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
impl VadAnalyzer for SileroVad {
    fn num_frames_required(&self) -> usize {
        self.inner.lock().unwrap().num_samples
    }

    async fn voice_confidence(&self, audio: Vec<u8>) -> f32 {
        match self.infer_async(audio).await {
            Ok(c) => c,
            Err(e) => {
                log::error!("SileroVad: inference error: {}", e);
                0.0
            }
        }
    }
}