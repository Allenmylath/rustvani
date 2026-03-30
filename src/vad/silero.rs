//! Silero VAD model wrapper using tract-onnx.
//!
//! The `.onnx` model is embedded at compile time via `include_bytes!()` —
//! zero file I/O at runtime, single binary deployment.
//!
//! Inference runs on `tokio::task::spawn_blocking` — a dedicated OS thread
//! from tokio's blocking pool. The async executor is never blocked.
//!
//! # Model file
//!
//! Download before building:
//! ```sh
//! mkdir -p src/vad/data
//! curl -L https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx \
//!      -o src/vad/data/silero_vad.onnx
//! ```

use std::sync::{Arc, Mutex};
use std::time::Instant;

use tract_onnx::prelude::*;

// Embed model at compile time — fastest possible load, no file I/O.
static MODEL_BYTES: &[u8] = include_bytes!("data/silero_vad.onnx");

/// How often to reset model internal state.
/// Copied from Python: `_MODEL_RESET_STATES_TIME = 5.0`
const MODEL_RESET_SECS: f64 = 5.0;

// ---------------------------------------------------------------------------
// Type alias for the compiled model
// ---------------------------------------------------------------------------

type SileroModel = SimplePlan<
    TypedFact,
    Box<dyn TypedOp>,
    Graph<TypedFact, Box<dyn TypedOp>>,
>;

// ---------------------------------------------------------------------------
// SileroVadInner — owns model + mutable state, lives on the blocking thread
// ---------------------------------------------------------------------------

struct SileroVadInner {
    model:       SileroModel,
    /// Hidden state: shape [2, 1, 128]
    h_state:     Vec<f32>,
    /// Context buffer: shape [1, context_size] (64 @ 16kHz, 32 @ 8kHz)
    context:     Vec<f32>,
    context_size: usize,
    sample_rate: i64,
    last_reset:  Instant,
}

impl SileroVadInner {
    fn build(sample_rate: u32) -> TractResult<Self> {
        let num_samples: usize = if sample_rate == 16000 { 512 } else { 256 };
        let context_size: usize = if sample_rate == 16000 { 64 } else { 32 };

        let model = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(MODEL_BYTES))?
            // input: float32[1, num_samples]
            .with_input_fact(
                0,
                InferenceFact::dt_shape(
                    DatumType::F32,
                    &[1usize, num_samples],
                ),
            )?
            // state: float32[2, 1, 128]
            .with_input_fact(
                1,
                InferenceFact::dt_shape(DatumType::F32, &[2usize, 1, 128]),
            )?
            // sr: int64 scalar
            .with_input_fact(
                2,
                InferenceFact::dt_shape(DatumType::I64, &[] as &[usize]),
            )?
            .into_optimized()?
            .into_runnable()?;

        Ok(Self {
            model,
            h_state: vec![0.0f32; 2 * 1 * 128],
            context: vec![0.0f32; context_size],
            context_size,
            sample_rate: sample_rate as i64,
            last_reset: Instant::now(),
        })
    }

    /// Reset internal state — mirrors Python's `reset_states()`.
    fn reset_states(&mut self) {
        self.h_state.fill(0.0);
        self.context.fill(0.0);
    }

    /// Run inference on one window of normalised float32 audio.
    ///
    /// `audio`: float32 samples, length = num_frames_required (512 or 256).
    /// Returns voice confidence in [0, 1].
    fn infer_f32(&mut self, audio: &[f32]) -> TractResult<f32> {
        let num_samples = audio.len();
        let context_size = self.context_size;

        // Build input tensor: concat context + audio along time axis
        // shape: [1, context_size + num_samples]
        let full_len = context_size + num_samples;
        let mut full_input = Vec::with_capacity(full_len);
        full_input.extend_from_slice(&self.context);
        full_input.extend_from_slice(audio);

        let input_tensor: Tensor = tract_ndarray::Array2::from_shape_vec(
            (1, full_len),
            full_input.clone(),
        )?
        .into();

        // State tensor: [2, 1, 128]
        let state_tensor: Tensor = tract_ndarray::Array3::from_shape_vec(
            (2, 1, 128),
            self.h_state.clone(),
        )?
        .into();

        // SR tensor: scalar int64
        let sr_tensor: Tensor = tract_ndarray::arr0(self.sample_rate).into();

        // Run model
        let outputs = self.model.run(tvec![
            input_tensor.into(),
            state_tensor.into(),
            sr_tensor.into(),
        ])?;

        // Extract confidence: output[0] shape [1, 1]
        let confidence = outputs[0]
            .to_array_view::<f32>()?
            .iter()
            .next()
            .copied()
            .unwrap_or(0.0);

        // Update hidden state from output[1]: shape [2, 1, 128]
        let new_state = outputs[1].to_array_view::<f32>()?;
        self.h_state.copy_from_slice(new_state.as_slice().unwrap());

        // Update context: last context_size samples of full_input
        let ctx_start = full_input.len().saturating_sub(context_size);
        self.context.copy_from_slice(&full_input[ctx_start..]);

        // Periodic model reset — mirrors Python's 5-second timer
        if self.last_reset.elapsed().as_secs_f64() >= MODEL_RESET_SECS {
            self.reset_states();
            self.last_reset = Instant::now();
        }

        Ok(confidence)
    }

    /// Convert raw PCM bytes to float32 and run inference.
    ///
    /// `audio_bytes`: signed 16-bit LE PCM, mono.
    /// Returns voice confidence in [0, 1].
    pub fn infer(&mut self, audio_bytes: &[u8]) -> TractResult<f32> {
        // Normalise i16 → f32, divide by 32768 (exact Python behaviour)
        let audio_f32: Vec<f32> = audio_bytes
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect();

        self.infer_f32(&audio_f32)
    }
}

// ---------------------------------------------------------------------------
// SileroVad — public API, Arc<Mutex<Inner>> for spawn_blocking
// ---------------------------------------------------------------------------

/// Thread-safe Silero VAD model wrapper.
///
/// `infer_async()` offloads to `tokio::task::spawn_blocking` — the async
/// executor is never blocked, and the model gets a real OS thread.
#[derive(Clone)]
pub struct SileroVad {
    inner: Arc<Mutex<SileroVadInner>>,
}

impl SileroVad {
    /// Load and compile the model for the given sample rate.
    /// Must be 8000 or 16000.
    ///
    /// This is a blocking operation — call once at startup from a
    /// `spawn_blocking` context or during initialisation.
    pub fn new(sample_rate: u32) -> Result<Self, String> {
        if sample_rate != 8000 && sample_rate != 16000 {
            return Err(format!(
                "Silero VAD only supports 8000 or 16000 Hz, got {}",
                sample_rate
            ));
        }
        let inner = SileroVadInner::build(sample_rate)
            .map_err(|e| format!("Failed to load Silero model: {}", e))?;

        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    /// Run inference asynchronously on one PCM window.
    ///
    /// Offloads to `tokio::task::spawn_blocking` so the async executor
    /// is never blocked. True OS-thread parallelism.
    ///
    /// `audio_bytes`: signed 16-bit LE PCM, mono.
    /// Returns voice confidence in [0, 1].
    pub async fn infer_async(&self, audio_bytes: Vec<u8>) -> Result<f32, String> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut guard = inner.lock().unwrap();
            guard
                .infer(&audio_bytes)
                .map_err(|e| format!("Silero inference error: {}", e))
        })
        .await
        .map_err(|e| format!("spawn_blocking error: {}", e))?
    }
}