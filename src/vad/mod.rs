pub mod analyzer;
pub mod params;
pub mod silero_ort;
pub mod silero_native;
pub mod state;

pub use analyzer::VadAnalyzer;
pub use params::{VadParams, VAD_CONFIDENCE, VAD_MIN_VOLUME, VAD_START_SECS, VAD_STOP_SECS};
pub use silero_ort::SileroVadOrt;
pub use silero_native::SileroVadNative;
pub use state::{StateMachine, VadState, calculate_audio_volume, exp_smoothing};

use std::sync::Arc;

/// Which VAD backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadBackend {
    /// Pure Rust engine — zero deps, ~5 MB memory, 16kHz only.
    Native,
    /// ONNX Runtime — supports 8kHz + 16kHz, larger footprint.
    Ort,
}

impl Default for VadBackend {
    fn default() -> Self { Self::Native }
}

/// Create a VAD analyzer from a backend choice.
pub fn create_vad(
    backend: VadBackend,
    sample_rate: u32,
) -> Result<Arc<dyn VadAnalyzer>, String> {
    match backend {
        VadBackend::Native => {
            let vad = SileroVadNative::new(sample_rate)?;
            Ok(Arc::new(vad))
        }
        VadBackend::Ort => {
            let vad = SileroVadOrt::new(sample_rate)?;
            Ok(Arc::new(vad))
        }
    }
}
