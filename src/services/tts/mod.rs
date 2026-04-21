pub mod piper;
pub mod sarvam;

pub use piper::{PiperModel, PiperQuality, PiperTtsConfig, PiperTtsHandler};
pub use sarvam::{SarvamTtsConfig, SarvamTtsHandler, TtsModelConfig, get_model_config};
