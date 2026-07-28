//! Text-to-speech services.
//!
//! Deepgram and Sarvam stream over WebSocket and are gated by the features that
//! pull in `tokio-tungstenite`; Piper runs locally and is gated by the feature
//! that pulls in `ort`.

#[cfg(feature = "tts-deepgram")]
pub mod deepgram;
#[cfg(feature = "tts-piper")]
pub mod piper;
#[cfg(feature = "tts-sarvam")]
pub mod sarvam;

#[cfg(feature = "tts-deepgram")]
pub use deepgram::{DeepgramEncoding, DeepgramTtsConfig, DeepgramTtsHandler};
#[cfg(feature = "tts-piper")]
pub use piper::{PiperModel, PiperQuality, PiperTtsConfig, PiperTtsHandler};
#[cfg(feature = "tts-sarvam")]
pub use sarvam::{SarvamTtsConfig, SarvamTtsHandler, TtsModelConfig, get_model_config};
