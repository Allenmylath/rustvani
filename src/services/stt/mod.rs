//! Speech-to-text services.
//!
//! Every backend here speaks WebSocket to its provider, so each is gated by the
//! feature that pulls in `tokio-tungstenite`.

#[cfg(feature = "stt-deepgram")]
pub mod deepgram;
#[cfg(feature = "stt-gnani")]
pub mod gnani;
#[cfg(feature = "stt-sarvam")]
pub mod sarvam;
#[cfg(feature = "stt-60db")]
pub mod sixtydb;

#[cfg(feature = "stt-deepgram")]
pub use deepgram::{DeepgramSttConfig, DeepgramSttHandler};
#[cfg(feature = "stt-gnani")]
pub use gnani::{GnaniSttConfig, GnaniSttHandler};
#[cfg(feature = "stt-sarvam")]
pub use sarvam::{NoiseBackend, SarvamSttConfig, SarvamSttHandler};
#[cfg(feature = "stt-60db")]
pub use sixtydb::{
    SixtyDbEncoding,
    SixtyDbSttConfig, SixtyDbSttHandler,
};
