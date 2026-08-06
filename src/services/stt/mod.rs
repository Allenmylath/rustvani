//! Speech-to-text services.
//!
//! Every backend here speaks WebSocket to its provider, so each is gated by the
//! feature that pulls in `tokio-tungstenite`.
//!
//! [`core`] holds the reusable half: the [`SttProvider`] base trait, the
//! generic [`SttService`] that implements `FrameHandler` for all of them, the
//! [`TurnGate`], and the [`AudioFrontend`]. A provider module contributes only
//! its wire protocol and its own config struct.

#[cfg(any(
    feature = "stt-deepgram",
    feature = "stt-gnani",
    feature = "stt-sarvam",
    feature = "stt-60db",
))]
pub mod core;

#[cfg(any(
    feature = "stt-deepgram",
    feature = "stt-gnani",
    feature = "stt-sarvam",
    feature = "stt-60db",
))]
pub use core::{
    AudioFrontend, AudioSpec, Handshake, InterimPolicy, NoiseBackend, Outgoing, SttCoreConfig,
    SttEvent, SttProvider, SttService, TurnGate, WsMessage,
};

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
// `NoiseBackend` now lives in `core::frontend` (it applies to every provider,
// not just Sarvam). `sarvam` re-exports it so existing paths keep resolving.
#[cfg(feature = "stt-sarvam")]
pub use sarvam::{SarvamSttConfig, SarvamSttHandler};
#[cfg(feature = "stt-60db")]
pub use sixtydb::{
    SixtyDbEncoding,
    SixtyDbSttConfig, SixtyDbSttHandler,
};
