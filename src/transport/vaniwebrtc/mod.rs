//! Peer-to-peer WebRTC transport (`vaniwebrtc`).
//!
//! Carries audio over a real P2P WebRTC connection (Opus/RTP/SRTP) with no
//! SFU/media server in the path. Signaling (SDP offer/answer + trickle ICE) and
//! control messages reuse a WebSocket + a WebRTC data channel respectively.
//!
//! See [`VaniWebRTCTransport`] for the entry point and [`VaniWebRTCParams`] for
//! configuration. The Opus codec is provided by `audiopus` (libopus) and a
//! 48 kHz [`Denoiser48k`] hook is reserved for a future DeepFilterNet stage.

pub mod codec;
pub mod params;
pub mod signaling;
pub mod transport;

pub use codec::Denoiser48k;
pub use params::{build_shared_udp_mux, DenoiserFactory, TurnServer, VaniWebRTCParams};
pub use signaling::SignalMsg;
pub use transport::VaniWebRTCTransport;
