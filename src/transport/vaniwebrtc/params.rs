//! Configuration for the peer-to-peer `vaniwebrtc` transport.

use std::sync::Arc;

use crate::transport::TransportParams;

use super::codec::Denoiser48k;

/// Builds a fresh per-connection [`Denoiser48k`] (each call gets its own state).
///
/// A factory (rather than a shared instance) is used because a denoiser holds
/// per-connection state and must not be shared across peers.
pub type DenoiserFactory = Arc<dyn Fn() -> Box<dyn Denoiser48k> + Send + Sync>;

/// Parameters for [`VaniWebRTCTransport`](super::transport::VaniWebRTCTransport).
#[derive(Clone)]
pub struct VaniWebRTCParams {
    /// Shared transport config (audio I/O rates, VAD, turn detection) — same
    /// role as `WebSocketParams.transport`.
    pub transport: TransportParams,

    /// ICE servers for NAT traversal, e.g. `["stun:stun.l.google.com:19302"]`.
    /// P2P only — no TURN/relay or SFU is required for LAN/STUN-reachable peers.
    pub ice_servers: Vec<String>,

    // ---- Opus tuning (answer-SDP `fmtp`) — protects the 48 kHz denoise stage ----
    /// `maxaveragebitrate` forced on the browser's Opus encoder. High values
    /// keep the full speech spectrum a full-band denoiser (DeepFilterNet) needs.
    pub opus_max_avg_bitrate: u32,
    /// When `true`, request full-band Opus (`maxplaybackrate=48000`).
    pub opus_fullband: bool,
    /// Opus discontinuous transmission. Off by default (steady frames in).
    pub opus_dtx: bool,

    /// Optional factory for a 48 kHz inbound denoiser (DeepFilterNet hook).
    /// `None` in v1 → transparent pass-through.
    pub denoiser_factory: Option<DenoiserFactory>,
}

impl Default for VaniWebRTCParams {
    fn default() -> Self {
        Self {
            transport: TransportParams {
                audio_in_enabled:         true,
                audio_in_sample_rate:     Some(16_000),
                audio_in_channels:        1,
                audio_in_passthrough:     true,
                audio_in_stream_on_start: true,
                audio_out_enabled:        true,
                ..TransportParams::default()
            },
            ice_servers:          vec!["stun:stun.l.google.com:19302".to_string()],
            opus_max_avg_bitrate: 510_000,
            opus_fullband:        true,
            opus_dtx:             false,
            denoiser_factory:     None,
        }
    }
}
