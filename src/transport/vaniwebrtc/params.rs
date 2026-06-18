//! Configuration for the peer-to-peer `vaniwebrtc` transport.

use std::sync::Arc;

use crate::transport::TransportParams;

use super::codec::Denoiser48k;

/// Builds a fresh per-connection [`Denoiser48k`] (each call gets its own state).
///
/// A factory (rather than a shared instance) is used because a denoiser holds
/// per-connection state and must not be shared across peers.
pub type DenoiserFactory = Arc<dyn Fn() -> Box<dyn Denoiser48k> + Send + Sync>;

/// A TURN server with long-term credentials (browser-style auth).
///
/// Kept separate from [`VaniWebRTCParams::ice_servers`] because `webrtc-rs`
/// rejects a TURN url that has no credentials, whereas STUN urls must have none.
#[derive(Clone)]
pub struct TurnServer {
    pub urls:       Vec<String>,
    pub username:   String,
    pub credential: String,
}

/// Bind one UDP socket and wrap it in a shared mux.
///
/// Call this **once** at application startup, then clone the returned `Arc` into
/// every connection's [`VaniWebRTCParams::udp_mux`]. All peer connections then
/// share a single bound UDP port (so it can be exposed on platforms like Fly.io
/// that only forward inbound UDP to a known port). Rebuilding the mux per
/// connection would re-`bind` the same port and fail with "address in use".
///
/// `bind_addr` is anything `UdpSocket::bind` accepts, e.g.
/// `"fly-global-services:3478"` or `"0.0.0.0:3478"`.
pub async fn build_shared_udp_mux(
    bind_addr: &str,
) -> std::io::Result<Arc<dyn webrtc::ice::udp_mux::UDPMux + Send + Sync>> {
    let socket = tokio::net::UdpSocket::bind(bind_addr).await?;
    Ok(webrtc::ice::udp_mux::UDPMuxDefault::new(
        webrtc::ice::udp_mux::UDPMuxParams::new(socket),
    ))
}

/// Parameters for [`VaniWebRTCTransport`](super::transport::VaniWebRTCTransport).
#[derive(Clone)]
pub struct VaniWebRTCParams {
    /// Shared transport config (audio I/O rates, VAD, turn detection) — same
    /// role as `WebSocketParams.transport`.
    pub transport: TransportParams,

    /// ICE servers for NAT traversal, e.g. `["stun:stun.l.google.com:19302"]`.
    /// P2P only — no TURN/relay or SFU is required for LAN/STUN-reachable peers.
    pub ice_servers: Vec<String>,

    /// TURN servers with credentials (opt-in fallback for locked-down networks).
    /// Empty by default — STUN-only is enough for most clients once a reachable
    /// Host candidate is advertised via [`nat_1to1_ips`](Self::nat_1to1_ips).
    pub turn_servers: Vec<TurnServer>,

    /// Public IPv4(s) advertised as Host candidates (NAT 1:1). Set this to the
    /// platform's dedicated public IPv4 (e.g. Fly's) so an IPv4 browser has a
    /// reachable candidate to pair with. Empty by default (use gathered addrs).
    pub nat_1to1_ips: Vec<String>,

    /// Pre-built shared UDP mux so all media flows over one bound port. Build it
    /// **once** at startup with [`build_shared_udp_mux`] and clone the `Arc` into
    /// every connection's params. `None` (default) = ephemeral per-connection
    /// ports, which is fine for LAN/STUN but not forwardable behind Fly's edge.
    pub udp_mux: Option<Arc<dyn webrtc::ice::udp_mux::UDPMux + Send + Sync>>,

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
            turn_servers:         vec![],
            nat_1to1_ips:         vec![],
            udp_mux:              None,
            opus_max_avg_bitrate: 510_000,
            opus_fullband:        true,
            opus_dtx:             false,
            denoiser_factory:     None,
        }
    }
}
