//! Transport configuration parameters.
//!
//! Mirrors Python's `TransportParams` with deprecated fields removed.

/// Configuration parameters for transport implementations.
///
/// Concrete transports (WebRTC, WebSocket, etc.) accept this struct
/// and pass it down to `BaseInputTransport` / `BaseOutputTransport`.
#[derive(Debug, Clone)]
pub struct TransportParams {
    // ---- Audio input ----

    /// Enable audio input streaming.
    pub audio_in_enabled: bool,

    /// Input sample rate in Hz.
    /// `None` means inherit from `StartFrame` (defaults to 16 000 Hz).
    pub audio_in_sample_rate: Option<u32>,

    /// Number of input audio channels.
    pub audio_in_channels: u16,

    /// Push `InputAudioRaw` frames downstream (passthrough to STT etc.).
    pub audio_in_passthrough: bool,

    /// Start reading audio immediately when the transport starts.
    /// Set to `false` if the concrete transport wants to control when
    /// audio streaming begins (e.g. wait for a participant to join).
    pub audio_in_stream_on_start: bool,

    // ---- Video input ----

    /// Enable video input streaming.
    pub video_in_enabled: bool,

    // ---- Audio output ----

    /// Enable audio output streaming.
    pub audio_out_enabled: bool,

    /// Output sample rate in Hz.
    /// `None` means inherit from `StartFrame` (defaults to 24 000 Hz).
    pub audio_out_sample_rate: Option<u32>,

    /// Number of output audio channels.
    pub audio_out_channels: u16,

    /// Output audio bitrate in bits per second.
    pub audio_out_bitrate: u32,

    // ---- VAD ----

    /// Enable Voice Activity Detection.
    ///
    /// When `true`, `BaseTransport` inserts a `VadProcessor` between
    /// `input()` and the rest of the pipeline automatically.
    /// Only effective when `audio_in_enabled` is also `true`.
    pub vad_enabled: bool,
}

impl Default for TransportParams {
    fn default() -> Self {
        Self {
            audio_in_enabled:         false,
            audio_in_sample_rate:     None,
            audio_in_channels:        1,
            audio_in_passthrough:     true,
            audio_in_stream_on_start: true,
            video_in_enabled:         false,
            audio_out_enabled:        false,
            audio_out_sample_rate:    None,
            audio_out_channels:       1,
            audio_out_bitrate:        96_000,
            video_out_enabled:        false,
            video_out_width:          1024,
            video_out_height:         768,
            video_out_framerate:      30,
            vad_enabled:              false,
        }
    }
}