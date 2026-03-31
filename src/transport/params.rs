//! Transport configuration parameters.
//!
//! Mirrors Python's `TransportParams` with deprecated fields removed.

/// Configuration parameters for transport implementations.
#[derive(Debug, Clone)]
pub struct TransportParams {
    pub audio_in_enabled:         bool,
    pub audio_in_sample_rate:     Option<u32>,
    pub audio_in_channels:        u16,
    pub audio_in_passthrough:     bool,
    pub audio_in_stream_on_start: bool,
    pub video_in_enabled:         bool,
    pub audio_out_enabled:        bool,
    pub audio_out_sample_rate:    Option<u32>,
    pub audio_out_channels:       u16,
    pub audio_out_bitrate:        u32,
    pub vad_enabled:              bool,
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
            vad_enabled:              false,
        }
    }
}