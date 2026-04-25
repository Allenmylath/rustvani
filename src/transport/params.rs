//! Transport configuration parameters.

use std::sync::Arc;
use crate::vad::{VadAnalyzer, VadParams};

#[derive(Clone)]
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
    /// VAD backend. `None` disables VAD entirely.
    pub vad_analyzer:             Option<Arc<dyn VadAnalyzer>>,
    /// VAD tuning parameters. Only used when `vad_analyzer` is `Some`.
    pub vad_params:               VadParams,
    pub audio_out_10ms_chunks:     u32,
}

impl std::fmt::Debug for TransportParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransportParams")
            .field("audio_in_enabled",         &self.audio_in_enabled)
            .field("audio_in_sample_rate",      &self.audio_in_sample_rate)
            .field("audio_in_channels",         &self.audio_in_channels)
            .field("audio_in_passthrough",      &self.audio_in_passthrough)
            .field("audio_in_stream_on_start",  &self.audio_in_stream_on_start)
            .field("video_in_enabled",          &self.video_in_enabled)
            .field("audio_out_enabled",         &self.audio_out_enabled)
            .field("audio_out_sample_rate",     &self.audio_out_sample_rate)
            .field("audio_out_channels",        &self.audio_out_channels)
            .field("audio_out_bitrate",         &self.audio_out_bitrate)
            .field("vad_analyzer",              &self.vad_analyzer.as_ref().map(|_| "Some(...)"))
            .field("vad_params",                &self.vad_params)
            .field("audio_out_10ms_chunks", &self.audio_out_10ms_chunks)
            .finish()
    }
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
            vad_analyzer:             None,
            vad_params:               VadParams::default(),
            audio_out_10ms_chunks:    4,
        }
    }
}
