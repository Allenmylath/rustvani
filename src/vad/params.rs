//! VAD configuration parameters.
//!
//! The defaults were measured on real speech (All India Radio Malayalam news, and browser-
//! microphone speech received over WebRTC/Opus), scoring how much of the speech reaches the STT,
//! how long VAD stays deaf, false turns on noise, and the STT's character error rate.
//!
//! `start_secs` matters most. A turn opens only after that much *unbroken* confident speech, and
//! after a browser's audio processing and Opus, Silero's confidence stays above 0.7 for runs of
//! only ~0.36 s (median). At the previous 0.4 s, turns often never opened: 38–69% of a speaker's
//! speech reached the STT, with deaf spells of up to 7 s. These defaults pass 99–100%, open no
//! turns on white noise down to -30 dBFS, and lowered Sarvam's character error rate on the news
//! from 3.9/2.6% to 3.0/1.3%. A 0.8 s stop ends turns at sentence pauses rather than every
//! breath (pipecat uses 0.2 s / 0.8 s too).

pub const VAD_CONFIDENCE: f32 = 0.5;
pub const VAD_START_SECS: f32 = 0.2;
pub const VAD_STOP_SECS:  f32 = 0.8;
pub const VAD_MIN_VOLUME: f32 = 0.2;

#[derive(Debug, Clone)]
pub struct VadParams {
    pub confidence: f32,
    pub start_secs: f32,
    pub stop_secs:  f32,
    pub min_volume: f32,
}

impl Default for VadParams {
    fn default() -> Self {
        Self {
            confidence: VAD_CONFIDENCE,
            start_secs: VAD_START_SECS,
            stop_secs:  VAD_STOP_SECS,
            min_volume: VAD_MIN_VOLUME,
        }
    }
}