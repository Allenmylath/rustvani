use async_trait::async_trait;

/// Trait for pluggable VAD backends.
///
/// Mirrors Python's `VADAnalyzer` abstract base class.
/// Implement this to swap in WebRTC VAD, energy-based detection, etc.
///
/// `SileroVadNative` is the default implementation.
#[async_trait]
pub trait VadAnalyzer: Send + Sync {
    /// Number of PCM frames required per inference call.
    /// 512 @ 16 kHz, 256 @ 8 kHz.
    fn num_frames_required(&self) -> usize;

    /// Sample rate the model runs at. Incoming audio is resampled to it
    /// ([`VadInput`](super::VadInput)), whatever rate it arrives at.
    ///
    /// The default infers it from Silero's window size (256 → 8 kHz,
    /// anything else → 16 kHz); backends with other rates override it.
    fn sample_rate(&self) -> u32 {
        if self.num_frames_required() == 256 { 8_000 } else { 16_000 }
    }

    /// Run inference on one window of PCM audio.
    ///
    /// `audio` is exactly `num_frames_required() * 2` bytes of i16 LE mono PCM.
    /// Returns confidence in range 0.0–1.0.
    async fn voice_confidence(&self, audio: Vec<u8>) -> f32;
}