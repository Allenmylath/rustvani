//! One-shot speech synthesis: a text in, its whole audio out.
//!
//! The streaming TTS handlers ([`SarvamTtsHandler`](super::sarvam::SarvamTtsHandler), Deepgram,
//! Piper) speak an LLM's answer as it is generated. [`TtsSynthesizer`] is the other shape: one
//! utterance, with a chosen voice and pace, returned complete, so the caller knows how long it is
//! before deciding when to play it. The [`DubbingService`](crate::services::dub::DubbingService)
//! is built on it.

use async_trait::async_trait;
use bytes::Bytes;
use tokio::sync::mpsc;

use crate::error::Result;

/// Synthesized audio: signed 16-bit little-endian PCM, interleaved when `channels > 1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pcm {
    pub samples: Bytes,
    pub rate: u32,
    pub channels: u16,
}

impl Pcm {
    pub fn frames(&self) -> usize {
        self.samples.len() / (2 * self.channels.max(1) as usize)
    }

    pub fn duration_secs(&self) -> f64 {
        self.frames() as f64 / self.rate.max(1) as f64
    }
}

/// An utterance's audio as it is synthesized: chunks of PCM (s16le, `channels` interleaved) at
/// `rate`, the first one sooner than the whole; the channel ends with the utterance, an `Err`
/// ends it early.
pub struct PcmChunks {
    pub rate: u32,
    pub channels: u16,
    pub chunks: mpsc::Receiver<Result<Bytes>>,
}

/// A text-to-speech backend that synthesizes one utterance at a time.
///
/// Language tags are BCP-47 as the caller names them (`hi`, `en-IN`); a provider maps them to its
/// own codes and says which it can speak through [`supports`](Self::supports).
#[async_trait]
pub trait TtsSynthesizer: Send + Sync + 'static {
    /// Provider name for billing and logs (`"sarvam"`).
    fn name(&self) -> &'static str;

    /// Model name for billing.
    fn model(&self) -> String;

    /// Whether `lang` can be spoken.
    fn supports(&self, lang: &str) -> bool;

    /// The pace range the provider accepts (1.0 is natural).
    fn pace_range(&self) -> (f32, f32) {
        (1.0, 1.0)
    }

    /// Speaks `text` in `lang` with `voice` (the provider's default when empty) at `pace`
    /// (clamped to [`pace_range`](Self::pace_range)), at `sample_rate` Hz if the provider can.
    async fn synthesize(&self, text: &str, lang: &str, voice: &str, pace: f32, sample_rate: u32) -> Result<Pcm>;

    /// Whether [`synthesize_streaming`](Self::synthesize_streaming) yields audio before the whole
    /// utterance is made (else it is [`synthesize`](Self::synthesize) in one chunk).
    fn streams(&self) -> bool {
        false
    }

    /// Like [`synthesize`](Self::synthesize), the audio arriving as it is made.
    async fn synthesize_streaming(&self, text: &str, lang: &str, voice: &str, pace: f32, sample_rate: u32) -> Result<PcmChunks> {
        let pcm = self.synthesize(text, lang, voice, pace, sample_rate).await?;
        let (tx, rx) = mpsc::channel(1);
        if !pcm.samples.is_empty() {
            let _ = tx.try_send(Ok(pcm.samples));
        }
        Ok(PcmChunks { rate: pcm.rate, channels: pcm.channels, chunks: rx })
    }
}
