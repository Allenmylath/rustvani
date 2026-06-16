use std::path::PathBuf;

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::{PipecatError, Result};
use super::super::encoder::{downmix_to_mono, encode_pcm_to_wav, resample_pcm};
use super::super::segment::AudioSegmentMeta;
use super::{AudioStorage, RecordedSegment};

/// Produces a single `recording.wav` per session on the local filesystem.
///
/// All speaking turns (user and bot) are placed at their correct wall-clock
/// positions and mixed into one mono WAV at 16 kHz — the same way you would
/// hear the conversation if you were in the room.
///
/// File path: `{base_dir}/{session_id}/recording.wav`
pub struct LocalAudioStorage {
    base_dir: PathBuf,
}

impl LocalAudioStorage {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self { base_dir: base_dir.into() }
    }
}

const TARGET_RATE: u32 = 16_000;

#[async_trait]
impl AudioStorage for LocalAudioStorage {
    async fn store_segment(
        &self,
        _session_id: Uuid,
        _segment_id: Uuid,
        _speaker: &str,
        _data: &[u8],
    ) -> Result<String> {
        Ok(String::new())
    }

    async fn save_metadata(&self, _session_id: Uuid, _meta: &AudioSegmentMeta) -> Result<()> {
        Ok(())
    }

    async fn finalize_recording(
        &self,
        session_id: Uuid,
        segments: &[RecordedSegment],
    ) -> Result<()> {
        if segments.is_empty() {
            return Ok(());
        }

        let owned: Vec<OwnedSeg> = segments
            .iter()
            .map(|s| OwnedSeg {
                pcm:          s.pcm.clone(),
                sample_rate:  s.sample_rate,
                num_channels: s.num_channels,
            })
            .collect();

        let mixed_pcm = tokio::task::spawn_blocking(move || mix_timeline(owned))
            .await
            .map_err(|e| PipecatError::pipeline(format!("audio mix join: {e}")))?;

        if mixed_pcm.is_empty() {
            return Ok(());
        }

        let dir = self.base_dir.join(session_id.to_string());
        tokio::fs::create_dir_all(&dir).await
            .map_err(|e| PipecatError::pipeline(format!("audio create dir: {e}")))?;

        let wav = tokio::task::spawn_blocking(move || encode_pcm_to_wav(&mixed_pcm, TARGET_RATE, 1))
            .await
            .map_err(|e| PipecatError::pipeline(format!("audio encode join: {e}")))?
            .map_err(|e| PipecatError::pipeline(format!("audio encode: {e}")))?;

        let path = dir.join("recording.wav");
        tokio::fs::write(&path, &wav).await
            .map_err(|e| PipecatError::pipeline(format!("audio write recording.wav: {e}")))?;

        log::info!(
            "AudioCapture: session {} → recording.wav ({} bytes)",
            session_id, wav.len()
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CPU-bound mixing — runs in spawn_blocking
// ---------------------------------------------------------------------------

struct OwnedSeg {
    pcm:          Vec<u8>,
    sample_rate:  u32,
    num_channels: u16,
}

/// Overlay (mix) all tracks on a shared timeline starting at index 0.
///
/// `AudioCaptureProcessor` emits the session as two equal-length,
/// time-synchronised tracks (user and bot) where each is silent while the other
/// speaks. Summing them sample-for-sample reconstructs the conversation as one
/// mono stream — voices only overlap where a real barge-in occurred. Tracks of
/// unequal length are handled by sizing the buffer to the longest.
fn mix_timeline(segments: Vec<OwnedSeg>) -> Vec<u8> {
    let tracks: Vec<Vec<u8>> = segments
        .into_iter()
        .filter(|s| !s.pcm.is_empty() && s.sample_rate != 0)
        .map(|s| {
            let mono = downmix_to_mono(&s.pcm, s.num_channels);
            resample_pcm(&mono, s.sample_rate, TARGET_RATE)
        })
        .filter(|t| !t.is_empty())
        .collect();

    let total_samples = tracks.iter().map(|t| t.len() / 2).max().unwrap_or(0);
    if total_samples == 0 {
        return Vec::new();
    }

    // Accumulate in i32 so overlapping turns don't clip during addition.
    let mut buf = vec![0i32; total_samples];
    for track in &tracks {
        for (i, chunk) in track.chunks_exact(2).enumerate() {
            buf[i] += i16::from_le_bytes([chunk[0], chunk[1]]) as i32;
        }
    }

    buf.iter()
        .flat_map(|&s| (s.clamp(i16::MIN as i32, i16::MAX as i32) as i16).to_le_bytes())
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` mono samples of constant value, as little-endian i16 bytes @ 16 kHz.
    fn seg(value: i16, n: usize) -> OwnedSeg {
        let pcm: Vec<u8> = std::iter::repeat(value.to_le_bytes())
            .take(n)
            .flatten()
            .collect();
        OwnedSeg { pcm, sample_rate: TARGET_RATE, num_channels: 1 }
    }

    /// A mono @ 16 kHz track from explicit samples.
    fn seg_from(samples: &[i16]) -> OwnedSeg {
        let pcm: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        OwnedSeg { pcm, sample_rate: TARGET_RATE, num_channels: 1 }
    }

    #[test]
    fn mix_overlays_tracks_sample_for_sample() {
        // Two equal-length tracks: one silent where the other speaks (the
        // normal user/bot case) → the sum reproduces each without clipping.
        let mut user = vec![1000i16; 100];
        user.extend(std::iter::repeat(0).take(100)); // silent while bot speaks
        let mut bot = vec![0i16; 100];               // silent while user speaks
        bot.extend(std::iter::repeat(-1000).take(100));

        let out = mix_timeline(vec![seg_from(&user), seg_from(&bot)]);
        let samples: Vec<i16> = out
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        assert_eq!(samples.len(), 200, "overlay length = longest track");
        assert!(samples[..100].iter().all(|&s| s == 1000), "user region");
        assert!(samples[100..].iter().all(|&s| s == -1000), "bot region");
    }

    #[test]
    fn mix_sums_overlapping_audio_with_clamp() {
        let a = vec![20000i16; 10];
        let b = vec![20000i16; 10];
        let out = mix_timeline(vec![seg_from(&a), seg_from(&b)]);
        let samples: Vec<i16> = out
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        // 40000 clamps to i16::MAX.
        assert!(samples.iter().all(|&s| s == i16::MAX));
    }

    #[test]
    fn mix_handles_unequal_lengths() {
        let out = mix_timeline(vec![seg(500, 50), seg(500, 30)]);
        assert_eq!(out.len() / 2, 50, "sized to the longest track");
    }

    #[test]
    fn mix_empty_is_empty() {
        assert!(mix_timeline(vec![]).is_empty());
    }
}
