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

        let session_start = segments.iter().map(|s| s.started_at).min().unwrap();
        let owned: Vec<OwnedSeg> = segments
            .iter()
            .map(|s| OwnedSeg {
                pcm:          s.pcm.clone(),
                sample_rate:  s.sample_rate,
                num_channels: s.num_channels,
                offset_ms:    (s.started_at - session_start).num_milliseconds().max(0) as u64,
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
    offset_ms:    u64,
}

fn mix_timeline(segments: Vec<OwnedSeg>) -> Vec<u8> {
    // Determine total buffer size needed (in mono samples at TARGET_RATE).
    let mut total_samples = 0usize;
    for seg in &segments {
        if seg.pcm.is_empty() || seg.sample_rate == 0 { continue; }
        let offset = ms_to_samples(seg.offset_ms);
        let mono_len = seg.pcm.len() / seg.num_channels.max(1) as usize;
        let out_len = resampled_count(mono_len / 2, seg.sample_rate);
        total_samples = total_samples.max(offset + out_len);
    }
    if total_samples == 0 { return Vec::new(); }

    // Accumulate in i32 so overlapping turns don't clip during addition.
    let mut buf = vec![0i32; total_samples];

    for seg in segments {
        if seg.pcm.is_empty() || seg.sample_rate == 0 { continue; }
        let offset = ms_to_samples(seg.offset_ms);
        let mono  = downmix_to_mono(&seg.pcm, seg.num_channels);
        let resampled = resample_pcm(&mono, seg.sample_rate, TARGET_RATE);
        for (i, chunk) in resampled.chunks_exact(2).enumerate() {
            let pos = offset + i;
            if pos < buf.len() {
                buf[pos] += i16::from_le_bytes([chunk[0], chunk[1]]) as i32;
            }
        }
    }

    // Clamp and convert to little-endian i16 bytes.
    buf.iter()
        .flat_map(|&s| {
            (s.clamp(i16::MIN as i32, i16::MAX as i32) as i16).to_le_bytes()
        })
        .collect()
}

#[inline]
fn ms_to_samples(ms: u64) -> usize {
    (ms * TARGET_RATE as u64 / 1000) as usize
}

#[inline]
fn resampled_count(n_in: usize, from_rate: u32) -> usize {
    ((n_in as f64) * (TARGET_RATE as f64) / (from_rate as f64)).ceil() as usize
}
