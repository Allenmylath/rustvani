use std::path::PathBuf;

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::{PipecatError, Result};
use super::super::segment::AudioSegmentMeta;
use super::AudioStorage;

/// Stores audio segments as WAV files on the local filesystem.
///
/// Files are written to `{base_dir}/{session_id}/{segment_id}.wav`.
/// The returned URL is the absolute path as a string.
///
/// Use this for development and single-machine deployments.
/// For production, use an S3-compatible implementation instead.
pub struct LocalAudioStorage {
    base_dir: PathBuf,
}

impl LocalAudioStorage {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self { base_dir: base_dir.into() }
    }
}

#[async_trait]
impl AudioStorage for LocalAudioStorage {
    async fn store_segment(
        &self,
        session_id: Uuid,
        segment_id: Uuid,
        _speaker: &str,
        data: &[u8],
    ) -> Result<String> {
        let dir = self.base_dir.join(session_id.to_string());
        tokio::fs::create_dir_all(&dir).await
            .map_err(|e| PipecatError::pipeline(format!("audio create dir: {e}")))?;

        let path = dir.join(format!("{segment_id}.wav"));
        tokio::fs::write(&path, data).await
            .map_err(|e| PipecatError::pipeline(format!("audio write file: {e}")))?;

        Ok(path.to_string_lossy().into_owned())
    }

    async fn save_metadata(&self, _session_id: Uuid, meta: &AudioSegmentMeta) -> Result<()> {
        log::info!(
            "audio_segment: {}",
            serde_json::to_string(meta).unwrap_or_default()
        );
        Ok(())
    }
}
