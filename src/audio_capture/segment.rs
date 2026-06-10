use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// In-memory audio accumulated during one speaking turn, sent to the drain task for encoding.
#[derive(Debug)]
pub struct PendingAudioSegment {
    pub segment_id:   Uuid,
    /// Matches `TranscriptEntry.turn_id` for the same turn — used to link audio to transcript.
    pub turn_id:      Option<Uuid>,
    /// "user" | "bot"
    pub speaker:      &'static str,
    /// Raw PCM bytes (16-bit little-endian signed samples).
    pub pcm:          Vec<u8>,
    pub sample_rate:  u32,
    pub num_channels: u16,
    pub started_at:   DateTime<Utc>,
    /// `true` when the bot was interrupted before the turn finished naturally.
    pub interrupted:  bool,
}

/// Metadata row written to `session_audio_segments` after the audio file is stored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSegmentMeta {
    pub segment_id:   Uuid,
    pub session_id:   Uuid,
    pub turn_id:      Option<Uuid>,
    pub speaker:      String,
    pub audio_url:    String,
    pub format:       String,
    pub sample_rate:  u32,
    pub num_channels: u16,
    pub duration_ms:  f64,
    pub byte_size:    u64,
    pub interrupted:  bool,
    pub started_at:   DateTime<Utc>,
    pub ended_at:     DateTime<Utc>,
}
