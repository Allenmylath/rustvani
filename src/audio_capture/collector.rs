use std::sync::Arc;

use chrono::Utc;
use tokio::sync::mpsc;
use uuid::Uuid;

use super::encoder::{encode_pcm_to_wav, pcm_duration_ms};
use super::segment::{AudioSegmentMeta, PendingAudioSegment};
use super::storage::AudioStorage;

// ---------------------------------------------------------------------------
// Public trait
// ---------------------------------------------------------------------------

pub trait AudioCaptureCollector: Send + Sync {
    /// Enqueue a completed audio segment for encoding and storage.
    /// Sync and non-blocking — drops if the channel is full.
    fn record_segment(&self, segment: PendingAudioSegment);
    fn session_id(&self) -> Uuid;
}

// ---------------------------------------------------------------------------
// SessionAudioCapture — real implementation
// ---------------------------------------------------------------------------

pub struct SessionAudioCapture {
    session_id: Uuid,
    tx: mpsc::Sender<PendingAudioSegment>,
}

impl SessionAudioCapture {
    /// `channel_capacity` — how many pending segments can be buffered.
    /// 64 is a safe default (each segment is ~160KB for a 5s mono 16kHz turn).
    pub fn new(
        session_id: Uuid,
        storage: Arc<dyn AudioStorage>,
        channel_capacity: usize,
    ) -> (Arc<Self>, tokio::task::JoinHandle<()>) {
        let (tx, rx) = mpsc::channel(channel_capacity);
        let collector = Arc::new(Self { session_id, tx });
        let handle = tokio::spawn(drain_task(session_id, rx, storage));
        (collector, handle)
    }
}

impl AudioCaptureCollector for SessionAudioCapture {
    fn record_segment(&self, segment: PendingAudioSegment) {
        if let Err(e) = self.tx.try_send(segment) {
            match e {
                mpsc::error::TrySendError::Full(_) => {
                    log::warn!("AudioCaptureCollector: channel full, dropping segment");
                }
                mpsc::error::TrySendError::Closed(_) => {}
            }
        }
    }

    fn session_id(&self) -> Uuid {
        self.session_id
    }
}

// ---------------------------------------------------------------------------
// Drain task — encodes PCM and writes to storage off the hot path
// ---------------------------------------------------------------------------

async fn drain_task(
    session_id: Uuid,
    mut rx: mpsc::Receiver<PendingAudioSegment>,
    storage: Arc<dyn AudioStorage>,
) {
    while let Some(seg) = rx.recv().await {
        process_segment(session_id, seg, &*storage).await;
    }
}

async fn process_segment(
    session_id: Uuid,
    seg: PendingAudioSegment,
    storage: &dyn AudioStorage,
) {
    if seg.pcm.is_empty() {
        return;
    }

    let ended_at = Utc::now();
    let duration_ms = pcm_duration_ms(seg.pcm.len(), seg.sample_rate, seg.num_channels);
    let byte_size_before_encode = seg.pcm.len() as u64;

    // Encode in a blocking thread so we don't block the tokio runtime.
    let pcm = seg.pcm;
    let sr = seg.sample_rate;
    let ch = seg.num_channels;
    let wav_bytes = match tokio::task::spawn_blocking(move || encode_pcm_to_wav(&pcm, sr, ch)).await {
        Ok(Ok(bytes)) => bytes,
        Ok(Err(e)) => {
            log::error!("AudioCapture: encode failed: {e}");
            return;
        }
        Err(e) => {
            log::error!("AudioCapture: encode task panicked: {e}");
            return;
        }
    };

    let wav_size = wav_bytes.len() as u64;

    let url = match storage.store_segment(session_id, seg.segment_id, seg.speaker, &wav_bytes).await {
        Ok(u) => u,
        Err(e) => {
            log::error!("AudioCapture: store_segment failed: {e}");
            return;
        }
    };

    let meta = AudioSegmentMeta {
        segment_id:   seg.segment_id,
        session_id,
        turn_id:      seg.turn_id,
        speaker:      seg.speaker.to_string(),
        audio_url:    url,
        format:       "wav".to_string(),
        sample_rate:  sr,
        num_channels: ch,
        duration_ms,
        byte_size:    wav_size,
        interrupted:  seg.interrupted,
        started_at:   seg.started_at,
        ended_at,
    };

    let _ = byte_size_before_encode; // acknowledged
    if let Err(e) = storage.save_metadata(session_id, &meta).await {
        log::error!("AudioCapture: save_metadata failed: {e}");
    }
}

// ---------------------------------------------------------------------------
// NoopAudioCaptureCollector — zero-cost default
// ---------------------------------------------------------------------------

pub struct NoopAudioCaptureCollector;

impl AudioCaptureCollector for NoopAudioCaptureCollector {
    fn record_segment(&self, _segment: PendingAudioSegment) {}
    fn session_id(&self) -> Uuid { Uuid::nil() }
}
