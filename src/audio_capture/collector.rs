use std::sync::Arc;

use tokio::sync::mpsc;
use uuid::Uuid;

use super::segment::PendingAudioSegment;
use super::storage::{AudioStorage, RecordedSegment};

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
// Drain task — collects all segments then writes one mixed recording
// ---------------------------------------------------------------------------

async fn drain_task(
    session_id: Uuid,
    mut rx: mpsc::Receiver<PendingAudioSegment>,
    storage: Arc<dyn AudioStorage>,
) {
    let mut segments: Vec<RecordedSegment> = Vec::new();

    while let Some(seg) = rx.recv().await {
        if seg.pcm.is_empty() {
            continue;
        }
        segments.push(RecordedSegment {
            speaker:      seg.speaker,
            pcm:          seg.pcm,
            sample_rate:  seg.sample_rate,
            num_channels: seg.num_channels,
            started_at:   seg.started_at,
        });
    }

    if let Err(e) = storage.finalize_recording(session_id, &segments).await {
        log::error!("AudioCapture: finalize_recording failed: {e}");
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
