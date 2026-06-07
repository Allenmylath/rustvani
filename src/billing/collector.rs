use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;
use uuid::Uuid;

use super::events::{BillingEvent, SessionSummary, TranscriptEntry};
use super::storage::BillingStorage;

// ---------------------------------------------------------------------------
// Public trait
// ---------------------------------------------------------------------------

/// Non-blocking billing event sink injected into service handlers.
///
/// `record()` is **sync** and never blocks the caller — it enqueues the event
/// on an internal bounded channel. A background drain task processes events
/// and writes to storage asynchronously.
pub trait BillingCollector: Send + Sync {
    fn record(&self, event: BillingEvent);
    fn session_id(&self) -> Uuid;

    /// Convenience wrapper: records a transcript turn without constructing the
    /// full `BillingEvent::Transcript` at the call site.
    fn record_transcript(&self, entry: TranscriptEntry) {
        self.record(BillingEvent::Transcript(entry));
    }
}

// ---------------------------------------------------------------------------
// SessionBilling — the real implementation
// ---------------------------------------------------------------------------

/// Per-session accumulator. Construct with `SessionBilling::new()`, then clone
/// the returned `Arc` into each service handler.
///
/// Construction returns a `JoinHandle` for the background drain task.
/// The handle completes once all `Arc<SessionBilling>` clones are dropped
/// (which closes the channel) and the final `finalize_session` write succeeds.
pub struct SessionBilling {
    session_id: Uuid,
    tx: mpsc::Sender<BillingEvent>,
    // Summary is shared with the drain task so callers can read totals if needed.
    summary: Arc<Mutex<SessionSummary>>,
}

impl SessionBilling {
    /// Create a new accumulator backed by `storage`.
    ///
    /// `channel_capacity` — how many events can be buffered before `record()`
    /// starts dropping. 256 is a safe default for most deployments.
    pub fn new(
        session_id: Uuid,
        storage: Arc<dyn BillingStorage>,
        channel_capacity: usize,
    ) -> (Arc<Self>, tokio::task::JoinHandle<()>) {
        Self::new_with_dir(session_id, storage, channel_capacity, None)
    }

    /// Like `new`, but also writes `transcript.json` into `session_dir` at
    /// session end. Pass `{base_dir}/{session_id}` to co-locate the transcript
    /// with the audio WAV files produced by `LocalAudioStorage`.
    pub fn new_with_dir(
        session_id: Uuid,
        storage: Arc<dyn BillingStorage>,
        channel_capacity: usize,
        session_dir: Option<PathBuf>,
    ) -> (Arc<Self>, tokio::task::JoinHandle<()>) {
        let (tx, rx) = mpsc::channel(channel_capacity);
        let summary = Arc::new(Mutex::new(SessionSummary {
            session_id,
            ..Default::default()
        }));

        let collector = Arc::new(Self {
            session_id,
            tx,
            summary: summary.clone(),
        });

        let handle = tokio::spawn(drain_task(rx, summary, storage, session_dir));
        (collector, handle)
    }

    /// Read a snapshot of the accumulated totals (useful for mid-session queries).
    pub fn snapshot(&self) -> SessionSummary {
        self.summary.lock().unwrap().clone()
    }
}

impl BillingCollector for SessionBilling {
    fn record(&self, event: BillingEvent) {
        if let Err(e) = self.tx.try_send(event) {
            match e {
                mpsc::error::TrySendError::Full(_) => {
                    log::warn!("BillingCollector: channel full, dropping event");
                }
                mpsc::error::TrySendError::Closed(_) => {
                    // Drain task exited early (e.g. panicked). Nothing we can do.
                }
            }
        }
    }

    fn session_id(&self) -> Uuid {
        self.session_id
    }
}

// ---------------------------------------------------------------------------
// Drain task — runs independently, never on the audio hot path
// ---------------------------------------------------------------------------

async fn drain_task(
    mut rx: mpsc::Receiver<BillingEvent>,
    summary: Arc<Mutex<SessionSummary>>,
    storage: Arc<dyn BillingStorage>,
    session_dir: Option<PathBuf>,
) {
    let mut transcripts: Vec<TranscriptEntry> = Vec::new();

    while let Some(event) = rx.recv().await {
        // Collect transcript entries for the end-of-session JSON file.
        if let BillingEvent::Transcript(ref entry) = event {
            transcripts.push(entry.clone());
        }

        // Hold the Mutex for microseconds only — never across .await.
        {
            let mut s = summary.lock().unwrap();
            apply_event(&mut s, &event);
        }

        if let Err(e) = storage.record_event(&event).await {
            log::error!("BillingStorage::record_event failed: {e}");
        }
    }

    // Channel closed — all SessionBilling clones have been dropped.
    let final_summary = summary.lock().unwrap().clone();
    log::info!(
        "billing: session {} ended — {:.2}s | LLM in={} out={} calls={} \
         | TTS chars={} calls={} | STT ms={:.0} calls={}",
        final_summary.session_id,
        final_summary.duration_secs.unwrap_or(0.0),
        final_summary.llm_input_tokens,
        final_summary.llm_output_tokens,
        final_summary.llm_calls,
        final_summary.tts_chars,
        final_summary.tts_calls,
        final_summary.stt_audio_ms,
        final_summary.stt_calls,
    );

    if let Err(e) = storage.finalize_session(&final_summary, &transcripts).await {
        log::error!("BillingStorage::finalize_session failed: {e}");
    }

    // Optionally also write transcript.json to a local session directory.
    if let Some(dir) = session_dir {
        write_transcript_json(&dir, &transcripts).await;
    }
}

async fn write_transcript_json(dir: &PathBuf, transcripts: &[TranscriptEntry]) {
    if let Err(e) = tokio::fs::create_dir_all(dir).await {
        log::error!("billing: could not create session dir {}: {e}", dir.display());
        return;
    }
    match serde_json::to_string_pretty(transcripts) {
        Ok(json) => {
            let path = dir.join("transcript.json");
            if let Err(e) = tokio::fs::write(&path, json).await {
                log::error!("billing: transcript write failed {}: {e}", path.display());
            } else {
                log::info!("billing: wrote {}", path.display());
            }
        }
        Err(e) => log::error!("billing: transcript serialize failed: {e}"),
    }
}

fn apply_event(s: &mut SessionSummary, event: &BillingEvent) {
    match event {
        BillingEvent::SessionStart { started_at, metadata, .. } => {
            s.started_at = Some(*started_at);
            s.metadata   = metadata.clone();
        }
        BillingEvent::SessionEnd { ended_at, finish_reason, .. } => {
            s.ended_at      = Some(*ended_at);
            s.finish_reason = Some(finish_reason.clone());
            if let Some(start) = s.started_at {
                let ms = (*ended_at - start).num_milliseconds();
                s.duration_secs = Some(ms as f64 / 1000.0);
            }
        }
        BillingEvent::LlmUsage { input_tokens, output_tokens, .. } => {
            s.llm_input_tokens  += input_tokens;
            s.llm_output_tokens += output_tokens;
            s.llm_calls         += 1;
        }
        BillingEvent::TtsUsage { char_count, .. } => {
            s.tts_chars += char_count;
            s.tts_calls += 1;
        }
        BillingEvent::SttUsage { audio_duration_ms, .. } => {
            s.stt_audio_ms += audio_duration_ms;
            s.stt_calls    += 1;
        }
        // Transcript entries are stored individually — nothing to aggregate.
        BillingEvent::Transcript(_) => {}
    }
}

// ---------------------------------------------------------------------------
// NoopBillingCollector — zero-cost default
// ---------------------------------------------------------------------------

pub struct NoopBillingCollector;

impl BillingCollector for NoopBillingCollector {
    fn record(&self, _event: BillingEvent) {}
    fn session_id(&self) -> Uuid { Uuid::nil() }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::billing::storage::BillingStorage;
    use async_trait::async_trait;
    use chrono::Utc;
    use std::sync::Mutex as StdMutex;

    struct MemStorage(StdMutex<Vec<BillingEvent>>, StdMutex<Option<SessionSummary>>);

    #[async_trait]
    impl BillingStorage for MemStorage {
        async fn record_event(&self, e: &BillingEvent) -> crate::error::Result<()> {
            self.0.lock().unwrap().push(e.clone());
            Ok(())
        }
        async fn finalize_session(&self, s: &SessionSummary) -> crate::error::Result<()> {
            *self.1.lock().unwrap() = Some(s.clone());
            Ok(())
        }
    }

    #[tokio::test]
    async fn session_billing_accumulates_events() {
        let session_id = Uuid::new_v4();
        let storage = Arc::new(MemStorage(
            StdMutex::new(vec![]),
            StdMutex::new(None),
        ));

        let (billing, handle) = SessionBilling::new(session_id, storage.clone(), 64);

        billing.record(BillingEvent::SessionStart {
            session_id,
            started_at: Utc::now(),
            metadata: Default::default(),
        });
        billing.record(BillingEvent::LlmUsage {
            session_id,
            provider: "openai".into(),
            model: "gpt-4o".into(),
            input_tokens: 100,
            output_tokens: 50,
            estimated: false,
            occurred_at: Utc::now(),
        });
        billing.record(BillingEvent::TtsUsage {
            session_id,
            provider: "deepgram".into(),
            voice: "aura-2-helena-en".into(),
            char_count: 80,
            occurred_at: Utc::now(),
        });
        billing.record(BillingEvent::SttUsage {
            session_id,
            provider: "gnani".into(),
            audio_duration_ms: 3500.0,
            occurred_at: Utc::now(),
        });
        billing.record(BillingEvent::SessionEnd {
            session_id,
            ended_at: Utc::now(),
            finish_reason: "end".into(),
        });

        // Drop collector to close channel → drain task finalises.
        drop(billing);
        handle.await.unwrap();

        let summary = storage.1.lock().unwrap().clone().unwrap();
        assert_eq!(summary.llm_input_tokens, 100);
        assert_eq!(summary.llm_output_tokens, 50);
        assert_eq!(summary.llm_calls, 1);
        assert_eq!(summary.tts_chars, 80);
        assert_eq!(summary.tts_calls, 1);
        assert_eq!(summary.stt_audio_ms, 3500.0);
        assert_eq!(summary.stt_calls, 1);
        assert!(summary.duration_secs.is_some());

        let events = storage.0.lock().unwrap();
        // 5 events: start, llm, tts, stt, end
        assert_eq!(events.len(), 5);
    }

    #[test]
    fn noop_collector_session_id_is_nil() {
        assert_eq!(NoopBillingCollector.session_id(), Uuid::nil());
    }

    #[test]
    fn noop_collector_record_does_not_panic() {
        NoopBillingCollector.record(BillingEvent::SessionStart {
            session_id: Uuid::new_v4(),
            started_at: Utc::now(),
            metadata: Default::default(),
        });
    }

    #[tokio::test]
    async fn multiple_llm_calls_sum_tokens_and_increment_call_count() {
        let session_id = Uuid::new_v4();
        let storage = Arc::new(MemStorage(StdMutex::new(vec![]), StdMutex::new(None)));
        let (billing, handle) = SessionBilling::new(session_id, storage.clone(), 64);

        for _ in 0..3 {
            billing.record(BillingEvent::LlmUsage {
                session_id,
                provider: "openai".into(),
                model: "gpt-4o".into(),
                input_tokens: 100,
                output_tokens: 50,
                estimated: false,
                occurred_at: Utc::now(),
            });
        }
        drop(billing);
        handle.await.unwrap();

        let s = storage.1.lock().unwrap().clone().unwrap();
        assert_eq!(s.llm_input_tokens, 300);
        assert_eq!(s.llm_output_tokens, 150);
        assert_eq!(s.llm_calls, 3);
    }

    #[tokio::test]
    async fn multiple_tts_calls_sum_chars_and_increment_call_count() {
        let session_id = Uuid::new_v4();
        let storage = Arc::new(MemStorage(StdMutex::new(vec![]), StdMutex::new(None)));
        let (billing, handle) = SessionBilling::new(session_id, storage.clone(), 64);

        for chars in [100usize, 200, 300] {
            billing.record(BillingEvent::TtsUsage {
                session_id,
                provider: "deepgram".into(),
                voice: "aura-2-helena-en".into(),
                char_count: chars,
                occurred_at: Utc::now(),
            });
        }
        drop(billing);
        handle.await.unwrap();

        let s = storage.1.lock().unwrap().clone().unwrap();
        assert_eq!(s.tts_chars, 600);
        assert_eq!(s.tts_calls, 3);
    }

    #[tokio::test]
    async fn multiple_stt_calls_sum_duration_and_increment_call_count() {
        let session_id = Uuid::new_v4();
        let storage = Arc::new(MemStorage(StdMutex::new(vec![]), StdMutex::new(None)));
        let (billing, handle) = SessionBilling::new(session_id, storage.clone(), 64);

        billing.record(BillingEvent::SttUsage { session_id, provider: "gnani".into(), audio_duration_ms: 1000.0, occurred_at: Utc::now() });
        billing.record(BillingEvent::SttUsage { session_id, provider: "gnani".into(), audio_duration_ms: 2500.0, occurred_at: Utc::now() });
        drop(billing);
        handle.await.unwrap();

        let s = storage.1.lock().unwrap().clone().unwrap();
        assert!((s.stt_audio_ms - 3500.0).abs() < 0.001);
        assert_eq!(s.stt_calls, 2);
    }

    #[tokio::test]
    async fn session_duration_computed_from_start_and_end_timestamps() {
        let session_id = Uuid::new_v4();
        let storage = Arc::new(MemStorage(StdMutex::new(vec![]), StdMutex::new(None)));
        let (billing, handle) = SessionBilling::new(session_id, storage.clone(), 64);

        let start = Utc::now();
        billing.record(BillingEvent::SessionStart {
            session_id,
            started_at: start,
            metadata: Default::default(),
        });
        let end = start + chrono::Duration::milliseconds(2000);
        billing.record(BillingEvent::SessionEnd {
            session_id,
            ended_at: end,
            finish_reason: "end".into(),
        });
        drop(billing);
        handle.await.unwrap();

        let s = storage.1.lock().unwrap().clone().unwrap();
        let dur = s.duration_secs.expect("duration_secs must be set");
        assert!((dur - 2.0).abs() < 0.01, "expected ~2.0s, got {dur}");
        assert_eq!(s.finish_reason.as_deref(), Some("end"));
    }

    #[tokio::test]
    async fn channel_overflow_drops_gracefully_without_panic() {
        let session_id = Uuid::new_v4();
        let storage = Arc::new(MemStorage(StdMutex::new(vec![]), StdMutex::new(None)));
        // Capacity 1 — all but the first event will be dropped via try_send
        let (billing, handle) = SessionBilling::new(session_id, storage.clone(), 1);

        for _ in 0..20 {
            billing.record(BillingEvent::LlmUsage {
                session_id,
                provider: "openai".into(),
                model: "gpt-4o".into(),
                input_tokens: 10,
                output_tokens: 5,
                estimated: false,
                occurred_at: Utc::now(),
            });
        }
        drop(billing);
        handle.await.unwrap(); // must not panic
    }

    #[tokio::test]
    async fn snapshot_returns_accumulated_state() {
        let session_id = Uuid::new_v4();
        let storage = Arc::new(MemStorage(StdMutex::new(vec![]), StdMutex::new(None)));
        let (billing, handle) = SessionBilling::new(session_id, storage.clone(), 64);

        billing.record(BillingEvent::LlmUsage {
            session_id,
            provider: "openai".into(),
            model: "gpt-4o".into(),
            input_tokens: 50,
            output_tokens: 25,
            estimated: false,
            occurred_at: Utc::now(),
        });

        // Give the drain task time to process the event.
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        let snap = billing.snapshot();
        assert_eq!(snap.session_id, session_id);
        assert_eq!(snap.llm_input_tokens, 50);

        drop(billing);
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn session_metadata_forwarded_to_summary() {
        let session_id = Uuid::new_v4();
        let storage = Arc::new(MemStorage(StdMutex::new(vec![]), StdMutex::new(None)));
        let (billing, handle) = SessionBilling::new(session_id, storage.clone(), 64);

        let mut meta = std::collections::HashMap::new();
        meta.insert("user_id".into(), "u_42".into());
        meta.insert("region".into(), "ap-south-1".into());
        billing.record(BillingEvent::SessionStart {
            session_id,
            started_at: Utc::now(),
            metadata: meta,
        });
        drop(billing);
        handle.await.unwrap();

        let s = storage.1.lock().unwrap().clone().unwrap();
        assert_eq!(s.metadata.get("user_id").map(|s| s.as_str()), Some("u_42"));
        assert_eq!(s.metadata.get("region").map(|s| s.as_str()), Some("ap-south-1"));
    }

    #[tokio::test]
    async fn all_events_forwarded_to_storage_record_event() {
        let session_id = Uuid::new_v4();
        let storage = Arc::new(MemStorage(StdMutex::new(vec![]), StdMutex::new(None)));
        let (billing, handle) = SessionBilling::new(session_id, storage.clone(), 64);

        let now = Utc::now();
        billing.record(BillingEvent::SessionStart  { session_id, started_at: now, metadata: Default::default() });
        billing.record(BillingEvent::LlmUsage      { session_id, provider: "openai".into(), model: "gpt-4o".into(), input_tokens: 10, output_tokens: 5, estimated: false, occurred_at: now });
        billing.record(BillingEvent::TtsUsage      { session_id, provider: "deepgram".into(), voice: "v".into(), char_count: 50, occurred_at: now });
        billing.record(BillingEvent::SttUsage      { session_id, provider: "gnani".into(), audio_duration_ms: 1000.0, occurred_at: now });
        billing.record(BillingEvent::SessionEnd    { session_id, ended_at: now, finish_reason: "end".into() });
        drop(billing);
        handle.await.unwrap();

        let events = storage.0.lock().unwrap();
        assert_eq!(events.len(), 5, "all 5 events must reach storage.record_event");
    }
}
