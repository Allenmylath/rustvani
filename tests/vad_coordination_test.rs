//! VAD coordination tests — client VAD + server VAD toggle-switch deduplication.
//!
//! Run with:
//!   cargo test --test vad_coordination_test
//!
//! What is tested:
//!   1. A ClientVADUserStartedSpeaking frame emits exactly one VADUserStartedSpeaking downstream.
//!   2. A second ClientVADUserStartedSpeaking while already speaking is a no-op (toggle blocks it).
//!   3. ClientVADUserStoppedSpeaking emits VADUserStoppedSpeaking and resets the toggle.
//!   4. A second ClientVADUserStoppedSpeaking while not speaking is a no-op.
//!   5. After a full start→stop cycle the toggle resets — a new start works again.

use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use tokio::sync::mpsc;

use rustvani::{
    clock::system_clock,
    error::Result,
    frames::{
        Frame, FrameDirection, FrameHandler, FrameKind, FrameProcessor, FrameInner, SystemFrame,
    },
    pipeline::{PipelineParams, PipelineTask},
    transport::{ChannelMessage, ChannelTransport, TransportParams},
};

// ---------------------------------------------------------------------------
// VadCollector — records every VADUser* event kind that flows through
// ---------------------------------------------------------------------------

struct VadCollector {
    events: Arc<Mutex<Vec<FrameKind>>>,
}

#[async_trait]
impl FrameHandler for VadCollector {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        let kind = frame.kind();
        if matches!(kind, FrameKind::VADUserStartedSpeaking | FrameKind::VADUserStoppedSpeaking) {
            self.events.lock().unwrap().push(kind);
        }
        processor.push_frame(frame, direction).await?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn params_no_vad() -> TransportParams {
    TransportParams {
        audio_in_enabled:         true,
        audio_in_sample_rate:     Some(16_000),
        audio_in_channels:        1,
        audio_in_passthrough:     false,
        audio_in_stream_on_start: false, // no audio task — we inject frames manually
        audio_out_enabled:        true,
        audio_out_sample_rate:    Some(16_000),
        audio_out_channels:       1,
        audio_out_10ms_chunks:    4,
        vad_analyzer:             None,  // server VAD disabled
        ..TransportParams::default()
    }
}

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

/// Builds the pipeline and returns (task, push_tx_for_transport, push_tx_for_test, collected_events).
fn build(
    transport: &ChannelTransport,
) -> (
    PipelineTask,
    mpsc::Sender<(Frame, FrameDirection)>,
    mpsc::Sender<(Frame, FrameDirection)>,
    Arc<Mutex<Vec<FrameKind>>>,
) {
    let events = Arc::new(Mutex::new(Vec::<FrameKind>::new()));
    let collector = FrameProcessor::new(
        "VadCollector",
        Box::new(VadCollector { events: events.clone() }),
        false,
    );

    let task = PipelineTask::new(
        vec![transport.input(), collector, transport.output()],
        PipelineParams::default(),
    );

    let push_for_transport = task.push_sender();
    let push_for_test = push_for_transport.clone();

    (task, push_for_transport, push_for_test, events)
}

/// Drain all currently-collected events into a snapshot Vec.
fn snapshot(events: &Arc<Mutex<Vec<FrameKind>>>) -> Vec<FrameKind> {
    events.lock().unwrap().clone()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Test 1: toggle deduplication — two starts → one event; two stops → one event.
#[tokio::test]
async fn test_client_vad_toggle_deduplicates() {
    let (incoming_tx, incoming_rx) = mpsc::channel::<ChannelMessage>(10);
    let (outgoing_tx, _outgoing_rx) = mpsc::channel::<ChannelMessage>(32);

    let transport = ChannelTransport::new("vad-dedup", params_no_vad(), incoming_rx);
    let (task, push_tx, inject, events) = build(&transport);

    let pipeline_handle = tokio::spawn(async move {
        task.run(system_clock(), None).await.ok();
    });
    let transport_handle = tokio::spawn(async move {
        transport.run(push_tx, outgoing_tx).await;
    });

    // Allow the pipeline to settle after Start frame.
    tokio::time::sleep(Duration::from_millis(80)).await;

    let ts = now_ts();

    // ── First start: should emit VADUserStartedSpeaking ─────────────────────
    inject
        .send((Frame::client_vad_user_started_speaking(ts), FrameDirection::Downstream))
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    let snap = snapshot(&events);
    assert_eq!(snap.len(), 1, "first start: expected 1 event, got {:?}", snap);
    assert_eq!(snap[0], FrameKind::VADUserStartedSpeaking, "expected VADUserStartedSpeaking");

    // ── Second start while already speaking: no-op ───────────────────────────
    inject
        .send((Frame::client_vad_user_started_speaking(ts), FrameDirection::Downstream))
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    let snap = snapshot(&events);
    assert_eq!(snap.len(), 1, "second start should be no-op, still 1 event, got {:?}", snap);

    // ── First stop: should emit VADUserStoppedSpeaking ───────────────────────
    inject
        .send((Frame::client_vad_user_stopped_speaking(ts), FrameDirection::Downstream))
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    let snap = snapshot(&events);
    assert_eq!(snap.len(), 2, "first stop: expected 2 events total, got {:?}", snap);
    assert_eq!(snap[1], FrameKind::VADUserStoppedSpeaking, "expected VADUserStoppedSpeaking");

    // ── Second stop while not speaking: no-op ────────────────────────────────
    inject
        .send((Frame::client_vad_user_stopped_speaking(ts), FrameDirection::Downstream))
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    let snap = snapshot(&events);
    assert_eq!(snap.len(), 2, "second stop should be no-op, still 2 events, got {:?}", snap);

    // Shutdown
    drop(incoming_tx);
    let _ = tokio::time::timeout(Duration::from_secs(3), pipeline_handle).await;
    let _ = tokio::time::timeout(Duration::from_secs(3), transport_handle).await;
}

/// Test 2: toggle resets — after a full start→stop cycle the next start works again.
#[tokio::test]
async fn test_client_vad_toggle_resets_after_stop() {
    let (incoming_tx, incoming_rx) = mpsc::channel::<ChannelMessage>(10);
    let (outgoing_tx, _outgoing_rx) = mpsc::channel::<ChannelMessage>(32);

    let transport = ChannelTransport::new("vad-reset", params_no_vad(), incoming_rx);
    let (task, push_tx, inject, events) = build(&transport);

    let pipeline_handle = tokio::spawn(async move {
        task.run(system_clock(), None).await.ok();
    });
    let transport_handle = tokio::spawn(async move {
        transport.run(push_tx, outgoing_tx).await;
    });

    tokio::time::sleep(Duration::from_millis(80)).await;

    let ts = now_ts();

    // First utterance: start → stop
    inject.send((Frame::client_vad_user_started_speaking(ts), FrameDirection::Downstream)).await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    inject.send((Frame::client_vad_user_stopped_speaking(ts), FrameDirection::Downstream)).await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    let snap = snapshot(&events);
    assert_eq!(snap.len(), 2, "first utterance: expected [Started, Stopped], got {:?}", snap);

    // Second utterance: start should work (toggle was reset by stop)
    inject.send((Frame::client_vad_user_started_speaking(ts), FrameDirection::Downstream)).await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    let snap = snapshot(&events);
    assert_eq!(snap.len(), 3, "second utterance start: expected 3 events total, got {:?}", snap);
    assert_eq!(snap[2], FrameKind::VADUserStartedSpeaking, "third event should be VADUserStartedSpeaking");

    // And stop it cleanly too
    inject.send((Frame::client_vad_user_stopped_speaking(ts), FrameDirection::Downstream)).await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    let snap = snapshot(&events);
    assert_eq!(snap.len(), 4, "second utterance stop: expected 4 events total, got {:?}", snap);
    assert_eq!(snap[3], FrameKind::VADUserStoppedSpeaking);

    drop(incoming_tx);
    let _ = tokio::time::timeout(Duration::from_secs(3), pipeline_handle).await;
    let _ = tokio::time::timeout(Duration::from_secs(3), transport_handle).await;
}

/// Test 3: order of events is always Started before Stopped within a single utterance.
#[tokio::test]
async fn test_client_vad_event_ordering() {
    let (incoming_tx, incoming_rx) = mpsc::channel::<ChannelMessage>(10);
    let (outgoing_tx, _outgoing_rx) = mpsc::channel::<ChannelMessage>(32);

    let transport = ChannelTransport::new("vad-order", params_no_vad(), incoming_rx);
    let (task, push_tx, inject, events) = build(&transport);

    let pipeline_handle = tokio::spawn(async move {
        task.run(system_clock(), None).await.ok();
    });
    let transport_handle = tokio::spawn(async move {
        transport.run(push_tx, outgoing_tx).await;
    });

    tokio::time::sleep(Duration::from_millis(80)).await;

    let ts = now_ts();

    // Two utterances in sequence
    for _ in 0..3 {
        inject.send((Frame::client_vad_user_started_speaking(ts), FrameDirection::Downstream)).await.unwrap();
        tokio::time::sleep(Duration::from_millis(30)).await;
        inject.send((Frame::client_vad_user_stopped_speaking(ts), FrameDirection::Downstream)).await.unwrap();
        tokio::time::sleep(Duration::from_millis(30)).await;
    }

    tokio::time::sleep(Duration::from_millis(50)).await;
    let snap = snapshot(&events);

    assert_eq!(snap.len(), 6, "3 utterances × 2 events = 6 total, got {:?}", snap);

    // Verify ordering: Started, Stopped, Started, Stopped, Started, Stopped
    let expected = [
        FrameKind::VADUserStartedSpeaking,
        FrameKind::VADUserStoppedSpeaking,
        FrameKind::VADUserStartedSpeaking,
        FrameKind::VADUserStoppedSpeaking,
        FrameKind::VADUserStartedSpeaking,
        FrameKind::VADUserStoppedSpeaking,
    ];
    assert_eq!(snap.as_slice(), &expected, "event sequence mismatch: {:?}", snap);

    drop(incoming_tx);
    let _ = tokio::time::timeout(Duration::from_secs(3), pipeline_handle).await;
    let _ = tokio::time::timeout(Duration::from_secs(3), transport_handle).await;
}
