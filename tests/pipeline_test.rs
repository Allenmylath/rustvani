use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::Mutex;

use pipecat_core::*;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// A FrameHandler that records every frame it receives (in order)
/// and then passes it along unchanged.
struct CapturingHandler {
    pub received: Arc<Mutex<Vec<(String, FrameDirection)>>>,
    pub pass_through: bool,
}

impl CapturingHandler {
    fn new() -> (Self, Arc<Mutex<Vec<(String, FrameDirection)>>>) {
        let store = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                received: store.clone(),
                pass_through: true,
            },
            store,
        )
    }

    fn new_sink() -> (Self, Arc<Mutex<Vec<(String, FrameDirection)>>>) {
        let store = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                received: store.clone(),
                pass_through: false,
            },
            store,
        )
    }
}

#[async_trait]
impl FrameHandler for CapturingHandler {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        self.received
            .lock()
            .await
            .push((frame.name().to_owned(), direction));

        if self.pass_through {
            processor.push_frame(frame, direction).await?;
        }
        Ok(())
    }
}

/// Shared setup helper: clock + no observer.
fn make_setup() -> FrameProcessorSetup {
    FrameProcessorSetup {
        clock: Arc::new(SystemClock),
        observer: None,
    }
}

// ---------------------------------------------------------------------------
// 1. StartFrame initialises processor state
// ---------------------------------------------------------------------------
#[tokio::test]
async fn test_start_frame_initialises_state() {
    let p = FrameProcessor::new("p", Box::new(PassthroughHandler), false);
    p.setup(make_setup()).await.unwrap();

    assert!(!p.metrics_enabled(), "metrics should be disabled before StartFrame");

    p.queue_frame(
        Frame::new_start(StartFrameData {
            enable_metrics: true,
            allow_interruptions: true,
            ..Default::default()
        }),
        FrameDirection::Downstream,
        None,
    )
    .await
    .unwrap();

    tokio::time::sleep(Duration::from_millis(50)).await;

    assert!(p.metrics_enabled(), "metrics should be enabled after StartFrame");
    assert!(p.interruptions_allowed(), "interruptions should be allowed");

    p.cleanup().await.unwrap();
}

// ---------------------------------------------------------------------------
// 2. Frames flow through a two-processor pipeline
// ---------------------------------------------------------------------------
#[tokio::test]
async fn test_frames_flow_downstream() {
    let (handler, received) = CapturingHandler::new_sink();

    let p1 = FrameProcessor::new("p1", Box::new(PassthroughHandler), false);
    let p2 = FrameProcessor::new("p2", Box::new(handler), false);

    p1.link(p2.clone()).await;
    p1.setup(make_setup()).await.unwrap();
    p2.setup(make_setup()).await.unwrap();

    let start = Frame::new_start(StartFrameData::default());
    p1.queue_frame(start, FrameDirection::Downstream, None)
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;

    p1.queue_frame(
        Frame::new_data(b"hello".to_vec()),
        FrameDirection::Downstream,
        None,
    )
    .await
    .unwrap();

    tokio::time::sleep(Duration::from_millis(50)).await;

    let frames = received.lock().await;
    assert!(
        frames.iter().any(|(name, dir)| name == "DataFrame" && *dir == FrameDirection::Downstream),
        "DataFrame should have reached p2 downstream, got: {:?}", *frames
    );

    p1.cleanup().await.unwrap();
    p2.cleanup().await.unwrap();
}

// ---------------------------------------------------------------------------
// 3. System frames are delivered before non-system frames (priority queue)
// ---------------------------------------------------------------------------
#[tokio::test]
async fn test_system_frames_arrive_before_data_frames() {
    let (handler, received) = CapturingHandler::new_sink();

    let p = FrameProcessor::new("p", Box::new(handler), false);
    p.setup(make_setup()).await.unwrap();

    let start = Frame::new_start(StartFrameData::default());
    p.queue_frame(start, FrameDirection::Downstream, None)
        .await
        .unwrap();

    for i in 0..5u8 {
        p.queue_frame(
            Frame::new_data(vec![i]),
            FrameDirection::Downstream,
            None,
        )
        .await
        .unwrap();
    }

    tokio::time::sleep(Duration::from_millis(100)).await;

    let frames = received.lock().await;
    assert_eq!(
        frames[0].0, "StartFrame",
        "StartFrame must be processed first. Got: {:?}", frames
    );

    p.cleanup().await.unwrap();
}

// ---------------------------------------------------------------------------
// 4. pause_processing_frames / resume_processing_frames
// ---------------------------------------------------------------------------
#[tokio::test]
async fn test_pause_and_resume() {
    let (handler, received) = CapturingHandler::new_sink();

    let p = FrameProcessor::new("p", Box::new(handler), false);
    p.setup(make_setup()).await.unwrap();

    p.queue_frame(
        Frame::new_start(StartFrameData::default()),
        FrameDirection::Downstream,
        None,
    )
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;

    p.pause_processing_frames().await;

    for _ in 0..3u8 {
        p.queue_frame(
            Frame::new_data(vec![42]),
            FrameDirection::Downstream,
            None,
        )
        .await
        .unwrap();
    }

    tokio::time::sleep(Duration::from_millis(30)).await;
    let count_while_paused = received
        .lock()
        .await
        .iter()
        .filter(|(n, _)| n == "DataFrame")
        .count();
    assert_eq!(count_while_paused, 0, "No DataFrames should arrive while paused");

    p.resume_processing_frames().await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    let count_after_resume = received
        .lock()
        .await
        .iter()
        .filter(|(n, _)| n == "DataFrame")
        .count();
    assert_eq!(count_after_resume, 3, "All 3 DataFrames should arrive after resume");

    p.cleanup().await.unwrap();
}

// ---------------------------------------------------------------------------
// 5. CancelFrame stops further processing
// ---------------------------------------------------------------------------
#[tokio::test]
async fn test_cancel_stops_processing() {
    let (handler, received) = CapturingHandler::new_sink();

    let p = FrameProcessor::new("p", Box::new(handler), false);
    p.setup(make_setup()).await.unwrap();

    p.queue_frame(
        Frame::new_start(StartFrameData::default()),
        FrameDirection::Downstream,
        None,
    )
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;

    p.queue_frame(Frame::new_cancel(), FrameDirection::Downstream, None)
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;

    let count_before = received.lock().await.len();
    p.queue_frame(
        Frame::new_data(vec![99]),
        FrameDirection::Downstream,
        None,
    )
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;

    let count_after = received.lock().await.len();
    assert_eq!(
        count_before, count_after,
        "No new frames should be processed after CancelFrame"
    );

    p.cleanup().await.unwrap();
}

// ---------------------------------------------------------------------------
// 6. broadcast_frame pushes downstream to next processor
// ---------------------------------------------------------------------------
#[tokio::test]
async fn test_broadcast_frame_sibling_ids() {
    let (h2, received_p2) = CapturingHandler::new_sink();

    let p1 = FrameProcessor::new("p1", Box::new(PassthroughHandler), false);
    let p2 = FrameProcessor::new("p2", Box::new(h2), false);

    p1.link(p2.clone()).await;
    p1.setup(make_setup()).await.unwrap();
    p2.setup(make_setup()).await.unwrap();

    p1.queue_frame(Frame::new_start(StartFrameData::default()), FrameDirection::Downstream, None).await.unwrap();
    p2.queue_frame(Frame::new_start(StartFrameData::default()), FrameDirection::Downstream, None).await.unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;

    p1.broadcast_frame(Frame::new_interruption()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    let p2_frames = received_p2.lock().await;
    assert!(
        p2_frames.iter().any(|(n, _)| n == "InterruptionFrame"),
        "p2 should receive the downstream InterruptionFrame from broadcast, got: {:?}", *p2_frames
    );

    p1.cleanup().await.unwrap();
    p2.cleanup().await.unwrap();
}

// ---------------------------------------------------------------------------
// 7. Interruption drains both queues, keeps UninterruptibleFrames
// ---------------------------------------------------------------------------
#[tokio::test]
async fn test_interruption_drains_queue_but_keeps_uninterruptible() {
    let (handler, received) = CapturingHandler::new_sink();

    let p = FrameProcessor::new("p", Box::new(handler), false);
    p.setup(make_setup()).await.unwrap();

    p.queue_frame(
        Frame::new_start(StartFrameData::default()),
        FrameDirection::Downstream,
        None,
    )
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;

    // Pause so frames queue up without being processed
    p.pause_processing_frames().await;

    p.queue_frame(
        Frame::new_data(b"discard_me".to_vec()),
        FrameDirection::Downstream,
        None,
    )
    .await
    .unwrap();

    p.queue_frame(
        Frame::new_uninterruptible(Frame::new_data(b"keep_me".to_vec())),
        FrameDirection::Downstream,
        None,
    )
    .await
    .unwrap();

    p.queue_frame(
        Frame::new_data(b"discard_me_too".to_vec()),
        FrameDirection::Downstream,
        None,
    )
    .await
    .unwrap();

    // Drain covers both input_queue and process_queue — no sleep needed.
    // Frames may still be in input_queue (not yet transferred by the input task)
    // but drain_process_queue now handles both atomically.
    p.drain_process_queue().await;

    p.resume_processing_frames().await;
    tokio::time::sleep(Duration::from_millis(80)).await;

    let frames = received.lock().await;
    let data_frames: Vec<_> = frames
        .iter()
        .filter(|(n, _)| n == "DataFrame" || n == "UninterruptibleFrame")
        .collect();

    assert_eq!(
        data_frames.len(), 1,
        "Only the UninterruptibleFrame should survive the drain, got: {:?}", data_frames
    );
    assert_eq!(data_frames[0].0, "UninterruptibleFrame");

    p.cleanup().await.unwrap();
}

// ---------------------------------------------------------------------------
// 8. push_error_frame fires on_error handlers and pushes upstream
// ---------------------------------------------------------------------------
#[tokio::test]
async fn test_push_error_frame_fires_event_handler() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let error_fired = Arc::new(AtomicBool::new(false));
    let error_fired_clone = error_fired.clone();

    let p = FrameProcessor::new("p", Box::new(PassthroughHandler), false);
    p.setup(make_setup()).await.unwrap();

    p.on_error(move |_err| {
        error_fired_clone.store(true, Ordering::Relaxed);
    })
    .await;

    p.queue_frame(
        Frame::new_start(StartFrameData::default()),
        FrameDirection::Downstream,
        None,
    )
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;

    p.push_error("something went wrong", false).await.unwrap();

    tokio::time::sleep(Duration::from_millis(30)).await;
    assert!(
        error_fired.load(Ordering::Relaxed),
        "on_error handler should have fired"
    );

    p.cleanup().await.unwrap();
}

// ---------------------------------------------------------------------------
// 9. Direct mode: frames are processed synchronously without tasks
// ---------------------------------------------------------------------------
#[tokio::test]
async fn test_direct_mode_processes_synchronously() {
    let (handler, received) = CapturingHandler::new_sink();

    let p = FrameProcessor::new("p", Box::new(handler), true);
    p.setup(make_setup()).await.unwrap();

    p.process_frame(
        Frame::new_start(StartFrameData::default()),
        FrameDirection::Downstream,
    )
    .await
    .unwrap();

    p.process_frame(
        Frame::new_data(b"direct".to_vec()),
        FrameDirection::Downstream,
    )
    .await
    .unwrap();

    let frames = received.lock().await;
    assert!(
        frames.iter().any(|(n, _)| n == "DataFrame"),
        "DataFrame should be captured synchronously in direct mode"
    );

    p.cleanup().await.unwrap();
}

// ---------------------------------------------------------------------------
// 10. Three-processor chain: p1 -> p2 -> p3
// ---------------------------------------------------------------------------
#[tokio::test]
async fn test_three_processor_chain() {
    let (h1, _r1) = CapturingHandler::new();
    let (h2, _r2) = CapturingHandler::new();
    let (h3, r3) = CapturingHandler::new_sink();

    let p1 = FrameProcessor::new("p1", Box::new(h1), false);
    let p2 = FrameProcessor::new("p2", Box::new(h2), false);
    let p3 = FrameProcessor::new("p3", Box::new(h3), false);

    p1.link(p2.clone()).await;
    p2.link(p3.clone()).await;

    for p in [&p1, &p2, &p3] {
        p.setup(make_setup()).await.unwrap();
    }

    let start = Frame::new_start(StartFrameData::default());
    p1.queue_frame(start, FrameDirection::Downstream, None)
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    p1.queue_frame(
        Frame::new_data(b"end_to_end".to_vec()),
        FrameDirection::Downstream,
        None,
    )
    .await
    .unwrap();

    tokio::time::sleep(Duration::from_millis(80)).await;

    let frames = r3.lock().await;
    assert!(
        frames.iter().any(|(n, _)| n == "DataFrame"),
        "DataFrame should have traversed the full chain to p3, got: {:?}", *frames
    );

    p1.cleanup().await.unwrap();
    p2.cleanup().await.unwrap();
    p3.cleanup().await.unwrap();
}
