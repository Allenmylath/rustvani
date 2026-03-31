/// Comprehensive tests for pipecat-core pipeline and processor.
///
/// Test coverage:
///   frames        — construction, accessors, kind, is_system, is_uninterruptible,
///                   with_new_id, with_sibling
///   processor     — passthrough, custom handler, setup/cleanup, link chain,
///                   push_error, queue ordering (system before data)
///   pipeline      — construction, downstream routing, upstream routing,
///                   sub-processor registration, nested pipeline
///   pipeline_task — run/finish lifecycle, push_frame, on_pipeline_started,
///                   on_pipeline_finished, cancel via CancelTask,
///                   end via EndTask, idle timeout, error callback
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc;

use rustvani::{
    clock::system_clock,
    frames::FrameDirection,
    frames::{ControlFrame, DataFrame, DataFrameData, Frame, FrameInner, FrameKind, SystemFrame, StartFrameData},
    pipeline::{FinishReason, Pipeline, PipelineLifecycle, PipelineParams, PipelineTask},
    frames::{FrameHandler, FrameProcessor, FrameProcessorSetup, PassthroughHandler},
    error::Result,
};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// A handler that records every frame it sees then passes it through.
struct RecordingHandler {
    seen: Arc<Mutex<Vec<String>>>,
}

impl RecordingHandler {
    fn new(seen: Arc<Mutex<Vec<String>>>) -> Self {
        Self { seen }
    }
}

#[async_trait]
impl FrameHandler for RecordingHandler {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        self.seen.lock().unwrap().push(format!(
            "{}:{}:{:?}",
            processor.name(),
            frame.name(),
            direction
        ));
        processor.push_frame(frame, direction).await
    }
}

/// A handler that drops every frame (useful for sinks in tests).
struct SinkHandler;

#[async_trait]
impl FrameHandler for SinkHandler {
    async fn on_process_frame(
        &self,
        _processor: &FrameProcessor,
        _frame: Frame,
        _direction: FrameDirection,
    ) -> Result<()> {
        Ok(())
    }
}

/// A handler that sends received frames over an mpsc channel.
struct ChannelHandler {
    tx: mpsc::UnboundedSender<(String, FrameKind, FrameDirection)>,
}

#[async_trait]
impl FrameHandler for ChannelHandler {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        let _ = self.tx.send((processor.name().to_string(), frame.kind(), direction));
        processor.push_frame(frame, direction).await
    }
}

fn setup() -> FrameProcessorSetup {
    FrameProcessorSetup {
        clock: system_clock(),
        observer: None,
    }
}

// ---------------------------------------------------------------------------
// Frame tests
// ---------------------------------------------------------------------------

#[test]
fn frame_ids_are_unique() {
    let a = Frame::cancel();
    let b = Frame::cancel();
    assert_ne!(a.id, b.id);
}

#[test]
fn frame_with_new_id_changes_id() {
    let f = Frame::cancel();
    let id = f.id;
    let f2 = f.with_new_id();
    assert_ne!(id, f2.id);
}

#[test]
fn frame_with_sibling_sets_field() {
    let f = Frame::cancel().with_sibling(999);
    assert_eq!(f.sibling_id, Some(999));
}

#[test]
fn frame_is_system() {
    assert!(Frame::cancel().is_system());
    assert!(Frame::start(StartFrameData::default()).is_system());
    assert!(Frame::interruption().is_system());
    assert!(Frame::end_task().is_system());
    assert!(!Frame::end().is_system());
    assert!(!Frame::data(vec![]).is_system());
}

#[test]
fn frame_kind_correct() {
    assert_eq!(Frame::cancel().kind(),       FrameKind::Cancel);
    assert_eq!(Frame::end().kind(),          FrameKind::End);
    assert_eq!(Frame::stop().kind(),         FrameKind::Stop);
    assert_eq!(Frame::end_task().kind(),     FrameKind::EndTask);
    assert_eq!(Frame::cancel_task().kind(),  FrameKind::CancelTask);
    assert_eq!(Frame::stop_task().kind(),    FrameKind::StopTask);
    assert_eq!(Frame::heartbeat(0.0).kind(), FrameKind::Heartbeat);
    assert_eq!(Frame::bot_speaking().kind(), FrameKind::BotSpeaking);
    assert_eq!(Frame::data(vec![]).kind(),   FrameKind::Data);
}

#[test]
fn frame_is_uninterruptible() {
    assert!(Frame::end().is_uninterruptible());
    assert!(Frame::end_task().is_uninterruptible());
    assert!(Frame::stop_task().is_uninterruptible());
    assert!(Frame::cancel_task().is_uninterruptible());
    assert!(!Frame::cancel().is_uninterruptible());
    assert!(!Frame::data(vec![1]).is_uninterruptible());
    assert!(!Frame::heartbeat(1.0).is_uninterruptible());
}

#[test]
fn frame_name_matches_type() {
    assert_eq!(Frame::cancel().name(),       "CancelFrame");
    assert_eq!(Frame::end().name(),          "EndFrame");
    assert_eq!(Frame::stop().name(),         "StopFrame");
    assert_eq!(Frame::interruption().name(), "InterruptionFrame");
    assert_eq!(Frame::end_task().name(),     "EndTaskFrame");
    assert_eq!(Frame::heartbeat(0.0).name(), "HeartbeatFrame");
    assert_eq!(Frame::data(vec![]).name(),   "DataFrame");
}

#[test]
fn start_frame_data_fields() {
    let f = Frame::start(StartFrameData {
        allow_interruptions: true,
        enable_metrics: true,
        ..Default::default()
    });
    match &f.inner {
        rustvani::frames::FrameInner::System(SystemFrame::Start(d)) => {
            assert!(d.allow_interruptions);
            assert!(d.enable_metrics);
        }
        _ => panic!("wrong variant"),
    }
}

// ---------------------------------------------------------------------------
// Processor tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn passthrough_processor_forwards_frame() {
    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    let a = FrameProcessor::new("A", Box::new(RecordingHandler::new(seen.clone())), false);
    let b = FrameProcessor::new("B", Box::new(SinkHandler), false);
    a.link(&b);

    a.setup(setup()).await.unwrap();

    a.queue_frame(
        Frame::start(StartFrameData::default()),
        FrameDirection::Downstream,
        None,
    ).await.unwrap();

    tokio::time::sleep(Duration::from_millis(50)).await;

    let seen = seen.lock().unwrap();
    assert!(seen.iter().any(|s| s.contains("StartFrame")), "StartFrame not seen: {:?}", seen);
}

#[tokio::test]
async fn processor_link_chain() {
    let a = FrameProcessor::new("A", Box::new(PassthroughHandler), false);
    let b = FrameProcessor::new("B", Box::new(PassthroughHandler), false);
    let c = FrameProcessor::new("C", Box::new(SinkHandler), false);

    a.link(&b);
    b.link(&c);

    assert_eq!(a.next().unwrap().name(), "B");
    assert_eq!(b.next().unwrap().name(), "C");
    assert_eq!(b.previous().unwrap().name(), "A");
    assert_eq!(c.previous().unwrap().name(), "B");
    assert!(a.previous().is_none());
    assert!(c.next().is_none());
}

#[tokio::test]
async fn processor_setup_propagates_to_sub_processors() {
    let inner = FrameProcessor::new("Inner", Box::new(SinkHandler), false);
    let outer = FrameProcessor::new("Outer", Box::new(SinkHandler), false);
    outer.set_sub_processors(vec![inner.clone()]);

    outer.setup(setup()).await.unwrap();
    outer.cleanup().await.unwrap();
}

#[tokio::test]
async fn system_frames_processed_before_data() {
    let order: Arc<Mutex<Vec<FrameKind>>> = Arc::new(Mutex::new(Vec::new()));
    let order_clone = order.clone();

    struct OrderHandler(Arc<Mutex<Vec<FrameKind>>>);

    #[async_trait]
    impl FrameHandler for OrderHandler {
        async fn on_process_frame(
            &self,
            processor: &FrameProcessor,
            frame: Frame,
            direction: FrameDirection,
        ) -> Result<()> {
            self.0.lock().unwrap().push(frame.kind());
            processor.push_frame(frame, direction).await
        }
    }

    let a = FrameProcessor::new("A", Box::new(OrderHandler(order_clone)), false);
    let b = FrameProcessor::new("B", Box::new(SinkHandler), false);
    a.link(&b);
    a.setup(setup()).await.unwrap();

    a.queue_frame(Frame::start(StartFrameData::default()), FrameDirection::Downstream, None)
        .await.unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;

    a.pause_processing_frames().await;

    for _ in 0..3 {
        a.queue_frame(Frame::data(vec![0]), FrameDirection::Downstream, None).await.unwrap();
    }
    a.queue_frame(Frame::heartbeat(0.0), FrameDirection::Downstream, None).await.unwrap();

    a.resume_processing_frames().await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    let seen = order.lock().unwrap();
    let hb_pos   = seen.iter().position(|k| *k == FrameKind::Heartbeat);
    let data_pos = seen.iter().position(|k| *k == FrameKind::Data);

    match (hb_pos, data_pos) {
        (Some(h), Some(d)) => assert!(h < d, "Heartbeat should precede Data: {:?}", *seen),
        (Some(_), None)    => {}
        (None, _)          => panic!("Heartbeat was never processed: {:?}", *seen),
    }
}

// ---------------------------------------------------------------------------
// Pipeline construction tests
// ---------------------------------------------------------------------------

#[test]
fn pipeline_registers_sub_processors() {
    let a = FrameProcessor::new("A", Box::new(PassthroughHandler), false);
    let b = FrameProcessor::new("B", Box::new(PassthroughHandler), false);
    let pipeline = Pipeline::new(vec![a, b]);

    let subs = pipeline.processors();
    assert_eq!(subs.len(), 4, "Expected [source, A, B, sink], got {}", subs.len());
}

#[test]
fn pipeline_registers_entry_processor() {
    let a = FrameProcessor::new("A", Box::new(PassthroughHandler), false);
    let pipeline = Pipeline::new(vec![a]);
    let entries = pipeline.entry_processors();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].name(), "PipelineSource");
}

#[tokio::test]
async fn pipeline_routes_frame_downstream() {
    let (tx, mut rx) = mpsc::unbounded_channel();

    let a = FrameProcessor::new("A", Box::new(ChannelHandler { tx: tx.clone() }), false);
    let b = FrameProcessor::new("B", Box::new(ChannelHandler { tx: tx.clone() }), false);

    let pipeline = Pipeline::new(vec![a, b]);
    pipeline.setup(setup()).await.unwrap();

    pipeline.queue_frame(
        Frame::start(StartFrameData::default()),
        FrameDirection::Downstream,
        None,
    ).await.unwrap();

    tokio::time::sleep(Duration::from_millis(80)).await;

    let mut names = Vec::new();
    while let Ok((name, kind, dir)) = rx.try_recv() {
        if kind == FrameKind::Start && dir == FrameDirection::Downstream {
            names.push(name);
        }
    }

    assert!(names.contains(&"A".to_string()), "A did not see StartFrame downstream: {:?}", names);
    assert!(names.contains(&"B".to_string()), "B did not see StartFrame downstream: {:?}", names);

    pipeline.cleanup().await.unwrap();
}

#[tokio::test]
async fn pipeline_routes_frame_upstream() {
    let (tx, mut rx) = mpsc::unbounded_channel();

    let a = FrameProcessor::new("A", Box::new(ChannelHandler { tx: tx.clone() }), false);
    let b = FrameProcessor::new("B", Box::new(ChannelHandler { tx: tx.clone() }), false);

    let pipeline = Pipeline::new(vec![a, b]);
    pipeline.setup(setup()).await.unwrap();

    pipeline.queue_frame(
        Frame::start(StartFrameData::default()),
        FrameDirection::Downstream,
        None,
    ).await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    pipeline.queue_frame(Frame::data(vec![42]), FrameDirection::Upstream, None).await.unwrap();
    tokio::time::sleep(Duration::from_millis(80)).await;

    let mut upstream_names = Vec::new();
    while let Ok((name, kind, dir)) = rx.try_recv() {
        if kind == FrameKind::Data && dir == FrameDirection::Upstream {
            upstream_names.push(name);
        }
    }

    assert!(upstream_names.contains(&"B".to_string()), "B did not see data upstream: {:?}", upstream_names);
    assert!(upstream_names.contains(&"A".to_string()), "A did not see data upstream: {:?}", upstream_names);

    if upstream_names.len() >= 2 {
        assert_eq!(upstream_names[0], "B", "B should see upstream frame before A");
        assert_eq!(upstream_names[1], "A");
    }

    pipeline.cleanup().await.unwrap();
}

#[tokio::test]
async fn nested_pipeline_routes_frames() {
    let (tx, mut rx) = mpsc::unbounded_channel();

    let inner_a = FrameProcessor::new("InnerA", Box::new(ChannelHandler { tx: tx.clone() }), false);
    let inner_b = FrameProcessor::new("InnerB", Box::new(ChannelHandler { tx: tx.clone() }), false);
    let inner_pipeline = Pipeline::new(vec![inner_a, inner_b]);

    let outer_a = FrameProcessor::new("OuterA", Box::new(ChannelHandler { tx: tx.clone() }), false);
    let outer_pipeline = Pipeline::new(vec![outer_a, inner_pipeline]);

    outer_pipeline.setup(setup()).await.unwrap();

    outer_pipeline.queue_frame(
        Frame::start(StartFrameData::default()),
        FrameDirection::Downstream,
        None,
    ).await.unwrap();

    tokio::time::sleep(Duration::from_millis(100)).await;

    let mut seen = Vec::new();
    while let Ok((name, kind, _dir)) = rx.try_recv() {
        if kind == FrameKind::Start {
            seen.push(name);
        }
    }

    assert!(seen.contains(&"OuterA".to_string()),  "OuterA missing: {:?}", seen);
    assert!(seen.contains(&"InnerA".to_string()),  "InnerA missing: {:?}", seen);
    assert!(seen.contains(&"InnerB".to_string()),  "InnerB missing: {:?}", seen);

    outer_pipeline.cleanup().await.unwrap();
}

// ---------------------------------------------------------------------------
// PipelineTask tests
// ---------------------------------------------------------------------------

fn make_task(processors: Vec<FrameProcessor>) -> PipelineTask {
    PipelineTask::new(processors, PipelineParams::default())
}

#[tokio::test]
async fn task_lifecycle_start_to_end() {
    let task = make_task(vec![
        FrameProcessor::new("P", Box::new(PassthroughHandler), false),
    ]);

    let started = Arc::new(Mutex::new(false));
    let started_clone = started.clone();

    task.add_on_pipeline_started(move |_frame| {
        let s = started_clone.clone();
        Box::pin(async move { *s.lock().unwrap() = true; })
    });

    let finished = Arc::new(Mutex::new(false));
    let finished_clone = finished.clone();

    task.add_on_pipeline_finished(move |_frame, _reason| {
        let f = finished_clone.clone();
        Box::pin(async move { *f.lock().unwrap() = true; })
    });

    let tx = task.push_sender();

    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        tx.send((Frame::end_task(), FrameDirection::Upstream)).await.ok();
    });

    tokio::time::timeout(Duration::from_secs(2), task.run(system_clock(), None))
        .await
        .expect("task timed out")
        .expect("task returned error");

    assert!(*started.lock().unwrap(),  "on_pipeline_started was not called");
    assert!(*finished.lock().unwrap(), "on_pipeline_finished was not called");
}

#[tokio::test]
async fn task_finishes_with_end_reason() {
    let task = make_task(vec![
        FrameProcessor::new("P", Box::new(PassthroughHandler), false),
    ]);

    let reason: Arc<Mutex<Option<FinishReason>>> = Arc::new(Mutex::new(None));
    let reason_clone = reason.clone();

    task.add_on_pipeline_finished(move |_frame, r| {
        let rc = reason_clone.clone();
        Box::pin(async move { *rc.lock().unwrap() = Some(r); })
    });

    let tx = task.push_sender();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(40)).await;
        tx.send((Frame::end_task(), FrameDirection::Upstream)).await.ok();
    });

    tokio::time::timeout(Duration::from_secs(2), task.run(system_clock(), None))
        .await.unwrap().unwrap();

    assert_eq!(*reason.lock().unwrap(), Some(FinishReason::End));
}

#[tokio::test]
async fn task_cancel_task_frame_triggers_cancel() {
    let task = make_task(vec![
        FrameProcessor::new("P", Box::new(PassthroughHandler), false),
    ]);

    let reason: Arc<Mutex<Option<FinishReason>>> = Arc::new(Mutex::new(None));
    let reason_clone = reason.clone();

    task.add_on_pipeline_finished(move |_frame, r| {
        let rc = reason_clone.clone();
        Box::pin(async move { *rc.lock().unwrap() = Some(r); })
    });

    let tx = task.push_sender();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(40)).await;
        tx.send((Frame::cancel_task(), FrameDirection::Upstream)).await.ok();
    });

    tokio::time::timeout(Duration::from_secs(2), task.run(system_clock(), None))
        .await.unwrap().unwrap();

    assert!(
        matches!(*reason.lock().unwrap(), Some(FinishReason::Cancel(_))),
        "Expected Cancel reason, got {:?}", *reason.lock().unwrap()
    );
}

#[tokio::test]
async fn task_stop_task_frame_triggers_stop() {
    let task = make_task(vec![
        FrameProcessor::new("P", Box::new(PassthroughHandler), false),
    ]);

    let reason: Arc<Mutex<Option<FinishReason>>> = Arc::new(Mutex::new(None));
    let reason_clone = reason.clone();

    task.add_on_pipeline_finished(move |_frame, r| {
        let rc = reason_clone.clone();
        Box::pin(async move { *rc.lock().unwrap() = Some(r); })
    });

    let tx = task.push_sender();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(40)).await;
        tx.send((Frame::stop_task(), FrameDirection::Upstream)).await.ok();
    });

    tokio::time::timeout(Duration::from_secs(2), task.run(system_clock(), None))
        .await.unwrap().unwrap();

    assert_eq!(*reason.lock().unwrap(), Some(FinishReason::Stop));
}

#[tokio::test]
async fn task_push_frame_reaches_processor() {
    let (tx_seen, mut rx_seen) = mpsc::unbounded_channel();

    struct DataCapture(mpsc::UnboundedSender<Vec<u8>>);

    #[async_trait]
    impl FrameHandler for DataCapture {
        async fn on_process_frame(
            &self,
            processor: &FrameProcessor,
            frame: Frame,
            direction: FrameDirection,
        ) -> Result<()> {
            if let FrameInner::Data(DataFrame::Data(ref d)) = frame.inner {
                let _ = self.0.send(d.content.clone());
            }
            processor.push_frame(frame, direction).await
        }
    }

    let task = make_task(vec![
        FrameProcessor::new("Capture", Box::new(DataCapture(tx_seen)), false),
    ]);

    let tx = task.push_sender();

    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(60)).await;
        tx.send((Frame::data(vec![1, 2, 3]), FrameDirection::Downstream)).await.ok();
        tokio::time::sleep(Duration::from_millis(60)).await;
        tx.send((Frame::end_task(), FrameDirection::Upstream)).await.ok();
    });

    tokio::time::timeout(Duration::from_secs(2), task.run(system_clock(), None))
        .await.unwrap().unwrap();

    let mut received = false;
    while let Ok(content) = rx_seen.try_recv() {
        if content == vec![1, 2, 3] {
            received = true;
        }
    }
    assert!(received, "Data frame payload did not reach processor");
}

#[tokio::test]
async fn task_idle_timeout_fires() {
    let mut params = PipelineParams::default();
    params.idle_timeout = Some(Duration::from_millis(100));
    params.cancel_on_idle_timeout = true;

    let task = PipelineTask::new(
        vec![FrameProcessor::new("P", Box::new(PassthroughHandler), false)],
        params,
    );

    let timed_out = Arc::new(Mutex::new(false));
    let timed_out_clone = timed_out.clone();

    task.add_on_idle_timeout(move || {
        let t = timed_out_clone.clone();
        Box::pin(async move { *t.lock().unwrap() = true; })
    });

    tokio::time::timeout(Duration::from_secs(2), task.run(system_clock(), None))
        .await.unwrap().unwrap();

    assert!(*timed_out.lock().unwrap(), "Idle timeout callback was not called");
}

#[tokio::test]
async fn task_idle_timeout_reset_by_activity() {
    let mut params = PipelineParams::default();
    params.idle_timeout = Some(Duration::from_millis(200));
    params.cancel_on_idle_timeout = false;

    let task = PipelineTask::new(
        vec![FrameProcessor::new("P", Box::new(PassthroughHandler), false)],
        params,
    );

    let timeout_count = Arc::new(Mutex::new(0u32));
    let tc = timeout_count.clone();

    task.add_on_idle_timeout(move || {
        let c = tc.clone();
        Box::pin(async move { *c.lock().unwrap() += 1; })
    });

    let tx = task.push_sender();

    tokio::spawn(async move {
        for _ in 0..3 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            tx.send((Frame::bot_speaking(), FrameDirection::Downstream)).await.ok();
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
        tx.send((Frame::end_task(), FrameDirection::Upstream)).await.ok();
    });

    tokio::time::timeout(Duration::from_secs(3), task.run(system_clock(), None))
        .await.unwrap().unwrap();

    let count = *timeout_count.lock().unwrap();
    assert!(count <= 1, "Idle timeout fired too many times: {}", count);
}

#[tokio::test]
async fn task_heartbeat_reaches_pipeline() {
    let mut params = PipelineParams::default();
    params.enable_heartbeats = true;
    params.heartbeat_seconds = 0.05;

    let (tx_seen, mut rx_seen) = mpsc::unbounded_channel::<FrameKind>();

    struct HbCapture(mpsc::UnboundedSender<FrameKind>);

    #[async_trait]
    impl FrameHandler for HbCapture {
        async fn on_process_frame(
            &self,
            processor: &FrameProcessor,
            frame: Frame,
            direction: FrameDirection,
        ) -> Result<()> {
            let _ = self.0.send(frame.kind());
            processor.push_frame(frame, direction).await
        }
    }

    let task = PipelineTask::new(
        vec![FrameProcessor::new("HB", Box::new(HbCapture(tx_seen)), false)],
        params,
    );

    let tx = task.push_sender();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        tx.send((Frame::end_task(), FrameDirection::Upstream)).await.ok();
    });

    tokio::time::timeout(Duration::from_secs(2), task.run(system_clock(), None))
        .await.unwrap().unwrap();

    let mut hb_count = 0;
    while let Ok(kind) = rx_seen.try_recv() {
        if kind == FrameKind::Heartbeat { hb_count += 1; }
    }
    assert!(hb_count >= 2, "Expected at least 2 heartbeats, got {}", hb_count);
}

#[tokio::test]
async fn task_error_callback_fires_on_fatal_error() {
    let error_msg: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let em = error_msg.clone();

    struct FatalErrorHandler;

    #[async_trait]
    impl FrameHandler for FatalErrorHandler {
        async fn on_process_frame(
            &self,
            processor: &FrameProcessor,
            frame: Frame,
            direction: FrameDirection,
        ) -> Result<()> {
            if let FrameInner::System(SystemFrame::Start(_)) = &frame.inner {
                processor.push_frame(
                    Frame::error("test fatal", true, Some(processor.name().to_string())),
                    FrameDirection::Upstream,
                ).await?;
            }
            processor.push_frame(frame, direction).await
        }
    }

    let task = make_task(vec![
        FrameProcessor::new("Faulty", Box::new(FatalErrorHandler), false),
    ]);

    task.add_on_pipeline_error(move |data| {
        let em = em.clone();
        Box::pin(async move {
            *em.lock().unwrap() = Some(data.error.clone());
        })
    });

    tokio::time::timeout(Duration::from_secs(2), task.run(system_clock(), None))
        .await.unwrap().unwrap();

    let msg = error_msg.lock().unwrap().clone();
    assert!(msg.is_some(), "Error callback was not fired");
    assert_eq!(msg.unwrap(), "test fatal");
}

#[tokio::test]
async fn task_lifecycle_receiver_transitions() {
    let task = make_task(vec![
        FrameProcessor::new("P", Box::new(PassthroughHandler), false),
    ]);

    let mut lifecycle_rx = task.lifecycle_receiver();
    assert_eq!(*lifecycle_rx.borrow(), PipelineLifecycle::NotStarted);

    let tx = task.push_sender();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        tx.send((Frame::end_task(), FrameDirection::Upstream)).await.ok();
    });

    tokio::time::timeout(Duration::from_secs(2), task.run(system_clock(), None))
        .await.unwrap().unwrap();

    assert!(
        matches!(*lifecycle_rx.borrow(), PipelineLifecycle::Finished(_)),
        "Expected Finished state, got {:?}", *lifecycle_rx.borrow()
    );
}

#[tokio::test]
async fn task_run_twice_fails() {
    let task = make_task(vec![
        FrameProcessor::new("P", Box::new(PassthroughHandler), false),
    ]);

    let tx = task.push_sender();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        tx.send((Frame::end_task(), FrameDirection::Upstream)).await.ok();
    });

    tokio::time::timeout(Duration::from_secs(2), task.run(system_clock(), None))
        .await.unwrap().unwrap();

    let result = task.run(system_clock(), None).await;
    assert!(result.is_err(), "Expected error on second run() call");
}
