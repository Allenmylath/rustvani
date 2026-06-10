//! PipelineTask — lifecycle management wrapper around a Pipeline.
//!
//! Topology (what PipelineTask builds internally):
//!
//! ```text
//!   push_frame() ──► mpsc ──► [TaskSource] → user_p1 → … → user_pN → [TaskSink]
//!                                  ↑                                        │
//!                            intercepts:                             intercepts:
//!                            EndTask → EndFrame ↓               StartFrame → on_started
//!                            CancelTask → CancelFrame ↓         EndFrame   → on_finished
//!                            StopTask → StopFrame ↓             StopFrame  → on_finished
//!                            InterruptionTask → interruption     CancelFrame→ lifecycle
//!                            BotSpeaking/UserSpeaking→idle reset  ErrorFrame → on_error
//! ```
//!
//! ## Event handlers
//!
//! Handlers are `Arc<dyn Fn(T) -> BoxFuture<'static, ()> + Send + Sync>` stored in
//! `Vec` behind `std::sync::Mutex`. The hot path checks `is_empty()` before locking
//! so frames that nobody is watching cost one atomic load and nothing else.
//!
//! Pattern (never hold Mutex across .await):
//! ```text
//! let cbs: Vec<_> = handlers.lock().unwrap().clone();  // clone Arc ptrs — cheap
//! drop(handlers);                                       // release before .await
//! for cb in &cbs { cb(arg.clone()).await; }
//! ```
//!
//! ## Lifecycle
//!
//! `tokio::sync::watch` is used ONLY for run/finish coordination between the task
//! loop and any external observers. It fires at most three times per session
//! (NotStarted → Running → Finished). Everything else uses direct callbacks.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use futures::future::BoxFuture;
use tokio::sync::{mpsc, watch, Notify};

use crate::billing::{BillingCollector, BillingEvent};
use crate::clock::BaseClock;

use crate::error::{PipecatError, Result};
use crate::frames::{ControlFrame, ErrorFrameData, Frame, FrameDirection, FrameHandler, FrameInner, FrameKind, FrameProcessor, FrameProcessorSetup, StartFrameData, SystemFrame};
use crate::observer::BaseObserver;


use super::pipeline::Pipeline;

// ---------------------------------------------------------------------------
// Callback type aliases
// ---------------------------------------------------------------------------

type AsyncCb0       = Arc<dyn Fn()                    -> BoxFuture<'static, ()> + Send + Sync>;
type AsyncCbFrame   = Arc<dyn Fn(Frame)               -> BoxFuture<'static, ()> + Send + Sync>;
type AsyncCbFinish  = Arc<dyn Fn(Frame, FinishReason) -> BoxFuture<'static, ()> + Send + Sync>;
type AsyncCbError   = Arc<dyn Fn(ErrorFrameData)      -> BoxFuture<'static, ()> + Send + Sync>;

// ---------------------------------------------------------------------------
// PipelineParams
// ---------------------------------------------------------------------------

/// Configuration for a `PipelineTask`.
#[derive(Clone)]
pub struct PipelineParams {
    // Pipeline-wide flags forwarded in StartFrame
    pub allow_interruptions: bool,
    pub enable_metrics: bool,
    pub enable_usage_metrics: bool,
    pub report_only_initial_ttfb: bool,

    // Heartbeat
    pub enable_heartbeats: bool,
    /// Interval between HeartbeatFrames pushed into the pipeline.
    pub heartbeat_seconds: f64,

    // Idle timeout
    /// Duration of silence before `on_idle_timeout` fires.
    /// `None` disables idle monitoring.
    pub idle_timeout: Option<Duration>,
    /// If `true`, an idle timeout sends a CancelFrame automatically.
    pub cancel_on_idle_timeout: bool,
    /// Frame kinds that reset the idle timer.
    /// Defaults are `BotSpeaking` and `UserSpeaking` — set explicitly to override.
    pub idle_timeout_frames: HashSet<FrameKind>,

    // Billing
    /// Optional billing collector. When set, `SessionStart` and `SessionEnd`
    /// events are emitted automatically at pipeline start and finish.
    pub billing_collector: Option<Arc<dyn BillingCollector>>,
    /// Key-value metadata forwarded in the `SessionStart` billing event.
    /// Useful for tagging sessions with user_id, tenant_id, phone_number, etc.
    pub billing_metadata: HashMap<String, String>,
}

impl Default for PipelineParams {
    fn default() -> Self {
        let mut idle_timeout_frames = HashSet::new();
        idle_timeout_frames.insert(FrameKind::BotSpeaking);
        idle_timeout_frames.insert(FrameKind::UserSpeaking);
        Self {
            allow_interruptions:      false,
            enable_metrics:           false,
            enable_usage_metrics:     false,
            report_only_initial_ttfb: false,
            enable_heartbeats:        false,
            heartbeat_seconds:        1.0,
            idle_timeout:             None,
            cancel_on_idle_timeout:   true,
            idle_timeout_frames,
            billing_collector:        None,
            billing_metadata:         HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// PipelineLifecycle / FinishReason
// ---------------------------------------------------------------------------

/// Why the pipeline finished.
#[derive(Debug, Clone, PartialEq)]
pub enum FinishReason {
    /// Graceful shutdown — EndFrame received.
    End,
    /// Stopped with connections kept alive (e.g. call hand-off) — StopFrame.
    Stop,
    /// Aborted — CancelFrame or explicit cancel.
    Cancel(Option<String>),
}

/// Observable lifecycle state of the pipeline.
/// Sent over a `tokio::sync::watch` channel — cheap to clone and observe.
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineLifecycle {
    NotStarted,
    Running,
    Finished(FinishReason),
}

// ---------------------------------------------------------------------------
// TaskState — interior shared between PipelineTask and its handlers
// ---------------------------------------------------------------------------

/// All state that both handler structs AND the run loop need to share.
/// Arc-wrapped so handlers can hold it without lifetime issues.
pub(crate) struct TaskState {
    // Lifecycle coordination — fires ≤3 times per session.
    pub(crate) lifecycle_tx: watch::Sender<PipelineLifecycle>,

    // Direct async callbacks — fast path when empty.
    pub(crate) on_pipeline_started:         std::sync::Mutex<Vec<AsyncCbFrame>>,
    pub(crate) on_pipeline_finished:        std::sync::Mutex<Vec<AsyncCbFinish>>,
    pub(crate) on_pipeline_error:           std::sync::Mutex<Vec<AsyncCbError>>,
    pub(crate) on_frame_reached_upstream:   std::sync::Mutex<Vec<AsyncCbFrame>>,
    pub(crate) on_frame_reached_downstream: std::sync::Mutex<Vec<AsyncCbFrame>>,
    pub(crate) on_idle_timeout:             std::sync::Mutex<Vec<AsyncCb0>>,

    // Per-direction frame filters for reached_upstream/downstream.
    pub(crate) upstream_filter:   std::sync::Mutex<HashSet<FrameKind>>,
    pub(crate) downstream_filter: std::sync::Mutex<HashSet<FrameKind>>,

    // Idle timer reset signal.
    pub(crate) idle_notify:         Arc<Notify>,
    pub(crate) idle_timeout_frames: HashSet<FrameKind>,
    pub(crate) cancel_on_idle_timeout: bool,

    pub(crate) cancelled: AtomicBool,
}

impl TaskState {
    fn new(params: &PipelineParams) -> (Arc<Self>, watch::Receiver<PipelineLifecycle>) {
        let (lifecycle_tx, lifecycle_rx) =
            watch::channel(PipelineLifecycle::NotStarted);
        let state = Arc::new(Self {
            lifecycle_tx,
            on_pipeline_started:         std::sync::Mutex::new(Vec::new()),
            on_pipeline_finished:        std::sync::Mutex::new(Vec::new()),
            on_pipeline_error:           std::sync::Mutex::new(Vec::new()),
            on_frame_reached_upstream:   std::sync::Mutex::new(Vec::new()),
            on_frame_reached_downstream: std::sync::Mutex::new(Vec::new()),
            on_idle_timeout:             std::sync::Mutex::new(Vec::new()),
            upstream_filter:             std::sync::Mutex::new(HashSet::new()),
            downstream_filter:           std::sync::Mutex::new(HashSet::new()),
            idle_notify:                 Arc::new(Notify::new()),
            idle_timeout_frames:         params.idle_timeout_frames.clone(),
            cancel_on_idle_timeout:      params.cancel_on_idle_timeout,
            cancelled:                   AtomicBool::new(false),
        });
        (state, lifecycle_rx)
    }
}

// ---------------------------------------------------------------------------
// TaskSourceHandler — intercepts frames arriving at the upstream boundary
// ---------------------------------------------------------------------------

/// Installed as the handler of the first processor in the task chain.
///
/// Downstream frames are normal pipeline entry — pass through.
/// Upstream frames have escaped the pipeline; many are task-control signals.
pub(crate) struct TaskSourceHandler {
    state: Arc<TaskState>,
}

#[async_trait]
impl FrameHandler for TaskSourceHandler {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match direction {
            FrameDirection::Downstream => {
                // Normal entry into the pipeline.
                processor.push_frame(frame, FrameDirection::Downstream).await
            }
            FrameDirection::Upstream => {
                // Frame has propagated past the entry point going upstream.
                // Intercept task-control frames; discard anything else
                // (source has no upstream neighbour).
                self.handle_upstream_escape(processor, frame).await
            }
        }
    }
}

impl TaskSourceHandler {
    async fn handle_upstream_escape(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
    ) -> Result<()> {
        match &frame.inner {
            // Task-control → convert to real lifecycle frame and push downstream.
            FrameInner::System(SystemFrame::EndTask { .. }) => {
                processor
                    .push_frame(Frame::end(), FrameDirection::Downstream)
                    .await?;
            }
            FrameInner::System(SystemFrame::CancelTask { .. }) => {
                processor
                    .push_frame(Frame::cancel(), FrameDirection::Downstream)
                    .await?;
            }
            FrameInner::System(SystemFrame::StopTask) => {
                processor
                    .push_frame(Frame::stop(), FrameDirection::Downstream)
                    .await?;
            }
            FrameInner::System(SystemFrame::InterruptionTask) => {
                processor.broadcast_interruption().await?;
            }
            FrameInner::System(SystemFrame::Error(ref data)) if data.fatal => {
                let data_clone = data.clone();
                fire_error(&self.state, data_clone).await;
                processor
                    .push_frame(Frame::cancel(), FrameDirection::Downstream)
                    .await?;
            }
            FrameInner::System(SystemFrame::Error(ref data)) => {
                let data_clone = data.clone();
                fire_error(&self.state, data_clone).await;
                processor.push_frame(frame, FrameDirection::Downstream).await?;
            }
            _ => {
                // Reset idle timer if configured.
                if self.state.idle_timeout_frames.contains(&frame.kind()) {
                    self.state.idle_notify.notify_one();
                }
                // Fire upstream-boundary callbacks if the frame kind is in the filter.
                let matches = self
                    .state
                    .upstream_filter
                    .lock()
                    .unwrap()
                    .contains(&frame.kind());
                if matches {
                    fire_frame_cbs(&self.state.on_frame_reached_upstream, frame).await;
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TaskSinkHandler — intercepts frames arriving at the downstream boundary
// ---------------------------------------------------------------------------

/// Installed as the handler of the last processor in the task chain.
///
/// Upstream frames propagate back through the pipeline — pass through.
/// Downstream frames have exited the pipeline; lifecycle frames are handled here.
pub(crate) struct TaskSinkHandler {
    state: Arc<TaskState>,
}

#[async_trait]
impl FrameHandler for TaskSinkHandler {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match direction {
            FrameDirection::Upstream => {
                // Normal upstream propagation.
                processor.push_frame(frame, FrameDirection::Upstream).await
            }
            FrameDirection::Downstream => {
                self.handle_downstream_escape(processor, frame).await
            }
        }
    }
}

impl TaskSinkHandler {
    async fn handle_downstream_escape(&self, processor: &FrameProcessor, frame: Frame) -> Result<()> {
        match &frame.inner {
            FrameInner::System(SystemFrame::Start(_)) => {
                let _ = self.state.lifecycle_tx.send(PipelineLifecycle::Running);
                fire_frame_cbs(&self.state.on_pipeline_started, frame.clone()).await;
                // Push through so PipelineSink also receives StartFrame and marks itself started.
                processor.push_frame(frame, FrameDirection::Downstream).await?;
            }
            FrameInner::Control(ControlFrame::End { .. }) => {
                fire_frame_finish_cbs(&self.state.on_pipeline_finished, frame.clone(), FinishReason::End).await;
                let _ = self.state.lifecycle_tx.send(PipelineLifecycle::Finished(FinishReason::End));
                processor.push_frame(frame, FrameDirection::Downstream).await?;
            }
            FrameInner::System(SystemFrame::Stop { .. }) => {
                fire_frame_finish_cbs(&self.state.on_pipeline_finished, frame.clone(), FinishReason::Stop).await;
                let _ = self.state.lifecycle_tx.send(PipelineLifecycle::Finished(FinishReason::Stop));
                processor.push_frame(frame, FrameDirection::Downstream).await?;
            }
            FrameInner::System(SystemFrame::Cancel { .. }) => {
                fire_frame_finish_cbs(&self.state.on_pipeline_finished, frame.clone(), FinishReason::Cancel(None)).await;
                let _ = self.state.lifecycle_tx.send(PipelineLifecycle::Finished(FinishReason::Cancel(None)));
                processor.push_frame(frame, FrameDirection::Downstream).await?;
            }
            FrameInner::System(SystemFrame::Error(ref data)) => {
                let data_clone = data.clone();
                fire_error(&self.state, data_clone).await;
                processor.push_frame(frame, FrameDirection::Downstream).await?;
            }
            FrameInner::System(
                SystemFrame::EndTask { .. }
                | SystemFrame::CancelTask { .. }
                | SystemFrame::StopTask
            ) => {
                log::warn!(
                    "TaskSink: task-control frame {} reached downstream boundary, ignoring",
                    frame.name()
                );
            }
            _ => {
                // Check idle timeout frames.
                if self.state.idle_timeout_frames.contains(&frame.kind()) {
                    self.state.idle_notify.notify_one();
                }
                // Check reached-downstream filter.
                let matches = self
                    .state
                    .downstream_filter
                    .lock()
                    .unwrap()
                    .contains(&frame.kind());
                if matches {
                    fire_frame_cbs(&self.state.on_frame_reached_downstream, frame.clone()).await;
                }
                processor.push_frame(frame, FrameDirection::Downstream).await?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helper: fire callbacks — never hold Mutex across .await
// ---------------------------------------------------------------------------

async fn fire_frame_cbs(
    handlers: &std::sync::Mutex<Vec<AsyncCbFrame>>,
    frame: Frame,
) {
    // Fast path: if nobody is listening, cost is one atomic read.
    if handlers.lock().unwrap().is_empty() {
        return;
    }
    let cbs: Vec<AsyncCbFrame> = handlers.lock().unwrap().clone();
    // Lock is dropped. Now await.
    for cb in &cbs {
        cb(frame.clone()).await;
    }
}

async fn fire_frame_finish_cbs(
    handlers: &std::sync::Mutex<Vec<AsyncCbFinish>>,
    frame: Frame,
    reason: FinishReason,
) {
    if handlers.lock().unwrap().is_empty() {
        return;
    }
    let cbs: Vec<AsyncCbFinish> = handlers.lock().unwrap().clone();
    for cb in &cbs {
        cb(frame.clone(), reason.clone()).await;
    }
}

async fn fire_error(state: &TaskState, data: ErrorFrameData) {
    if state.on_pipeline_error.lock().unwrap().is_empty() {
        return;
    }
    let cbs: Vec<AsyncCbError> = state.on_pipeline_error.lock().unwrap().clone();
    for cb in &cbs {
        cb(data.clone()).await;
    }
}

// ---------------------------------------------------------------------------
// PipelineTask
// ---------------------------------------------------------------------------

/// Manages the full lifecycle of a pipeline: setup, frame injection,
/// heartbeat, idle timeout, and graceful shutdown.
///
/// # Usage
///
/// ```text
/// let task = PipelineTask::new(vec![stt, llm, tts], PipelineParams::default());
///
/// // Register handlers before run().
/// task.add_on_pipeline_started(|frame| Box::pin(async move {
///     println!("pipeline started");
/// }));
///
/// // Get a sender for injecting external frames (e.g. from transport).
/// let tx = task.push_sender();
/// tokio::spawn(async move {
///     tx.send((Frame::data(audio_bytes), FrameDirection::Downstream)).await.ok();
/// });
///
/// // Block until the pipeline finishes.
/// task.run(clock, None).await?;
/// ```text
pub struct PipelineTask {
    /// The assembled pipeline (task_source → user processors → task_sink).
    pipeline: FrameProcessor,
    params: PipelineParams,
    state: Arc<TaskState>,

    /// External frame injection channel.
    /// Clone the sender with `push_sender()` before calling `run()`.
    push_tx: mpsc::Sender<(Frame, FrameDirection)>,
    /// Held in a Mutex so `run()` can take it without &mut self.
    push_rx: std::sync::Mutex<Option<mpsc::Receiver<(Frame, FrameDirection)>>>,

    /// Lifecycle observer — subscribe before run() to watch state transitions.
    lifecycle_rx: std::sync::Mutex<watch::Receiver<PipelineLifecycle>>,
}

impl PipelineTask {
    // ---- Construction ----

    /// Create a new task from an ordered list of user processors.
    ///
    /// Call `add_on_*` methods before `run()`, then call `run()`.
    pub fn new(processors: Vec<FrameProcessor>, params: PipelineParams) -> Self {
        let (state, lifecycle_rx) = TaskState::new(&params);

        // Build the chain with our task source and sink bookending the user procs.
        let task_source = FrameProcessor::new(
            "PipelineTaskSource",
            Box::new(TaskSourceHandler { state: state.clone() }),
            true, // direct mode — no task loops, processed inline
        );

        let task_sink = FrameProcessor::new(
            "PipelineTaskSink",
            Box::new(TaskSinkHandler { state: state.clone() }),
            true, // direct mode
        );

        let all: Vec<FrameProcessor> = std::iter::once(task_source)
            .chain(processors)
            .chain(std::iter::once(task_sink))
            .collect();

        let pipeline = Pipeline::new(all);

        let (push_tx, push_rx) = mpsc::channel(64);

        Self {
            pipeline,
            params,
            state,
            push_tx,
            push_rx: std::sync::Mutex::new(Some(push_rx)),
            lifecycle_rx: std::sync::Mutex::new(lifecycle_rx),
        }
    }

    // ---- External push ----

    /// Clone this sender and pass it to your transport to inject frames.
    pub fn push_sender(&self) -> mpsc::Sender<(Frame, FrameDirection)> {
        self.push_tx.clone()
    }

    /// Convenience: push a single frame directly (awaits channel send).
    pub async fn push_frame(&self, frame: Frame, direction: FrameDirection) -> Result<()> {
        self.push_tx
            .send((frame, direction))
            .await
            .map_err(|_| PipecatError::pipeline("Push channel closed"))
    }

    // ---- Lifecycle observer ----

    /// Subscribe to lifecycle transitions.
    /// Must be called before `run()`. Returns a cloned `watch::Receiver`.
    pub fn lifecycle_receiver(&self) -> watch::Receiver<PipelineLifecycle> {
        self.lifecycle_rx.lock().unwrap().clone()
    }

    // ---- Callback registration ----

    /// Called once when `StartFrame` exits the downstream end.
    pub fn add_on_pipeline_started<F>(&self, f: F)
    where
        F: Fn(Frame) -> BoxFuture<'static, ()> + Send + Sync + 'static,
    {
        self.state
            .on_pipeline_started
            .lock()
            .unwrap()
            .push(Arc::new(f));
    }

    /// Called when `EndFrame` or `StopFrame` exits the downstream end.
    pub fn add_on_pipeline_finished<F>(&self, f: F)
    where
        F: Fn(Frame, FinishReason) -> BoxFuture<'static, ()> + Send + Sync + 'static,
    {
        self.state
            .on_pipeline_finished
            .lock()
            .unwrap()
            .push(Arc::new(f));
    }

    /// Called when an `ErrorFrame` reaches either boundary.
    pub fn add_on_pipeline_error<F>(&self, f: F)
    where
        F: Fn(ErrorFrameData) -> BoxFuture<'static, ()> + Send + Sync + 'static,
    {
        self.state
            .on_pipeline_error
            .lock()
            .unwrap()
            .push(Arc::new(f));
    }

    /// Called when a frame matching the upstream filter reaches the upstream boundary.
    pub fn add_on_frame_reached_upstream<F>(&self, f: F)
    where
        F: Fn(Frame) -> BoxFuture<'static, ()> + Send + Sync + 'static,
    {
        self.state
            .on_frame_reached_upstream
            .lock()
            .unwrap()
            .push(Arc::new(f));
    }

    /// Called when a frame matching the downstream filter reaches the downstream boundary.
    pub fn add_on_frame_reached_downstream<F>(&self, f: F)
    where
        F: Fn(Frame) -> BoxFuture<'static, ()> + Send + Sync + 'static,
    {
        self.state
            .on_frame_reached_downstream
            .lock()
            .unwrap()
            .push(Arc::new(f));
    }

    /// Called when no idle-resetting frame arrives within `idle_timeout`.
    pub fn add_on_idle_timeout<F>(&self, f: F)
    where
        F: Fn() -> BoxFuture<'static, ()> + Send + Sync + 'static,
    {
        self.state
            .on_idle_timeout
            .lock()
            .unwrap()
            .push(Arc::new(f));
    }

    // ---- Filter configuration ----

    /// Only fire `on_frame_reached_upstream` for frames with these kinds.
    pub fn set_upstream_filter(&self, kinds: HashSet<FrameKind>) {
        *self.state.upstream_filter.lock().unwrap() = kinds;
    }

    /// Only fire `on_frame_reached_downstream` for frames with these kinds.
    pub fn set_downstream_filter(&self, kinds: HashSet<FrameKind>) {
        *self.state.downstream_filter.lock().unwrap() = kinds;
    }

    // ---- Run ----

    /// Setup and run the pipeline to completion.
    ///
    /// Blocks until the pipeline reaches `Finished` (via EndFrame, StopFrame,
    /// CancelFrame, or an explicit cancel). Call `push_sender()` BEFORE `run()`
    /// to get the frame injection handle.
    pub async fn run(
        &self,
        clock: Arc<dyn BaseClock>,
        observer: Option<Arc<dyn BaseObserver>>,
    ) -> Result<()> {
        // Take the push receiver (can only run once).
        let push_rx = self
            .push_rx
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| PipecatError::pipeline("PipelineTask::run() called more than once"))?;

        // Setup the pipeline.
        self.pipeline
            .setup(FrameProcessorSetup { clock, observer })
            .await?;

        // Register billing lifecycle hooks when a collector is configured.
        if let Some(bc) = self.params.billing_collector.clone() {
            let bc_start = bc.clone();
            let meta = self.params.billing_metadata.clone();
            self.add_on_pipeline_started(move |_frame| {
                let bc = bc_start.clone();
                let m  = meta.clone();
                Box::pin(async move {
                    bc.record(BillingEvent::SessionStart {
                        session_id: bc.session_id(),
                        started_at: Utc::now(),
                        metadata:   m,
                    });
                })
            });

            let bc_end = bc.clone();
            self.add_on_pipeline_finished(move |_frame, reason| {
                let bc = bc_end.clone();
                let finish = match &reason {
                    FinishReason::End        => "end",
                    FinishReason::Stop       => "stop",
                    FinishReason::Cancel(_)  => "cancel",
                }
                .to_string();
                Box::pin(async move {
                    bc.record(BillingEvent::SessionEnd {
                        session_id:    bc.session_id(),
                        ended_at:      Utc::now(),
                        finish_reason: finish,
                    });
                })
            });
        }

        // Push StartFrame — this initialises all processors.
        let start_frame = Frame::start(StartFrameData {
            allow_interruptions:      self.params.allow_interruptions,
            enable_metrics:           self.params.enable_metrics,
            enable_usage_metrics:     self.params.enable_usage_metrics,
            report_only_initial_ttfb: self.params.report_only_initial_ttfb,
            ..Default::default()
        });
        self.pipeline
            .queue_frame(start_frame, FrameDirection::Downstream, None)
            .await?;

        // Spawn push-queue processor: reads external frames and feeds the pipeline.
        let pipeline_for_push = self.pipeline.clone();
        let push_task = tokio::spawn(async move {
            let mut rx = push_rx;
            while let Some((frame, direction)) = rx.recv().await {
                let _ = pipeline_for_push
                    .queue_frame(frame, direction, None)
                    .await;
            }
        });

        // Spawn heartbeat task (optional).
        let heartbeat_task = if self.params.enable_heartbeats {
            let pipeline_hb = self.pipeline.clone();
            let period = Duration::from_secs_f64(self.params.heartbeat_seconds);
            Some(tokio::spawn(async move {
                loop {
                    tokio::time::sleep(period).await;
                    let ts = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs_f64();
                    let _ = pipeline_hb
                        .queue_frame(
                            Frame::heartbeat(ts),
                            FrameDirection::Downstream,
                            None,
                        )
                        .await;
                }
            }))
        } else {
            None
        };

        // Spawn idle monitor (optional).
        let idle_task = if let Some(timeout) = self.params.idle_timeout {
            let state = self.state.clone();
            let pipeline_idle = self.pipeline.clone();
            Some(tokio::spawn(async move {
                let idle_notify = state.idle_notify.clone();
                loop {
                    // Sleep for `timeout`; if idle_notify fires first, reset.
                    let timed_out = tokio::time::timeout(
                        timeout,
                        idle_notify.notified(),
                    )
                    .await
                    .is_err(); // err = timeout elapsed

                    if timed_out {
                        // Fire idle timeout callbacks.
                        let cbs: Vec<AsyncCb0> =
                            state.on_idle_timeout.lock().unwrap().clone();
                        for cb in &cbs {
                            cb().await;
                        }

                        if state.cancel_on_idle_timeout {
                            log::info!("PipelineTask: idle timeout — cancelling pipeline");
                            let _ = pipeline_idle
                                .queue_frame(
                                    Frame::cancel(),
                                    FrameDirection::Downstream,
                                    None,
                                )
                                .await;
                            break;
                        }
                        // If not auto-cancelling, keep monitoring.
                    }
                    // Got notified (activity detected) — restart the timer.
                }
            }))
        } else {
            None
        };

        // Wait for the pipeline to signal Finished.
        let mut lifecycle_rx = self.lifecycle_rx.lock().unwrap().clone();
        loop {
            lifecycle_rx
                .changed()
                .await
                .map_err(|_| PipecatError::pipeline("Lifecycle channel closed unexpectedly"))?;

            if matches!(
                *lifecycle_rx.borrow(),
                PipelineLifecycle::Finished(_)
            ) {
                break;
            }
        }

        // Shut down background tasks.
        push_task.abort();
        if let Some(t) = heartbeat_task { t.abort(); }
        if let Some(t) = idle_task       { t.abort(); }

        // Cleanup all processors.
        self.pipeline.cleanup().await?;

        Ok(())
    }
}
