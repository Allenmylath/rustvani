use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use async_trait::async_trait;
use futures::future::BoxFuture;
use tokio::sync::{Mutex, Notify, RwLock};
use tokio::task::JoinHandle;

use crate::clock::BaseClock;
use crate::direction::FrameDirection;
use crate::error::Result;
use crate::frames::{ErrorFrameData, Frame, PauseResumeFrameData, StartFrameData, next_frame_id};
use crate::metrics::{FrameProcessorMetrics, LLMTokenUsage};
use crate::observer::{BaseObserver, FrameProcessed, FramePushed};
use crate::queue::{FrameProcessorQueue, ProcessQueue, QueueCallback};

// ---------------------------------------------------------------------------
// Public callback type (user-facing API to queue_frame)
// ---------------------------------------------------------------------------

/// A callback invoked after a frame has been fully processed.
/// Receives the processor that handled it, the frame, and its direction.
pub type FrameCallback = Box<
    dyn FnOnce(FrameProcessor, Frame, FrameDirection) -> BoxFuture<'static, ()> + Send,
>;

// ---------------------------------------------------------------------------
// FrameProcessorSetup
// ---------------------------------------------------------------------------

/// Configuration passed to `FrameProcessor::setup()`.
pub struct FrameProcessorSetup {
    pub clock: Arc<dyn BaseClock>,
    pub observer: Option<Arc<dyn BaseObserver>>,
}

// ---------------------------------------------------------------------------
// FrameHandler trait — the override point (equivalent to Python's `process_frame`)
// ---------------------------------------------------------------------------

/// Implement this trait to define custom per-processor frame handling logic.
///
/// `on_process_frame` is called for **every** frame (system and non-system)
/// after the base infrastructure has handled system-level concerns.
///
/// Implementations that want to pass frames through must call
/// `processor.push_frame(frame, direction).await`.
#[async_trait]
pub trait FrameHandler: Send + Sync {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()>;

    /// Whether this handler can generate metrics (mirrors Python's `can_generate_metrics`).
    fn can_generate_metrics(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// PassthroughHandler — default: push every frame unchanged
// ---------------------------------------------------------------------------

pub struct PassthroughHandler;

#[async_trait]
impl FrameHandler for PassthroughHandler {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        processor.push_frame(frame, direction).await
    }
}

// ---------------------------------------------------------------------------
// Event-handler type aliases
// ---------------------------------------------------------------------------

type FrameEventFn = Box<dyn Fn(&Frame) + Send + Sync>;
type ErrorEventFn = Box<dyn Fn(&ErrorFrameData) + Send + Sync>;

// ---------------------------------------------------------------------------
// Global processor ID counter
// ---------------------------------------------------------------------------

static PROCESSOR_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_processor_id() -> u64 {
    PROCESSOR_ID_COUNTER.fetch_add(1, Ordering::SeqCst)
}

// Timeout used when cancelling the input frame task (mirrors Python's INPUT_TASK_CANCEL_TIMEOUT_SECS = 3).
const INPUT_TASK_CANCEL_TIMEOUT_SECS: f64 = 3.0;

// ---------------------------------------------------------------------------
// Inner — all shared mutable state, accessed via Arc
// ---------------------------------------------------------------------------

struct Inner {
    name: String,
    id: u64,

    // Linked-list links (prev uses Weak to avoid reference cycles)
    prev: RwLock<Option<Weak<Inner>>>,
    next: RwLock<Option<Arc<Inner>>>,

    // ---- Queues ----
    /// Priority queue for all incoming frames (system frames first).
    input_queue: FrameProcessorQueue,
    /// FIFO queue for non-system frames awaiting processing.
    process_queue: ProcessQueue,

    // ---- Atomic flags ----
    cancelling: AtomicBool,
    started: AtomicBool,
    should_block_system_frames: AtomicBool,
    should_block_frames: AtomicBool,

    // ---- Async events (tokio::sync::Notify) ----
    /// Signals the input task to unblock after system-frame blocking is cleared.
    input_event: Notify,
    /// Signals the process task to unblock after frame blocking is cleared.
    process_event: Notify,

    // ---- Task handles (std Mutex — never held across .await) ----
    input_task: std::sync::Mutex<Option<JoinHandle<()>>>,
    process_task: std::sync::Mutex<Option<JoinHandle<()>>>,

    /// The frame currently being processed by the process task (for interruption logic).
    process_current_frame: Mutex<Option<Frame>>,

    // ---- State flags set from StartFrame ----
    allow_interruptions: AtomicBool,
    enable_metrics: AtomicBool,
    enable_usage_metrics: AtomicBool,
    report_only_initial_ttfb: AtomicBool,
    deprecated_openaillmcontext: AtomicBool,

    // ---- Infrastructure ----
    clock: RwLock<Option<Arc<dyn BaseClock>>>,
    observer: RwLock<Option<Arc<dyn BaseObserver>>>,

    // ---- Event handlers (sync, matching Python's `sync=True`) ----
    on_before_process_frame: Mutex<Vec<FrameEventFn>>,
    on_after_process_frame:  Mutex<Vec<FrameEventFn>>,
    on_before_push_frame:    Mutex<Vec<FrameEventFn>>,
    on_after_push_frame:     Mutex<Vec<FrameEventFn>>,
    on_error:                Mutex<Vec<ErrorEventFn>>,

    // ---- Metrics ----
    metrics: FrameProcessorMetrics,

    // ---- User-supplied frame handler ----
    handler: Box<dyn FrameHandler>,

    // ---- Direct mode: skip queues and process immediately ----
    enable_direct_mode: bool,
}

// ---------------------------------------------------------------------------
// FrameProcessor — the public newtype wrapper around Arc<Inner>
// ---------------------------------------------------------------------------

/// A single node in a Pipecat pipeline.
///
/// Processors are cloneable handles to shared state (`Arc<Inner>`).
/// Use `link()` to connect processors and `setup()` to start the internal tasks.
#[derive(Clone)]
pub struct FrameProcessor(Arc<Inner>);

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

impl FrameProcessor {
    /// Create a new processor.
    ///
    /// # Arguments
    /// * `name`               — human-readable name shown in logs.
    /// * `handler`            — custom frame processing logic (or `PassthroughHandler`).
    /// * `enable_direct_mode` — skip internal queues and process frames synchronously
    ///                          in the caller's task (no parallelism, low latency).
    pub fn new(
        name: impl Into<String>,
        handler: Box<dyn FrameHandler>,
        enable_direct_mode: bool,
    ) -> Self {
        let name = name.into();
        let id = next_processor_id();
        let metrics = FrameProcessorMetrics::new();
        metrics.set_processor_name(&name);

        FrameProcessor(Arc::new(Inner {
            name,
            id,
            prev: RwLock::new(None),
            next: RwLock::new(None),
            input_queue: FrameProcessorQueue::new(),
            process_queue: ProcessQueue::new(),
            cancelling: AtomicBool::new(false),
            started: AtomicBool::new(false),
            should_block_system_frames: AtomicBool::new(false),
            should_block_frames: AtomicBool::new(false),
            input_event: Notify::new(),
            process_event: Notify::new(),
            input_task: std::sync::Mutex::new(None),
            process_task: std::sync::Mutex::new(None),
            process_current_frame: Mutex::new(None),
            allow_interruptions: AtomicBool::new(false),
            enable_metrics: AtomicBool::new(false),
            enable_usage_metrics: AtomicBool::new(false),
            report_only_initial_ttfb: AtomicBool::new(false),
            deprecated_openaillmcontext: AtomicBool::new(false),
            clock: RwLock::new(None),
            observer: RwLock::new(None),
            on_before_process_frame: Mutex::new(Vec::new()),
            on_after_process_frame:  Mutex::new(Vec::new()),
            on_before_push_frame:    Mutex::new(Vec::new()),
            on_after_push_frame:     Mutex::new(Vec::new()),
            on_error:                Mutex::new(Vec::new()),
            metrics,
            handler,
            enable_direct_mode,
        }))
    }
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

impl FrameProcessor {
    pub fn id(&self) -> u64 {
        self.0.id
    }

    pub fn name(&self) -> &str {
        &self.0.name
    }

    /// Returns the list of sub-processors (non-compound processors return empty).
    pub fn processors(&self) -> Vec<FrameProcessor> {
        vec![]
    }

    /// Returns entry processors for compound processors (empty for base).
    pub fn entry_processors(&self) -> Vec<FrameProcessor> {
        vec![]
    }

    pub async fn next(&self) -> Option<FrameProcessor> {
        self.0
            .next
            .read()
            .await
            .as_ref()
            .map(|a| FrameProcessor(a.clone()))
    }

    pub async fn previous(&self) -> Option<FrameProcessor> {
        self.0
            .prev
            .read()
            .await
            .as_ref()
            .and_then(|w| w.upgrade())
            .map(FrameProcessor)
    }

    pub fn metrics_enabled(&self) -> bool {
        self.0.enable_metrics.load(Ordering::Relaxed)
    }

    pub fn usage_metrics_enabled(&self) -> bool {
        self.0.enable_usage_metrics.load(Ordering::Relaxed)
    }

    pub fn report_only_initial_ttfb(&self) -> bool {
        self.0.report_only_initial_ttfb.load(Ordering::Relaxed)
    }

    pub fn interruptions_allowed(&self) -> bool {
        self.0.allow_interruptions.load(Ordering::Relaxed)
    }

    /// Whether this processor can generate metrics (delegates to the handler).
    pub fn can_generate_metrics(&self) -> bool {
        self.0.handler.can_generate_metrics()
    }

    /// Collect all processors (recursively) that can generate metrics.
    pub fn processors_with_metrics(&self) -> Vec<FrameProcessor> {
        vec![]
    }
}

// ---------------------------------------------------------------------------
// Lifecycle: setup / cleanup
// ---------------------------------------------------------------------------

impl FrameProcessor {
    /// Initialise this processor with a clock and optional observer,
    /// then start the internal input-frame task.
    pub async fn setup(&self, setup: FrameProcessorSetup) -> Result<()> {
        *self.0.clock.write().await = Some(setup.clock);
        *self.0.observer.write().await = setup.observer;

        if !self.0.enable_direct_mode {
            self.create_input_task();
        }

        Ok(())
    }

    /// Shut down both internal tasks and release resources.
    pub async fn cleanup(&self) -> Result<()> {
        self.cancel_input_task().await;
        self.cancel_process_task().await;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Linking
// ---------------------------------------------------------------------------

impl FrameProcessor {
    /// Connect this processor to `next` in the pipeline.
    /// Also sets `next`'s `prev` link back to `self`.
    pub async fn link(&self, next: FrameProcessor) {
        log::debug!("Linking {} -> {}", self.name(), next.name());
        *self.0.next.write().await = Some(next.0.clone());
        *next.0.prev.write().await = Some(Arc::downgrade(&self.0));
    }
}

// ---------------------------------------------------------------------------
// Frame queuing — public entry point
// ---------------------------------------------------------------------------

impl FrameProcessor {
    /// Queue a frame for processing.
    ///
    /// If the processor is currently cancelling, the frame is silently dropped.
    /// An optional `callback` is invoked after the frame has been processed.
    pub async fn queue_frame(
        &self,
        frame: Frame,
        direction: FrameDirection,
        callback: Option<FrameCallback>,
    ) -> Result<()> {
        if self.0.cancelling.load(Ordering::Relaxed) {
            return Ok(());
        }

        // Wrap the user FrameCallback into a pre-bound QueueCallback.
        let queue_cb: Option<QueueCallback> = callback.map(|cb| {
            let proc = self.clone();
            let f = frame.clone();
            let d = direction;
            let boxed: QueueCallback =
                Box::new(move || -> BoxFuture<'static, ()> { Box::pin(async move { cb(proc, f, d).await }) });
            boxed
        });

        if self.0.enable_direct_mode {
            self.internal_process_frame(frame, direction, queue_cb).await;
        } else {
            self.0.input_queue.put((frame, direction, queue_cb)).await;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Frame blocking / unblocking
// ---------------------------------------------------------------------------

impl FrameProcessor {
    /// Block non-system frames from being processed (they accumulate in the queue).
    pub async fn pause_processing_frames(&self) {
        log::trace!("{}: pausing frame processing", self.name());
        self.0.should_block_frames.store(true, Ordering::Relaxed);
        // The process_event is NOT set — the task will wait on it.
    }

    /// Resume processing non-system frames.
    pub async fn resume_processing_frames(&self) {
        log::trace!("{}: resuming frame processing", self.name());
        self.0.should_block_frames.store(false, Ordering::Relaxed);
        self.0.process_event.notify_one();
    }

    /// Block system frames (stall the input task).
    pub async fn pause_processing_system_frames(&self) {
        log::trace!("{}: pausing system frame processing", self.name());
        self.0
            .should_block_system_frames
            .store(true, Ordering::Relaxed);
    }

    /// Resume system frame processing.
    pub async fn resume_processing_system_frames(&self) {
        log::trace!("{}: resuming system frame processing", self.name());
        self.0
            .should_block_system_frames
            .store(false, Ordering::Relaxed);
        self.0.input_event.notify_one();
    }
}

// ---------------------------------------------------------------------------
// process_frame — the override point + system-frame handler
// ---------------------------------------------------------------------------

impl FrameProcessor {
    /// Handle a frame.
    ///
    /// System frames are handled here (start, cancel, interruption, pause/resume).
    /// After system handling the user's [`FrameHandler::on_process_frame`] is called.
    ///
    /// This is the Rust equivalent of Python's `FrameProcessor.process_frame`.
    pub async fn process_frame(
        &self,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        // --- Observer notification ---
        if let Some(obs) = self.0.observer.read().await.as_ref() {
            let ts = self.get_time();
            obs.on_process_frame(FrameProcessed {
                processor_name: self.0.name.clone(),
                frame: frame.clone(),
                direction,
                timestamp: ts,
            })
            .await;
        }

        // --- System-frame handling ---
        match &frame {
            Frame::Start(_, data) => {
                self.handle_start(data.clone()).await;
            }
            Frame::Interruption(_, _) => {
                self.start_interruption().await?;
                self.stop_all_metrics().await;
            }
            Frame::Cancel(_, _) => {
                self.handle_cancel().await;
            }
            Frame::PauseProcessing(_, data) => {
                self.handle_pause(data.clone(), false).await;
            }
            Frame::PauseProcessingUrgent(_, data) => {
                self.handle_pause(data.clone(), true).await;
            }
            Frame::ResumeProcessing(_, data) => {
                self.handle_resume(data.clone(), false).await;
            }
            Frame::ResumeProcessingUrgent(_, data) => {
                self.handle_resume(data.clone(), true).await;
            }
            _ => {}
        }

        // --- User handler ---
        self.0.handler.on_process_frame(self, frame, direction).await
    }
}

// ---------------------------------------------------------------------------
// push_frame — propagate a frame to the adjacent processor
// ---------------------------------------------------------------------------

impl FrameProcessor {
    /// Push a frame downstream (or upstream) to the adjacent processor.
    ///
    /// Drops the frame silently if `StartFrame` has not yet been received.
    pub async fn push_frame(
        &self,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        if !self.check_started(&frame) {
            return Ok(());
        }

        // Before-push event handlers
        {
            let handlers = self.0.on_before_push_frame.lock().await;
            for h in handlers.iter() {
                h(&frame);
            }
        }

        self.internal_push_frame(frame.clone(), direction).await?;

        // After-push event handlers
        {
            let handlers = self.0.on_after_push_frame.lock().await;
            for h in handlers.iter() {
                h(&frame);
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// push_error / push_error_frame
// ---------------------------------------------------------------------------

impl FrameProcessor {
    /// Create and push an `ErrorFrame` upstream.
    pub async fn push_error(
        &self,
        error_msg: impl Into<String>,
        fatal: bool,
    ) -> Result<()> {
        let data = ErrorFrameData {
            error: error_msg.into(),
            fatal,
            processor_name: Some(self.0.name.clone()),
            broadcast_sibling_id: None,
        };
        self.push_error_frame(data).await
    }

    /// Push a pre-constructed `ErrorFrameData` upstream (calls `on_error` handlers first).
    pub async fn push_error_frame(&self, mut error: ErrorFrameData) -> Result<()> {
        if error.processor_name.is_none() {
            error.processor_name = Some(self.0.name.clone());
        }

        // Fire on_error event handlers
        {
            let handlers = self.0.on_error.lock().await;
            for h in handlers.iter() {
                h(&error);
            }
        }

        log::error!(
            "{} error: {}",
            error.processor_name.as_deref().unwrap_or("unknown"),
            error.error
        );

        let frame = Frame::Error(next_frame_id(), error);
        self.internal_push_frame(frame, FrameDirection::Upstream).await
    }
}

// ---------------------------------------------------------------------------
// Broadcast helpers
// ---------------------------------------------------------------------------

impl FrameProcessor {
    /// Broadcast a frame both downstream and upstream.
    ///
    /// Two fresh copies of `template` are created (new IDs, sibling IDs cross-linked),
    /// then pushed in each direction.
    pub async fn broadcast_frame(&self, template: Frame) -> Result<()> {
        let mut downstream = template.clone().with_new_id();
        let mut upstream = template.with_new_id();

        let ds_id = downstream.id();
        let us_id = upstream.id();

        downstream = downstream.with_sibling(us_id);
        upstream = upstream.with_sibling(ds_id);

        self.push_frame(downstream, FrameDirection::Downstream).await?;
        self.push_frame(upstream, FrameDirection::Upstream).await
    }

    /// Broadcast an existing frame instance (shallow-clone, fresh IDs, cross-linked siblings).
    pub async fn broadcast_frame_instance(&self, frame: Frame) -> Result<()> {
        self.broadcast_frame(frame).await
    }

    /// Broadcast an `InterruptionFrame`, reset the process task, and stop metrics.
    pub async fn broadcast_interruption(&self) -> Result<()> {
        log::debug!("{}: broadcasting interruption", self.name());
        self.reset_process_task().await;
        self.stop_all_metrics().await;
        self.broadcast_frame(Frame::new_interruption()).await
    }
}

// ---------------------------------------------------------------------------
// Metrics forwarding
// ---------------------------------------------------------------------------

impl FrameProcessor {
    pub async fn start_ttfb_metrics(&self, start_time: Option<f64>) {
        if self.can_generate_metrics() && self.metrics_enabled() {
            self.0
                .metrics
                .start_ttfb_metrics(start_time, self.report_only_initial_ttfb())
                .await;
        }
    }

    pub async fn stop_ttfb_metrics(&self, end_time: Option<f64>) {
        if self.can_generate_metrics() && self.metrics_enabled() {
            if let Some(frame) = self.0.metrics.stop_ttfb_metrics(end_time).await {
                let _ = Box::pin(self.push_frame(frame, FrameDirection::Downstream)).await;
            }
        }
    }

    pub async fn start_processing_metrics(&self, start_time: Option<f64>) {
        if self.can_generate_metrics() && self.metrics_enabled() {
            self.0.metrics.start_processing_metrics(start_time).await;
        }
    }

    pub async fn stop_processing_metrics(&self, end_time: Option<f64>) {
        if self.can_generate_metrics() && self.metrics_enabled() {
            if let Some(frame) = self.0.metrics.stop_processing_metrics(end_time).await {
                let _ = Box::pin(self.push_frame(frame, FrameDirection::Downstream)).await;
            }
        }
    }

    pub async fn start_llm_usage_metrics(&self, tokens: &LLMTokenUsage) {
        if self.can_generate_metrics() && self.usage_metrics_enabled() {
            if let Some(frame) = self.0.metrics.start_llm_usage_metrics(tokens).await {
                let _ = Box::pin(self.push_frame(frame, FrameDirection::Downstream)).await;
            }
        }
    }

    pub async fn start_tts_usage_metrics(&self, text: &str) {
        if self.can_generate_metrics() && self.usage_metrics_enabled() {
            if let Some(frame) = self.0.metrics.start_tts_usage_metrics(text).await {
                let _ = Box::pin(self.push_frame(frame, FrameDirection::Downstream)).await;
            }
        }
    }

    pub async fn start_text_aggregation_metrics(&self) {
        if self.can_generate_metrics() && self.metrics_enabled() {
            self.0.metrics.start_text_aggregation_metrics().await;
        }
    }

    pub async fn stop_text_aggregation_metrics(&self) {
        if self.can_generate_metrics() && self.metrics_enabled() {
            if let Some(frame) = self.0.metrics.stop_text_aggregation_metrics().await {
                let _ = Box::pin(self.push_frame(frame, FrameDirection::Downstream)).await;
            }
        }
    }

    pub async fn stop_all_metrics(&self) {
        self.stop_ttfb_metrics(None).await;
        self.stop_processing_metrics(None).await;
        self.stop_text_aggregation_metrics().await;
    }
}

// ---------------------------------------------------------------------------
// Event-handler registration
// ---------------------------------------------------------------------------

impl FrameProcessor {
    /// Register a callback for `on_before_process_frame`.
    pub async fn on_before_process_frame<F>(&self, f: F)
    where
        F: Fn(&Frame) + Send + Sync + 'static,
    {
        self.0.on_before_process_frame.lock().await.push(Box::new(f));
    }

    /// Register a callback for `on_after_process_frame`.
    pub async fn on_after_process_frame<F>(&self, f: F)
    where
        F: Fn(&Frame) + Send + Sync + 'static,
    {
        self.0.on_after_process_frame.lock().await.push(Box::new(f));
    }

    /// Register a callback for `on_before_push_frame`.
    pub async fn on_before_push_frame<F>(&self, f: F)
    where
        F: Fn(&Frame) + Send + Sync + 'static,
    {
        self.0.on_before_push_frame.lock().await.push(Box::new(f));
    }

    /// Register a callback for `on_after_push_frame`.
    pub async fn on_after_push_frame<F>(&self, f: F)
    where
        F: Fn(&Frame) + Send + Sync + 'static,
    {
        self.0.on_after_push_frame.lock().await.push(Box::new(f));
    }

    /// Register a callback for `on_error`.
    pub async fn on_error<F>(&self, f: F)
    where
        F: Fn(&ErrorFrameData) + Send + Sync + 'static,
    {
        self.0.on_error.lock().await.push(Box::new(f));
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

impl FrameProcessor {
    fn get_time(&self) -> f64 {
        // Try synchronous read — if clock isn't set yet return 0.0
        if let Ok(guard) = self.0.clock.try_read() {
            if let Some(clk) = guard.as_ref() {
                return clk.get_time();
            }
        }
        0.0
    }

    fn check_started(&self, frame: &Frame) -> bool {
        if !self.0.started.load(Ordering::Relaxed) {
            log::error!(
                "{} trying to push {} but StartFrame not received yet",
                self.name(),
                frame.name()
            );
            return false;
        }
        true
    }

    // ---- System-frame sub-handlers ----

    async fn handle_start(&self, data: StartFrameData) {
        self.0.started.store(true, Ordering::Relaxed);
        self.0
            .allow_interruptions
            .store(data.allow_interruptions, Ordering::Relaxed);
        self.0
            .enable_metrics
            .store(data.enable_metrics, Ordering::Relaxed);
        self.0
            .enable_usage_metrics
            .store(data.enable_usage_metrics, Ordering::Relaxed);
        self.0
            .report_only_initial_ttfb
            .store(data.report_only_initial_ttfb, Ordering::Relaxed);
        self.0
            .deprecated_openaillmcontext
            .store(
                data.metadata.contains_key("deprecated_openaillmcontext"),
                Ordering::Relaxed,
            );

        if !self.0.enable_direct_mode {
            self.create_process_task();
        }
    }

    async fn handle_cancel(&self) {
        self.0.cancelling.store(true, Ordering::Relaxed);
        self.cancel_process_task().await;
    }

    async fn handle_pause(&self, data: PauseResumeFrameData, _urgent: bool) {
        if data.processor_name == self.0.name {
            self.pause_processing_frames().await;
        }
    }

    async fn handle_resume(&self, data: PauseResumeFrameData, _urgent: bool) {
        if data.processor_name == self.0.name {
            self.resume_processing_frames().await;
        }
    }

    // ---- Interruption ----

    /// Drain non-uninterruptible frames from the process queue.
    /// Exposed publicly for testing; production code triggers this via `broadcast_interruption`.
    pub async fn drain_process_queue(&self) {
        self.reset_process_queue().await;
    }

    /// Handle an interruption: if the current frame is uninterruptible, only drain the queue.
    /// Otherwise cancel and recreate the process task.
    pub async fn start_interruption(&self) -> Result<()> {
        let current = self.0.process_current_frame.lock().await.clone();
        match current {
            Some(f) if f.is_uninterruptible() => {
                self.reset_process_queue().await;
            }
            _ => {
                self.cancel_process_task().await;
                self.create_process_task();
            }
        }
        Ok(())
    }

    // ---- Internal push ----

    async fn internal_push_frame(
        &self,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        let ts = self.get_time();

        match direction {
            FrameDirection::Downstream => {
                let next_guard = self.0.next.read().await;
                if let Some(next_inner) = next_guard.as_ref() {
                    let next = FrameProcessor(next_inner.clone());
                    log::trace!(
                        "Pushing {} downstream from {} to {}",
                        frame.name(),
                        self.name(),
                        next.name()
                    );
                    if let Some(obs) = self.0.observer.read().await.as_ref() {
                        obs.on_push_frame(FramePushed {
                            source_name: self.0.name.clone(),
                            destination_name: next.0.name.clone(),
                            frame: frame.clone(),
                            direction,
                            timestamp: ts,
                        })
                        .await;
                    }
                    Box::pin(next.queue_frame(frame, direction, None)).await?;
                }
            }
            FrameDirection::Upstream => {
                let prev_guard = self.0.prev.read().await;
                if let Some(prev_weak) = prev_guard.as_ref() {
                    if let Some(prev_inner) = prev_weak.upgrade() {
                        let prev = FrameProcessor(prev_inner);
                        log::trace!(
                            "Pushing {} upstream from {} to {}",
                            frame.name(),
                            self.name(),
                            prev.name()
                        );
                        if let Some(obs) = self.0.observer.read().await.as_ref() {
                            obs.on_push_frame(FramePushed {
                                source_name: self.0.name.clone(),
                                destination_name: prev.0.name.clone(),
                                frame: frame.clone(),
                                direction,
                                timestamp: ts,
                            })
                            .await;
                        }
                        Box::pin(prev.queue_frame(frame, direction, None)).await?;
                    }
                }
            }
        }

        Ok(())
    }

    // ---- Internal process_frame (wraps event handlers + callback) ----

    async fn internal_process_frame(
        &self,
        frame: Frame,
        direction: FrameDirection,
        callback: Option<QueueCallback>,
    ) {
        // Before-process event handlers
        {
            let handlers = self.0.on_before_process_frame.lock().await;
            for h in handlers.iter() {
                h(&frame);
            }
        }

        if let Err(e) = self.process_frame(frame.clone(), direction).await {
            let _ = self
                .push_error(format!("Error processing frame: {}", e), false)
                .await;
        }

        // Fire post-process callback
        if let Some(cb) = callback {
            cb().await;
        }

        // After-process event handlers
        {
            let handlers = self.0.on_after_process_frame.lock().await;
            for h in handlers.iter() {
                h(&frame);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Task management (private)
// ---------------------------------------------------------------------------

impl FrameProcessor {
    fn create_input_task(&self) {
        let inner = self.0.clone();
        let handle = tokio::spawn(async move {
            input_frame_task_handler(inner).await;
        });
        *self.0.input_task.lock().unwrap() = Some(handle);
    }

    fn create_process_task(&self) {
        // Reset blocking state before spawning.
        self.0.should_block_frames.store(false, Ordering::Relaxed);
        let inner = self.0.clone();
        let handle = tokio::spawn(async move {
            process_frame_task_handler(inner).await;
        });
        *self.0.process_task.lock().unwrap() = Some(handle);
    }

    async fn cancel_input_task(&self) {
        let handle = self.0.input_task.lock().unwrap().take();
        if let Some(h) = handle {
            h.abort();
            let _ = tokio::time::timeout(
                Duration::from_secs_f64(INPUT_TASK_CANCEL_TIMEOUT_SECS),
                h,
            )
            .await;
        }
    }

    async fn cancel_process_task(&self) {
        let handle = self.0.process_task.lock().unwrap().take();
        if let Some(h) = handle {
            h.abort();
            let _ = tokio::time::timeout(Duration::from_secs(1), h).await;
        }
    }

    async fn reset_process_task(&self) {
        self.0.should_block_frames.store(false, Ordering::Relaxed);
        self.reset_process_queue().await;
    }

    async fn reset_process_queue(&self) {
        self.0.input_queue.drain_keep_uninterruptible().await;
        self.0.process_queue.drain_keep_uninterruptible().await;
    }
}

// ---------------------------------------------------------------------------
// Async task loops (free functions so they can be spawned without capturing &self)
// ---------------------------------------------------------------------------

/// Input task: pulls from the priority queue, routes system frames for immediate
/// processing and non-system frames into the process queue.
async fn input_frame_task_handler(inner: Arc<Inner>) {
    loop {
        let (frame, direction, callback) = inner.input_queue.get().await;

        // Block if pausing system frames was requested
        if inner.should_block_system_frames.load(Ordering::Relaxed) {
            log::trace!("{}: system frame processing paused", &inner.name);
            inner.input_event.notified().await;
            inner
                .should_block_system_frames
                .store(false, Ordering::Relaxed);
            log::trace!("{}: system frame processing resumed", &inner.name);
        }

        let processor = FrameProcessor(inner.clone());

        if frame.is_system() {
            // Process system frames immediately in this task
            processor
                .internal_process_frame(frame, direction, callback)
                .await;
        } else if !inner.cancelling.load(Ordering::Relaxed) {
            // Queue non-system frames for the process task
            inner
                .process_queue
                .put((frame, direction, callback))
                .await;
        }
    }
}

/// Process task: pulls non-system frames from the process queue and handles them.
async fn process_frame_task_handler(inner: Arc<Inner>) {
    loop {
        let (frame, direction, callback) = inner.process_queue.get().await;

        *inner.process_current_frame.lock().await = Some(frame.clone());

        // Block if pausing non-system frames was requested
        if inner.should_block_frames.load(Ordering::Relaxed) {
            log::trace!("{}: frame processing paused", &inner.name);
            inner.process_event.notified().await;
            inner.should_block_frames.store(false, Ordering::Relaxed);
            log::trace!("{}: frame processing resumed", &inner.name);
        }

        let processor = FrameProcessor(inner.clone());
        processor
            .internal_process_frame(frame, direction, callback)
            .await;

        *inner.process_current_frame.lock().await = None;
    }
}
