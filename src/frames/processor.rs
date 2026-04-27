use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use async_trait::async_trait;
use futures::future::BoxFuture;
use tokio::sync::{Mutex, Notify, RwLock};
use tokio::task::JoinHandle;

use crate::clock::BaseClock;
use super::direction::FrameDirection;
use crate::error::Result;
use async_recursion::async_recursion;
use super::{ControlFrame, DataFrame, ErrorFrameData, Frame, FrameInner, StartFrameData, SystemFrame, next_frame_id};
use crate::metrics::{FrameProcessorMetrics, LLMTokenUsage};
use crate::observer::{BaseObserver, FrameProcessed, FramePushed};
use super::queue::{FrameProcessorQueue, ProcessQueue, QueueCallback};

// ---------------------------------------------------------------------------
// Public callback type (user-facing API to queue_frame)
// ---------------------------------------------------------------------------

pub type FrameCallback = Box<
    dyn FnOnce(FrameProcessor, Frame, FrameDirection) -> BoxFuture<'static, ()> + Send,
>;

// ---------------------------------------------------------------------------
// FrameProcessorSetup
// ---------------------------------------------------------------------------

/// Configuration passed to `FrameProcessor::setup()`.
#[derive(Clone)]
pub struct FrameProcessorSetup {
    pub clock: Arc<dyn BaseClock>,
    pub observer: Option<Arc<dyn BaseObserver>>,
}

// ---------------------------------------------------------------------------
// FrameHandler trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait FrameHandler: Send + Sync {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()>;

    fn can_generate_metrics(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// PassthroughHandler
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

const INPUT_TASK_CANCEL_TIMEOUT_SECS: f64 = 3.0;

// ---------------------------------------------------------------------------
// Inner — all shared mutable state
// ---------------------------------------------------------------------------

pub(crate) struct Inner {
    pub(crate) name: String,
    id: u64,

    // Linked-list links.
    // std::sync::RwLock is safe here: we always clone the pointer out
    // and drop the guard before any .await.
    prev: std::sync::RwLock<Option<Weak<Inner>>>,
    next: std::sync::RwLock<Option<Arc<Inner>>>,

    // Sub-processors for compound processors (Pipeline etc.).
    // Written once at construction, read many times.
    sub_processors: std::sync::RwLock<Vec<FrameProcessor>>,
    entry_processors_list: std::sync::RwLock<Vec<FrameProcessor>>,

    // ---- Queues ----
    input_queue: FrameProcessorQueue,
    process_queue: ProcessQueue,

    // ---- Atomic flags ----
    cancelling: AtomicBool,
    started: AtomicBool,
    should_block_system_frames: AtomicBool,
    should_block_frames: AtomicBool,

    // ---- Async events ----
    input_event: Notify,
    process_event: Notify,

    // ---- Task handles (std Mutex — never held across .await) ----
    input_task: std::sync::Mutex<Option<JoinHandle<()>>>,
    process_task: std::sync::Mutex<Option<JoinHandle<()>>>,

    process_current_frame: Mutex<Option<Frame>>,

    // ---- State flags from StartFrame ----
    allow_interruptions: AtomicBool,
    enable_metrics: AtomicBool,
    enable_usage_metrics: AtomicBool,
    report_only_initial_ttfb: AtomicBool,
    deprecated_openaillmcontext: AtomicBool,

    // ---- Infrastructure ----
    // These ARE held across .await (observer callbacks), so tokio RwLock.
    clock: RwLock<Option<Arc<dyn BaseClock>>>,
    observer: RwLock<Option<Arc<dyn BaseObserver>>>,

    // ---- Event handlers (sync, std Mutex) ----
    on_before_process_frame: std::sync::Mutex<Vec<FrameEventFn>>,
    on_after_process_frame:  std::sync::Mutex<Vec<FrameEventFn>>,
    on_before_push_frame:    std::sync::Mutex<Vec<FrameEventFn>>,
    on_after_push_frame:     std::sync::Mutex<Vec<FrameEventFn>>,
    on_error:                std::sync::Mutex<Vec<ErrorEventFn>>,

    // ---- Metrics ----
    metrics: FrameProcessorMetrics,

    // ---- User-supplied handler ----
    handler: Box<dyn FrameHandler>,

    // ---- Direct mode: skip queues ----
    enable_direct_mode: bool,
}

// ---------------------------------------------------------------------------
// WeakFrameProcessor — non-owning handle
// ---------------------------------------------------------------------------

/// A weak reference to a `FrameProcessor` that does not prevent deallocation.
#[derive(Clone)]
pub struct WeakFrameProcessor(pub(crate) Weak<Inner>);

impl WeakFrameProcessor {
    pub fn upgrade(&self) -> Option<FrameProcessor> {
        self.0.upgrade().map(FrameProcessor)
    }
}

// ---------------------------------------------------------------------------
// FrameProcessor — the public newtype wrapper around Arc<Inner>
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct FrameProcessor(pub(crate) Arc<Inner>);

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

impl FrameProcessor {
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
            prev: std::sync::RwLock::new(None),
            next: std::sync::RwLock::new(None),
            sub_processors: std::sync::RwLock::new(Vec::new()),
            entry_processors_list: std::sync::RwLock::new(Vec::new()),
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
            on_before_process_frame: std::sync::Mutex::new(Vec::new()),
            on_after_process_frame:  std::sync::Mutex::new(Vec::new()),
            on_before_push_frame:    std::sync::Mutex::new(Vec::new()),
            on_after_push_frame:     std::sync::Mutex::new(Vec::new()),
            on_error:                std::sync::Mutex::new(Vec::new()),
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
    pub fn id(&self) -> u64 { self.0.id }

    pub fn name(&self) -> &str { &self.0.name }

    /// Return all sub-processors (compound processors only; base returns empty).
    pub fn processors(&self) -> Vec<FrameProcessor> {
        self.0.sub_processors.read().unwrap().clone()
    }

    /// Return entry processors (first processor(s) in a compound processor).
    pub fn entry_processors(&self) -> Vec<FrameProcessor> {
        self.0.entry_processors_list.read().unwrap().clone()
    }

    /// Set sub-processors. Called once during Pipeline construction.
    pub fn set_sub_processors(&self, processors: Vec<FrameProcessor>) {
        *self.0.sub_processors.write().unwrap() = processors;
    }

    /// Set entry processors. Called once during Pipeline construction.
    pub fn set_entry_processors(&self, processors: Vec<FrameProcessor>) {
        *self.0.entry_processors_list.write().unwrap() = processors;
    }

    /// Downgrade to a non-owning weak reference.
    pub fn downgrade(&self) -> WeakFrameProcessor {
        WeakFrameProcessor(Arc::downgrade(&self.0))
    }

    /// Sync accessor — safe because we never hold the lock across .await.
    pub fn next(&self) -> Option<FrameProcessor> {
        self.0.next.read().unwrap().as_ref().map(|a| FrameProcessor(a.clone()))
    }

    /// Sync accessor — safe because we never hold the lock across .await.
    pub fn previous(&self) -> Option<FrameProcessor> {
        self.0.prev.read().unwrap()
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

    pub fn can_generate_metrics(&self) -> bool {
        self.0.handler.can_generate_metrics()
    }

    /// Recursively collect all processors that can generate metrics.
    pub fn processors_with_metrics(&self) -> Vec<FrameProcessor> {
        let mut result = Vec::new();
        for p in self.processors().iter() {
            if p.can_generate_metrics() {
                result.push(p.clone());
            }
            result.extend(p.processors_with_metrics());
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Lifecycle: setup / cleanup
// ---------------------------------------------------------------------------

impl FrameProcessor {
    /// Initialise this processor (and all sub-processors recursively).
    #[async_recursion]
    pub async fn setup(&self, setup: FrameProcessorSetup) -> Result<()> {
        *self.0.clock.write().await = Some(setup.clock.clone());
        *self.0.observer.write().await = setup.observer.clone();

        if !self.0.enable_direct_mode {
            self.create_input_task();
        }

        // Propagate to sub-processors (Pipeline, etc.)
        let sub_procs = self.0.sub_processors.read().unwrap().clone();
        for p in sub_procs {
            p.setup(setup.clone()).await?;
        }

        Ok(())
    }

    /// Shut down this processor (and all sub-processors recursively).
    #[async_recursion]
    pub async fn cleanup(&self) -> Result<()> {
        self.cancel_input_task().await;
        self.cancel_process_task().await;

        let sub_procs = self.0.sub_processors.read().unwrap().clone();
        for p in sub_procs {
            p.cleanup().await?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Linking — sync, safe because locks are never held across .await
// ---------------------------------------------------------------------------

impl FrameProcessor {
    /// Connect `self → next` in the pipeline (also sets `next.prev = self`).
    pub fn link(&self, next: &FrameProcessor) {
        log::debug!("Linking {} -> {}", self.name(), next.name());
        *self.0.next.write().unwrap() = Some(next.0.clone());
        *next.0.prev.write().unwrap() = Some(Arc::downgrade(&self.0));
    }
}

// ---------------------------------------------------------------------------
// Frame queuing — public entry point
// ---------------------------------------------------------------------------

impl FrameProcessor {
    #[async_recursion]
    pub async fn queue_frame(
        &self,
        frame: Frame,
        direction: FrameDirection,
        callback: Option<FrameCallback>,
    ) -> Result<()> {
        if self.0.cancelling.load(Ordering::Relaxed) {
            return Ok(());
        }

        let queue_cb: Option<QueueCallback> = callback.map(|cb| {
            let proc = self.clone();
            let f = frame.clone();
            let d = direction;
            let boxed: QueueCallback =
                Box::new(move || -> BoxFuture<'static, ()> {
                    Box::pin(async move { cb(proc, f, d).await })
                });
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
    pub async fn pause_processing_frames(&self) {
        log::trace!("{}: pausing frame processing", self.name());
        self.0.should_block_frames.store(true, Ordering::Relaxed);
    }

    pub async fn resume_processing_frames(&self) {
        log::trace!("{}: resuming frame processing", self.name());
        self.0.should_block_frames.store(false, Ordering::Relaxed);
        self.0.process_event.notify_one();
    }

    pub async fn pause_processing_system_frames(&self) {
        log::trace!("{}: pausing system frame processing", self.name());
        self.0.should_block_system_frames.store(true, Ordering::Relaxed);
    }

    pub async fn resume_processing_system_frames(&self) {
        log::trace!("{}: resuming system frame processing", self.name());
        self.0.should_block_system_frames.store(false, Ordering::Relaxed);
        self.0.input_event.notify_one();
    }
}

// ---------------------------------------------------------------------------
// process_frame
// ---------------------------------------------------------------------------

impl FrameProcessor {
    #[async_recursion]
    pub async fn process_frame(
        &self,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        // Observer notification
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

        // System-frame handling
        match &frame.inner {
            FrameInner::System(SystemFrame::Start(data)) => {
                self.handle_start(data.clone()).await;
            }
            FrameInner::System(SystemFrame::Interruption) => {
                self.start_interruption().await?;
                self.stop_all_metrics().await;
            }
            FrameInner::System(SystemFrame::Cancel { .. }) => {
                self.handle_cancel().await;
            }
            FrameInner::System(SystemFrame::PauseProcessor { name }) => {
                self.handle_pause(name.clone(), false).await;
            }
            FrameInner::System(SystemFrame::PauseProcessorUrgent { name }) => {
                self.handle_pause(name.clone(), true).await;
            }
            FrameInner::System(SystemFrame::ResumeProcessor { name }) => {
                self.handle_resume(name.clone(), false).await;
            }
            FrameInner::System(SystemFrame::ResumeProcessorUrgent { name }) => {
                self.handle_resume(name.clone(), true).await;
            }
            _ => {}
        }

        self.0.handler.on_process_frame(self, frame, direction).await
    }
}

// ---------------------------------------------------------------------------
// push_frame
// ---------------------------------------------------------------------------

impl FrameProcessor {
    #[async_recursion]
    pub async fn push_frame(
        &self,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        if !self.check_started(&frame) {
            return Ok(());
        }

        {
            let handlers = self.0.on_before_push_frame.lock().unwrap();
            for h in handlers.iter() { h(&frame); }
        }

        self.internal_push_frame(frame.clone(), direction).await?;

        {
            let handlers = self.0.on_after_push_frame.lock().unwrap();
            for h in handlers.iter() { h(&frame); }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// push_error / push_error_frame
// ---------------------------------------------------------------------------

impl FrameProcessor {
    pub async fn push_error(&self, error_msg: impl Into<String>, fatal: bool) -> Result<()> {
        let data = ErrorFrameData {
            error: error_msg.into(),
            fatal,
            processor_name: Some(self.0.name.clone()),
        };
        self.push_error_frame(data).await
    }

    pub async fn push_error_frame(&self, mut error: ErrorFrameData) -> Result<()> {
        if error.processor_name.is_none() {
            error.processor_name = Some(self.0.name.clone());
        }

        {
            let handlers = self.0.on_error.lock().unwrap();
            for h in handlers.iter() { h(&error); }
        }

        log::error!(
            "{} error: {}",
            error.processor_name.as_deref().unwrap_or("unknown"),
            error.error
        );

        let frame = Frame::error(error.error.clone(), error.fatal, error.processor_name.clone());
        // NOTE: error frame constructed above; broadcast_sibling_id removed from ErrorFrameData
        self.internal_push_frame(frame, FrameDirection::Upstream).await
    }
}

// ---------------------------------------------------------------------------
// Broadcast helpers
// ---------------------------------------------------------------------------

impl FrameProcessor {
    pub async fn broadcast_frame(&self, template: Frame) -> Result<()> {
        let mut downstream = template.clone().with_new_id();
        let mut upstream   = template.with_new_id();

        let ds_id = downstream.id;
        let us_id = upstream.id;

        downstream = downstream.with_sibling(us_id);
        upstream   = upstream.with_sibling(ds_id);

        self.push_frame(downstream, FrameDirection::Downstream).await?;
        self.push_frame(upstream,   FrameDirection::Upstream).await
    }

    pub async fn broadcast_frame_instance(&self, frame: Frame) -> Result<()> {
        self.broadcast_frame(frame).await
    }

    pub async fn broadcast_interruption(&self) -> Result<()> {
        log::debug!("{}: broadcasting interruption", self.name());
        self.reset_process_task().await;
        self.stop_all_metrics().await;
        self.broadcast_frame(Frame::interruption()).await
    }
}

// ---------------------------------------------------------------------------
// Metrics forwarding
// ---------------------------------------------------------------------------

impl FrameProcessor {
    pub async fn start_ttfb_metrics(&self, start_time: Option<f64>) {
        if self.can_generate_metrics() && self.metrics_enabled() {
            self.0.metrics.start_ttfb_metrics(start_time, self.report_only_initial_ttfb()).await;
        }
    }

    #[async_recursion]
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

    #[async_recursion]
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
    pub fn on_before_process_frame<F>(&self, f: F)
    where F: Fn(&Frame) + Send + Sync + 'static {
        self.0.on_before_process_frame.lock().unwrap().push(Box::new(f));
    }

    pub fn on_after_process_frame<F>(&self, f: F)
    where F: Fn(&Frame) + Send + Sync + 'static {
        self.0.on_after_process_frame.lock().unwrap().push(Box::new(f));
    }

    pub fn on_before_push_frame<F>(&self, f: F)
    where F: Fn(&Frame) + Send + Sync + 'static {
        self.0.on_before_push_frame.lock().unwrap().push(Box::new(f));
    }

    pub fn on_after_push_frame<F>(&self, f: F)
    where F: Fn(&Frame) + Send + Sync + 'static {
        self.0.on_after_push_frame.lock().unwrap().push(Box::new(f));
    }

    pub fn on_error<F>(&self, f: F)
    where F: Fn(&ErrorFrameData) + Send + Sync + 'static {
        self.0.on_error.lock().unwrap().push(Box::new(f));
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

impl FrameProcessor {
    fn get_time(&self) -> f64 {
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
                self.name(), frame.name()
            );
            return false;
        }
        true
    }

    // ---- System-frame sub-handlers ----

    async fn handle_start(&self, data: StartFrameData) {
        self.0.started.store(true, Ordering::Relaxed);
        self.0.allow_interruptions.store(data.allow_interruptions, Ordering::Relaxed);
        self.0.enable_metrics.store(data.enable_metrics, Ordering::Relaxed);
        self.0.enable_usage_metrics.store(data.enable_usage_metrics, Ordering::Relaxed);
        self.0.report_only_initial_ttfb.store(data.report_only_initial_ttfb, Ordering::Relaxed);
        self.0.deprecated_openaillmcontext.store(
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

    async fn handle_pause(&self, name: String, _urgent: bool) {
        if name == self.0.name {
            self.pause_processing_frames().await;
        }
    }

    async fn handle_resume(&self, name: String, _urgent: bool) {
        if name == self.0.name {
            self.resume_processing_frames().await;
        }
    }

    // ---- Interruption ----

    pub async fn drain_process_queue(&self) {
        self.reset_process_queue().await;
    }

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

    // ---- Internal push: lock is dropped before any .await ----

    #[async_recursion]
    async fn internal_push_frame(&self, frame: Frame, direction: FrameDirection) -> Result<()> {
        let ts = self.get_time();

        match direction {
            FrameDirection::Downstream => {
                // Clone the Arc out of the lock, then drop the lock before .await
                let next_opt = {
                    let guard = self.0.next.read().unwrap();
                    guard.as_ref().map(|a| FrameProcessor(a.clone()))
                };
                if let Some(next) = next_opt {
                    log::trace!(
                        "Pushing {} downstream: {} -> {}",
                        frame.name(), self.name(), next.name()
                    );
                    if let Some(obs) = self.0.observer.read().await.as_ref() {
                        obs.on_push_frame(FramePushed {
                            source_name:      self.0.name.clone(),
                            destination_name: next.0.name.clone(),
                            frame:            frame.clone(),
                            direction,
                            timestamp:        ts,
                        })
                        .await;
                    }
                    Box::pin(next.queue_frame(frame, direction, None)).await?;
                }
            }
            FrameDirection::Upstream => {
                let prev_opt = {
                    let guard = self.0.prev.read().unwrap();
                    guard.as_ref().and_then(|w| w.upgrade()).map(FrameProcessor)
                };
                if let Some(prev) = prev_opt {
                    log::trace!(
                        "Pushing {} upstream: {} -> {}",
                        frame.name(), self.name(), prev.name()
                    );
                    if let Some(obs) = self.0.observer.read().await.as_ref() {
                        obs.on_push_frame(FramePushed {
                            source_name:      self.0.name.clone(),
                            destination_name: prev.0.name.clone(),
                            frame:            frame.clone(),
                            direction,
                            timestamp:        ts,
                        })
                        .await;
                    }
                    Box::pin(prev.queue_frame(frame, direction, None)).await?;
                }
            }
        }

        Ok(())
    }

    // ---- Internal process (wraps event handlers + callback) ----

    #[async_recursion]
    async fn internal_process_frame(
        &self,
        frame: Frame,
        direction: FrameDirection,
        callback: Option<QueueCallback>,
    ) {
        {
            let handlers = self.0.on_before_process_frame.lock().unwrap();
            for h in handlers.iter() { h(&frame); }
        }

        if let Err(e) = self.process_frame(frame.clone(), direction).await {
            let _ = self.push_error(format!("Error processing frame: {}", e), false).await;
        }

        if let Some(cb) = callback { cb().await; }

        {
            let handlers = self.0.on_after_process_frame.lock().unwrap();
            for h in handlers.iter() { h(&frame); }
        }
    }
}

// ---------------------------------------------------------------------------
// Task management (private)
// ---------------------------------------------------------------------------

impl FrameProcessor {
    fn create_input_task(&self) {
        let inner = self.0.clone();
        let handle = tokio::spawn(async move { input_frame_task_handler(inner).await; });
        *self.0.input_task.lock().unwrap() = Some(handle);
    }

    fn create_process_task(&self) {
        self.0.should_block_frames.store(false, Ordering::Relaxed);
        let inner = self.0.clone();
        let handle = tokio::spawn(async move { process_frame_task_handler(inner).await; });
        *self.0.process_task.lock().unwrap() = Some(handle);
    }

    async fn cancel_input_task(&self) {
        let handle = self.0.input_task.lock().unwrap().take();
        if let Some(h) = handle {
            h.abort();
            let _ = tokio::time::timeout(
                Duration::from_secs_f64(INPUT_TASK_CANCEL_TIMEOUT_SECS), h
            ).await;
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
// Async task loops
// ---------------------------------------------------------------------------

async fn input_frame_task_handler(inner: Arc<Inner>) {
    loop {
        let (frame, direction, callback) = inner.input_queue.get().await;

        if inner.should_block_system_frames.load(Ordering::Relaxed) {
            log::trace!("{}: system frame processing paused", &inner.name);
            inner.input_event.notified().await;
            inner.should_block_system_frames.store(false, Ordering::Relaxed);
            log::trace!("{}: system frame processing resumed", &inner.name);
        }

        let processor = FrameProcessor(inner.clone());

        if frame.is_system() {
            processor.internal_process_frame(frame, direction, callback).await;
        } else if !inner.cancelling.load(Ordering::Relaxed) {
            inner.process_queue.put((frame, direction, callback)).await;
        }
    }
}

async fn process_frame_task_handler(inner: Arc<Inner>) {
    loop {
        let (frame, direction, callback) = inner.process_queue.get().await;

        *inner.process_current_frame.lock().await = Some(frame.clone());

        if inner.should_block_frames.load(Ordering::Relaxed) {
            log::trace!("{}: frame processing paused", &inner.name);
            inner.process_event.notified().await;
            inner.should_block_frames.store(false, Ordering::Relaxed);
            log::trace!("{}: frame processing resumed", &inner.name);
        }

        let processor = FrameProcessor(inner.clone());
        processor.internal_process_frame(frame, direction, callback).await;

        *inner.process_current_frame.lock().await = None;
    }
}
