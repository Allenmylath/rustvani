//! `BaseAgent` — a pipeline wrapped in a bus endpoint.
//!
//! Mirrors Pipecat's `BaseWorker`: the base agent IS the task router. Job
//! requests arriving over the bus are matched against named handlers
//! (registered with [`BaseAgent::on_task`]) and executed in their own tokio
//! task so the bus dispatch loop is never blocked by job work. Task
//! replies/updates are routed back into the agent's [`TaskContext`], which
//! is what makes `TaskHandle::await_completion` resolve end-to-end.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::BoxFuture;
use serde_json::Value;
use tokio::sync::{mpsc, Mutex, OnceCell};
use tokio::task::JoinHandle;

use crate::clock::BaseClock;
use crate::error::{PipecatError, Result};
use crate::frames::{Frame, FrameDirection};
use crate::observer::BaseObserver;
use crate::pipeline::PipelineTask;

use super::bus::{AgentBus, BusMessage, BusPayload, BusSubscriber, TaskStatus};
use super::edges::BusOutputEdge;
use super::registry::AgentRegistry;
use super::task::{TaskContext, TaskUpdate};

/// How long `end()` waits for each child agent to finish before pushing
/// the agent's own EndFrame. The runner's join timeout is the backstop.
const CHILD_END_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

// ---------------------------------------------------------------------------
// Task handler types
// ---------------------------------------------------------------------------

/// Everything a task handler needs to process one request and reply,
/// without threading names and ids manually.
#[derive(Clone)]
pub struct TaskRequestCtx {
    /// Unique id of this task.
    pub task_id: String,
    /// Handler name the request asked for (`None` = default handler).
    pub task_name: Option<String>,
    /// Arbitrary JSON payload from the requester.
    pub payload: Option<Value>,
    /// Name of the requesting agent — replies go back here.
    pub source: String,
    /// Name of the agent executing the handler.
    pub agent_name: String,
    /// Turn epoch the request was fenced to, echoed back on replies so the
    /// requesting coordinator can quarantine a stale result (turn-level ACID,
    /// Phase 2). `None` for unfenced requests.
    pub turn_epoch: Option<u64>,
    /// The executing agent's task context, for replies and nested dispatch.
    pub task_ctx: Arc<TaskContext>,
}

impl TaskRequestCtx {
    /// Send the terminal response for this task back to the requester,
    /// echoing the request's `turn_epoch` so it can be fenced.
    pub async fn complete(&self, status: TaskStatus, response: Option<Value>) {
        self.task_ctx
            .complete_task_fenced(
                &self.agent_name,
                &self.source,
                self.task_id.clone(),
                status,
                response,
                self.turn_epoch,
            )
            .await;
    }

    /// Send a `TaskStreamStart` chunk back to the requester.
    pub async fn stream_start(&self, data: Option<Value>) {
        self.task_ctx
            .stream_start(&self.agent_name, &self.source, self.task_id.clone(), data)
            .await;
    }

    /// Send a `TaskStreamData` chunk back to the requester.
    pub async fn stream_data(&self, data: Option<Value>) {
        self.task_ctx
            .stream_data(&self.agent_name, &self.source, self.task_id.clone(), data)
            .await;
    }

    /// Send a `TaskStreamEnd` chunk back to the requester.
    pub async fn stream_end(&self, data: Option<Value>) {
        self.task_ctx
            .stream_end(&self.agent_name, &self.source, self.task_id.clone(), data)
            .await;
    }
}

/// An async job handler registered with [`BaseAgent::on_task`].
pub type TaskHandler = Arc<dyn Fn(TaskRequestCtx) -> BoxFuture<'static, ()> + Send + Sync>;

/// Bookkeeping for one in-flight job on this agent.
struct ActiveJob {
    /// The requesting agent (terminal responses go back here).
    source: String,
    /// Handle of the spawned handler task.
    join: JoinHandle<()>,
}

// ---------------------------------------------------------------------------
// Agent trait
// ---------------------------------------------------------------------------

/// A named participant managed by an `AgentRunner`.
#[async_trait]
pub trait Agent: BusSubscriber {
    /// Parent agent name, if any.
    fn parent(&self) -> Option<&str>;

    /// Called once before the agent starts.
    async fn setup(&self, bus: Arc<dyn AgentBus>, registry: Arc<AgentRegistry>) -> Result<()>;

    /// Start the agent's pipeline. Blocks until the pipeline finishes.
    async fn run(
        &self,
        clock: Arc<dyn BaseClock>,
        observer: Option<Arc<dyn BaseObserver>>,
    ) -> Result<()>;

    /// Gracefully end the agent.
    async fn end(&self, reason: Option<String>) -> Result<()>;

    /// Hard cancel the agent.
    async fn cancel(&self, reason: Option<String>) -> Result<()>;

    /// Whether the agent is currently active.
    fn active(&self) -> bool;

    /// Whether the agent receives pipeline frames from the bus.
    fn bridged(&self) -> bool;

    /// Whether the agent has finished setup and is running.
    fn ready(&self) -> bool;
}

// ---------------------------------------------------------------------------
// BaseAgent
// ---------------------------------------------------------------------------

/// Default [`Agent`] implementation wrapping a [`PipelineTask`].
///
/// Construction is builder-style: create with [`BaseAgent::new`], then chain
/// [`BaseAgent::with_parent`], [`BaseAgent::on_task`] /
/// [`BaseAgent::on_task_default`] before handing it to the runner.
pub struct BaseAgent {
    name: String,
    parent: Option<String>,
    pipeline_task: PipelineTask,
    push_tx: mpsc::Sender<(Frame, FrameDirection)>,
    active: Arc<AtomicBool>,
    /// Bridged-input peer filter:
    /// - `None` — not bridged, ignore `Frame` payloads.
    /// - `Some([])` — accept `Frame` payloads from any source.
    /// - `Some(names)` — accept only when the message source is listed.
    bridged: Option<Vec<String>>,
    ready: AtomicBool,
    /// Single-shot shutdown guard: the first `end()`/`cancel()` wins;
    /// repeats (e.g. the runner backstop after a bus-delivered End) are
    /// no-ops, so the cascade and EndFrame happen exactly once.
    ending: AtomicBool,
    bus: OnceCell<Arc<dyn AgentBus>>,
    registry: OnceCell<Arc<AgentRegistry>>,
    task_ctx: OnceCell<Arc<TaskContext>>,
    /// Named job handlers (immutable after construction).
    handlers: HashMap<String, TaskHandler>,
    /// Fallback handler for requests without a matching name.
    default_handler: Option<TaskHandler>,
    /// In-flight jobs, keyed by task id. Arc so spawned handler tasks can
    /// remove themselves on completion.
    active_jobs: Arc<Mutex<HashMap<String, ActiveJob>>>,
    /// Output edge at the pipeline tail, if this agent publishes frames.
    /// The bus handle is injected into it during `setup()`.
    output_edge: Option<BusOutputEdge>,
}

impl BaseAgent {
    /// Create an agent wrapping `pipeline_task`.
    ///
    /// `bridged` controls which peers may inject pipeline frames over the
    /// bus (see the field docs); `active_on_start` sets the initial
    /// activation state.
    pub fn new(
        name: impl Into<String>,
        pipeline_task: PipelineTask,
        bridged: Option<Vec<String>>,
        active_on_start: bool,
    ) -> Self {
        let push_tx = pipeline_task.push_sender();
        Self {
            name: name.into(),
            parent: None,
            pipeline_task,
            push_tx,
            active: Arc::new(AtomicBool::new(active_on_start)),
            bridged,
            ready: AtomicBool::new(false),
            ending: AtomicBool::new(false),
            bus: OnceCell::new(),
            registry: OnceCell::new(),
            task_ctx: OnceCell::new(),
            handlers: HashMap::new(),
            default_handler: None,
            active_jobs: Arc::new(Mutex::new(HashMap::new())),
            output_edge: None,
        }
    }

    /// Create a fully bridged agent: builds a pipeline from `processors`
    /// with a [`BusOutputEdge`] appended at the tail, accepts bridged
    /// frames from `peers` (empty = any source), and publishes outgoing
    /// frames to the same peers (empty = broadcast).
    ///
    /// The edge shares the agent's activation flag: when the agent is
    /// inactive, frames still flow inside the pipeline but nothing is
    /// published, and bridged input is not injected — Pipecat-style
    /// handoff between several brains and one transport.
    pub fn bridged_pipeline(
        name: impl Into<String>,
        processors: Vec<crate::frames::FrameProcessor>,
        params: crate::pipeline::PipelineParams,
        peers: Vec<String>,
        active_on_start: bool,
    ) -> Self {
        let name = name.into();
        let edge = BusOutputEdge::new(name.clone(), peers.clone());
        let mut all = processors;
        all.push(edge.to_processor());
        let pipeline_task = PipelineTask::new(all, params);
        let agent = Self::new(name, pipeline_task, Some(peers), active_on_start);
        edge.bind_activation(agent.active.clone());
        agent.with_output_edge(edge)
    }

    /// Attach a pre-placed output edge (builder). Use this when you built
    /// the [`PipelineTask`] yourself and placed the edge's processor at the
    /// tail — the agent will inject the bus into it during `setup()` and
    /// share its activation flag with it.
    pub fn with_output_edge(mut self, edge: BusOutputEdge) -> Self {
        edge.bind_activation(self.active.clone());
        self.output_edge = Some(edge);
        self
    }

    /// Set the parent agent name (builder).
    pub fn with_parent(mut self, parent: impl Into<String>) -> Self {
        self.parent = Some(parent.into());
        self
    }

    /// Register a named job handler (builder). Requests whose `task_name`
    /// matches `name` are routed to `handler`, each in its own tokio task.
    pub fn on_task(mut self, name: impl Into<String>, handler: TaskHandler) -> Self {
        self.handlers.insert(name.into(), handler);
        self
    }

    /// Register the fallback handler (builder) for requests with no
    /// matching named handler (including `task_name: None`).
    pub fn on_task_default(mut self, handler: TaskHandler) -> Self {
        self.default_handler = Some(handler);
        self
    }

    /// The agent's task context, available after `setup()`.
    /// Use it to `dispatch` jobs to other agents.
    pub fn task_ctx(&self) -> Option<Arc<TaskContext>> {
        self.task_ctx.get().cloned()
    }

    /// The wrapped pipeline task — register frame callbacks or push frames
    /// before/while the agent runs.
    pub fn pipeline(&self) -> &PipelineTask {
        &self.pipeline_task
    }

    /// Shared activation flag (used by bus edge processors for gating).
    pub fn active_flag(&self) -> Arc<AtomicBool> {
        self.active.clone()
    }

    async fn announce_ready(&self) {
        let registry = match self.registry.get() {
            Some(r) => r,
            None => return,
        };

        let info = super::registry::AgentInfo {
            name: self.name.clone(),
            runner: registry.runner_name().to_string(),
            parent: self.parent.clone(),
            active: self.active.load(Ordering::Relaxed),
            bridged: self.bridged.is_some(),
            started_at: Some(crate::clock::system_clock().get_time()),
        };

        registry.register(info.clone()).await;

        if let Some(bus) = self.bus.get() {
            let msg = BusMessage::new(
                self.name.clone(),
                None,
                BusPayload::AgentReady {
                    runner: info.runner,
                    parent: info.parent,
                    active: info.active,
                    bridged: info.bridged,
                    started_at: info.started_at,
                },
            );
            bus.send(msg).await;
        }
    }

    /// Whether a bridged `Frame` payload from `source` should be injected.
    fn accepts_bridged_from(&self, source: &str) -> bool {
        match &self.bridged {
            None => false,
            Some(names) => names.is_empty() || names.iter().any(|n| n == source),
        }
    }

    /// Route a `TaskRequest` to the matching handler, spawned in its own
    /// task so the bus dispatch loop is never blocked by job work.
    async fn handle_task_request(
        &self,
        task_id: &str,
        task_name: &Option<String>,
        payload: &Option<Value>,
        source: &str,
        turn_epoch: Option<u64>,
    ) {
        let task_ctx = match self.task_ctx.get() {
            Some(ctx) => ctx.clone(),
            None => {
                log::error!(
                    "Agent '{}': TaskRequest before setup, dropping task {}",
                    self.name,
                    task_id
                );
                return;
            }
        };

        let handler = task_name
            .as_deref()
            .and_then(|n| self.handlers.get(n))
            .or(self.default_handler.as_ref())
            .cloned();

        let handler = match handler {
            Some(h) => h,
            None => {
                log::warn!(
                    "Agent '{}': no handler for task '{}', failing task {}",
                    self.name,
                    task_name.as_deref().unwrap_or("<default>"),
                    task_id
                );
                task_ctx
                    .complete_task_fenced(
                        &self.name,
                        source,
                        task_id.to_string(),
                        TaskStatus::Failed,
                        Some(serde_json::json!({
                            "error": "no handler",
                            "task_name": task_name,
                        })),
                        turn_epoch,
                    )
                    .await;
                return;
            }
        };

        let ctx = TaskRequestCtx {
            task_id: task_id.to_string(),
            task_name: task_name.clone(),
            payload: payload.clone(),
            source: source.to_string(),
            agent_name: self.name.clone(),
            turn_epoch,
            task_ctx,
        };

        // Insert while holding the lock so the spawned task's self-removal
        // (which also takes the lock) cannot run before the insert.
        let mut jobs = self.active_jobs.lock().await;
        let jobs_for_task = self.active_jobs.clone();
        let tid = task_id.to_string();
        let fut = handler(ctx);
        let join = tokio::spawn(async move {
            fut.await;
            jobs_for_task.lock().await.remove(&tid);
        });
        jobs.insert(
            task_id.to_string(),
            ActiveJob {
                source: source.to_string(),
                join,
            },
        );
    }

    /// Abort a running job and send the requester a terminal `Cancelled`
    /// response (Pipecat behavior: the requester always gets a terminal
    /// response, same path as success/failure).
    async fn handle_task_cancel(&self, task_id: &str, reason: &Option<String>) {
        let job = self.active_jobs.lock().await.remove(task_id);
        if let Some(job) = job {
            job.join.abort();
            if let Some(ctx) = self.task_ctx.get() {
                ctx.complete_task(
                    &self.name,
                    &job.source,
                    task_id.to_string(),
                    TaskStatus::Cancelled,
                    reason.as_ref().map(|r| serde_json::json!({ "reason": r })),
                )
                .await;
            }
        }
    }

    /// Forward End/Cancel to all child agents. For End, wait (bounded) for
    /// each child to finish so children stop before this agent's own
    /// pipeline does; Cancel propagates without waiting.
    async fn cascade_to_children(&self, reason: &Option<String>, is_end: bool) {
        let (bus, registry) = match (self.bus.get(), self.registry.get()) {
            (Some(b), Some(r)) => (b, r),
            _ => return,
        };

        let children = registry.children_of(&self.name).await;
        if children.is_empty() {
            return;
        }

        for child in &children {
            let payload = if is_end {
                BusPayload::End {
                    reason: reason.clone(),
                }
            } else {
                BusPayload::Cancel {
                    reason: reason.clone(),
                }
            };
            bus.send(BusMessage::new(
                self.name.clone(),
                Some(child.clone()),
                payload,
            ))
            .await;
        }

        if is_end {
            for child in &children {
                if tokio::time::timeout(CHILD_END_TIMEOUT, registry.wait_finished(child))
                    .await
                    .is_err()
                {
                    log::warn!(
                        "Agent '{}': child '{}' did not finish within {:?}, continuing shutdown",
                        self.name,
                        child,
                        CHILD_END_TIMEOUT
                    );
                }
            }
        }
    }

    /// Lifecycle cleanup: abort all in-flight jobs, send their requesters a
    /// terminal `Cancelled`, and fail every task handle this agent is
    /// awaiting so nothing hangs forever.
    async fn cleanup_jobs(&self, reason: &str) {
        let jobs: Vec<(String, ActiveJob)> = self.active_jobs.lock().await.drain().collect();
        if let Some(ctx) = self.task_ctx.get() {
            for (task_id, job) in jobs {
                job.join.abort();
                ctx.complete_task(
                    &self.name,
                    &job.source,
                    task_id,
                    TaskStatus::Cancelled,
                    Some(serde_json::json!({ "reason": reason })),
                )
                .await;
            }
            ctx.fail_all_pending(reason).await;
        } else {
            for (_, job) in jobs {
                job.join.abort();
            }
        }
    }
}

#[async_trait]
impl BusSubscriber for BaseAgent {
    fn name(&self) -> &str {
        &self.name
    }

    async fn on_bus_message(&self, message: Arc<BusMessage>) {
        if message.source == self.name {
            return;
        }

        if let Some(target) = &message.target {
            if target != &self.name {
                return;
            }
        }

        match &message.payload {
            // --- Bridged pipeline frames ---
            // Inactive agents skip bridged input entirely (handoff gating).
            BusPayload::Frame { frame, direction }
                if self.active.load(Ordering::Relaxed)
                    && self.accepts_bridged_from(&message.source) =>
            {
                let _ = self.push_tx.send((frame.clone(), *direction)).await;
            }

            // --- Lifecycle ---
            BusPayload::Activate { .. } => {
                self.active.store(true, Ordering::Relaxed);
            }
            BusPayload::Deactivate => {
                self.active.store(false, Ordering::Relaxed);
            }
            BusPayload::End { reason } => {
                let _ = self.end(reason.clone()).await;
            }
            BusPayload::Cancel { reason } => {
                let _ = self.cancel(reason.clone()).await;
            }

            // --- Job routing (this agent as executor) ---
            BusPayload::TaskRequest {
                task_id,
                task_name,
                payload,
            } => {
                self.handle_task_request(
                    task_id,
                    task_name,
                    payload,
                    &message.source,
                    message.turn_epoch,
                )
                .await;
            }
            BusPayload::TaskCancel { task_id, reason } => {
                self.handle_task_cancel(task_id, reason).await;
            }

            // --- Task replies (this agent as requester) ---
            BusPayload::TaskResponse {
                task_id,
                status,
                response,
            }
            | BusPayload::TaskResponseUrgent {
                task_id,
                status,
                response,
            } => {
                if let Some(ctx) = self.task_ctx.get() {
                    ctx.route_update(
                        task_id,
                        TaskUpdate::Response {
                            status: *status,
                            response: response.clone(),
                        },
                    )
                    .await;
                }
            }
            BusPayload::TaskUpdate { task_id, update }
            | BusPayload::TaskUpdateUrgent { task_id, update } => {
                if let Some(ctx) = self.task_ctx.get() {
                    ctx.route_update(
                        task_id,
                        TaskUpdate::Update {
                            update: update.clone(),
                        },
                    )
                    .await;
                }
            }
            BusPayload::TaskStreamStart { task_id, data } => {
                if let Some(ctx) = self.task_ctx.get() {
                    ctx.route_update(task_id, TaskUpdate::StreamStart { data: data.clone() })
                        .await;
                }
            }
            BusPayload::TaskStreamData { task_id, data } => {
                if let Some(ctx) = self.task_ctx.get() {
                    ctx.route_update(task_id, TaskUpdate::StreamData { data: data.clone() })
                        .await;
                }
            }
            BusPayload::TaskStreamEnd { task_id, data } => {
                if let Some(ctx) = self.task_ctx.get() {
                    ctx.route_update(task_id, TaskUpdate::StreamEnd { data: data.clone() })
                        .await;
                }
            }

            // Registry traffic is handled by the runner subscriber;
            // TaskUpdateRequest has no progress-reporting hook yet.
            _ => {}
        }
    }
}

#[async_trait]
impl Agent for BaseAgent {
    fn parent(&self) -> Option<&str> {
        self.parent.as_deref()
    }

    async fn setup(&self, bus: Arc<dyn AgentBus>, registry: Arc<AgentRegistry>) -> Result<()> {
        self.bus
            .set(bus.clone())
            .map_err(|_| PipecatError::pipeline("BaseAgent::setup called more than once"))?;
        self.registry
            .set(registry.clone())
            .map_err(|_| PipecatError::pipeline("BaseAgent::setup called more than once"))?;
        self.task_ctx
            .set(Arc::new(TaskContext::new(bus.clone(), registry)))
            .map_err(|_| PipecatError::pipeline("BaseAgent::setup called more than once"))?;
        if let Some(edge) = &self.output_edge {
            edge.set_bus(bus);
        }
        Ok(())
    }

    async fn run(
        &self,
        clock: Arc<dyn BaseClock>,
        observer: Option<Arc<dyn BaseObserver>>,
    ) -> Result<()> {
        self.ready.store(true, Ordering::Relaxed);
        self.announce_ready().await;
        // Expose this agent's TaskContext to its pipeline so coordinator
        // processors can dispatch to peers over the bus. Set before run() so
        // it reaches processor setup.
        if let Some(tc) = self.task_ctx.get() {
            self.pipeline_task.set_bus_handle(tc.clone());
        }
        let result = self.pipeline_task.run(clock, observer).await;
        // Never leave a requester (or our own pending handles) hanging.
        self.cleanup_jobs("agent ended").await;
        // Resolve any parent waiting on the End cascade.
        if let Some(registry) = self.registry.get() {
            registry.mark_finished(&self.name).await;
        }
        result
    }

    async fn end(&self, reason: Option<String>) -> Result<()> {
        if self.ending.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        // Children stop before this agent's own pipeline does.
        self.cascade_to_children(&reason, true).await;
        self.cleanup_jobs(reason.as_deref().unwrap_or("agent ended"))
            .await;
        let frame = match reason {
            Some(r) => Frame::end_with(r),
            None => Frame::end(),
        };
        let _ = self.push_tx.send((frame, FrameDirection::Downstream)).await;
        Ok(())
    }

    async fn cancel(&self, reason: Option<String>) -> Result<()> {
        if self.ending.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        // Propagate to children without waiting.
        self.cascade_to_children(&reason, false).await;
        self.cleanup_jobs(reason.as_deref().unwrap_or("agent cancelled"))
            .await;
        let frame = match reason {
            Some(r) => Frame::cancel_with(r),
            None => Frame::cancel(),
        };
        let _ = self.push_tx.send((frame, FrameDirection::Downstream)).await;
        Ok(())
    }

    fn active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    fn bridged(&self) -> bool {
        self.bridged.is_some()
    }

    fn ready(&self) -> bool {
        self.ready.load(Ordering::Relaxed)
    }
}
