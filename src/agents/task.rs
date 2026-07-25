//! Task coordination between agents: dispatch, streaming updates,
//! completion, and cancellation.
//!
//! A [`TaskContext`] is owned by each agent (constructed in
//! `BaseAgent::setup`). `dispatch()` sends a `TaskRequest` over the bus and
//! returns a [`TaskHandle`] the caller can await; incoming responses are
//! routed back through [`TaskContext::route_update`] by the agent's bus
//! message handler.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::time::timeout;

use crate::error::{PipecatError, Result};

use super::bus::{AgentBus, BusMessage, BusPayload, TaskStatus};
use super::registry::AgentRegistry;

/// Default time `dispatch` waits for a target agent to become ready.
pub const DEFAULT_READY_TIMEOUT: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------
// TaskUpdate
// ---------------------------------------------------------------------------

/// One update delivered to a [`TaskHandle`].
#[derive(Debug, Clone)]
pub enum TaskUpdate {
    /// Terminal response — resolves `await_completion`.
    Response {
        /// Terminal status.
        status: TaskStatus,
        /// Arbitrary JSON result.
        response: Option<Value>,
    },
    /// First chunk of a streaming reply.
    StreamStart {
        /// Arbitrary JSON chunk.
        data: Option<Value>,
    },
    /// Intermediate chunk of a streaming reply.
    StreamData {
        /// Arbitrary JSON chunk.
        data: Option<Value>,
    },
    /// Last chunk of a streaming reply.
    StreamEnd {
        /// Arbitrary JSON chunk.
        data: Option<Value>,
    },
    /// Non-terminal progress update.
    Update {
        /// Arbitrary JSON update.
        update: Option<Value>,
    },
}

// ---------------------------------------------------------------------------
// TaskResult
// ---------------------------------------------------------------------------

/// Terminal outcome of a dispatched task.
#[derive(Debug, Clone)]
pub struct TaskResult {
    /// Id of the task.
    pub task_id: String,
    /// Terminal status.
    pub status: TaskStatus,
    /// Arbitrary JSON result.
    pub response: Option<Value>,
}

// ---------------------------------------------------------------------------
// TaskHandle
// ---------------------------------------------------------------------------

/// Caller-side handle for a dispatched task.
pub struct TaskHandle {
    /// Id of the task.
    pub task_id: String,
    /// Name of the agent executing the task.
    pub target_agent: String,
    rx: mpsc::UnboundedReceiver<TaskUpdate>,
}

impl TaskHandle {
    /// Wait for the task to complete with an optional timeout.
    ///
    /// Non-terminal updates (stream chunks, progress) are skipped; the first
    /// `Response` resolves the call. Errors if the channel closes without a
    /// response (e.g. the dispatching agent's context was torn down).
    pub async fn await_completion(
        mut self,
        timeout_duration: Option<Duration>,
    ) -> Result<TaskResult> {
        let task_id = self.task_id.clone();
        loop {
            let update = match timeout_duration {
                Some(d) => timeout(d, self.rx.recv())
                    .await
                    .map_err(|_| PipecatError::pipeline("Task timeout"))?,
                None => self.rx.recv().await,
            };

            match update {
                Some(TaskUpdate::Response { status, response }) => {
                    return Ok(TaskResult {
                        task_id,
                        status,
                        response,
                    })
                }
                Some(_) => continue,
                None => return Err(PipecatError::pipeline("Task did not return a response")),
            }
        }
    }

    /// Stream all updates until a `Response` is received.
    ///
    /// Returns the collected non-terminal updates and the terminal result.
    pub async fn stream_updates(
        self,
        timeout_duration: Option<Duration>,
    ) -> Result<(Vec<TaskUpdate>, TaskResult)> {
        let mut rx = self.rx;
        let task_id = self.task_id.clone();
        let mut updates = Vec::new();

        loop {
            let update = match timeout_duration {
                Some(d) => timeout(d, rx.recv())
                    .await
                    .map_err(|_| PipecatError::pipeline("Task stream timeout"))?,
                None => rx.recv().await,
            };

            match update {
                Some(TaskUpdate::Response { status, response }) => {
                    return Ok((
                        updates,
                        TaskResult {
                            task_id,
                            status,
                            response,
                        },
                    ));
                }
                Some(u) => updates.push(u),
                None => {
                    return Err(PipecatError::pipeline(
                        "Task stream closed without response",
                    ))
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TaskContext
// ---------------------------------------------------------------------------

/// Async callback invoked for each routed update of a watched task.
pub type UpdateHandler =
    Arc<dyn Fn(TaskUpdate) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

/// Per-agent task coordination state: pending handles awaiting responses
/// and registered update handlers.
pub struct TaskContext {
    bus: Arc<dyn AgentBus>,
    registry: Arc<AgentRegistry>,
    pending: Mutex<HashMap<String, mpsc::UnboundedSender<TaskUpdate>>>,
    update_handlers: Mutex<HashMap<String, Vec<UpdateHandler>>>,
    // TODO(job-groups): multi-target fan-out with cancel_on_error would
    // track group membership here, keyed alongside `pending`.
}

/// Lets a `TaskContext` be injected into a pipeline as a bus handle and
/// recovered by a coordinator processor via `as_any().downcast_ref()`.
impl crate::frames::ProcessorBusHandle for TaskContext {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl TaskContext {
    /// Create a task context bound to a bus and the agent registry.
    ///
    /// The registry lets [`TaskContext::dispatch`] gate on target readiness
    /// instead of silently dropping requests to agents that have not
    /// subscribed yet.
    pub fn new(bus: Arc<dyn AgentBus>, registry: Arc<AgentRegistry>) -> Self {
        Self {
            bus,
            registry,
            pending: Mutex::new(HashMap::new()),
            update_handlers: Mutex::new(HashMap::new()),
        }
    }

    /// Dispatch a task to a target agent with the default ready timeout
    /// ([`DEFAULT_READY_TIMEOUT`]).
    ///
    /// Thin wrapper around [`TaskContext::dispatch_with`].
    pub async fn dispatch(
        &self,
        source: &str,
        target: &str,
        task_name: Option<String>,
        payload: Option<Value>,
    ) -> Result<TaskHandle> {
        self.dispatch_with(
            source,
            target,
            task_name,
            payload,
            Some(DEFAULT_READY_TIMEOUT),
        )
        .await
    }

    /// Dispatch a task to a target agent.
    ///
    /// If the target is not yet in the registry, waits up to `ready_timeout`
    /// for it to announce readiness (no polling — resolved from a registry
    /// watch). `None` skips the gate entirely and sends immediately.
    /// Returns an error if the timeout elapses before the target is ready.
    pub async fn dispatch_with(
        &self,
        source: &str,
        target: &str,
        task_name: Option<String>,
        payload: Option<Value>,
        ready_timeout: Option<Duration>,
    ) -> Result<TaskHandle> {
        self.dispatch_inner(source, target, task_name, payload, ready_timeout, None)
            .await
    }

    /// Dispatch a task **fenced to a turn epoch** (turn-level ACID, Phase 2).
    ///
    /// Identical to [`dispatch_with`](Self::dispatch_with) but stamps
    /// `turn_epoch` on the outgoing `TaskRequest`. The executing agent echoes
    /// the epoch back onto its replies, letting the coordinator quarantine a
    /// result whose turn has been superseded by a barge-in. See
    /// `doc/turn-acid-phase2.md`.
    pub async fn dispatch_fenced(
        &self,
        source: &str,
        target: &str,
        task_name: Option<String>,
        payload: Option<Value>,
        turn_epoch: u64,
        ready_timeout: Option<Duration>,
    ) -> Result<TaskHandle> {
        self.dispatch_inner(
            source,
            target,
            task_name,
            payload,
            ready_timeout,
            Some(turn_epoch),
        )
        .await
    }

    async fn dispatch_inner(
        &self,
        source: &str,
        target: &str,
        task_name: Option<String>,
        payload: Option<Value>,
        ready_timeout: Option<Duration>,
        turn_epoch: Option<u64>,
    ) -> Result<TaskHandle> {
        if let Some(wait) = ready_timeout {
            self.await_target_ready(target, wait).await?;
        }

        let task_id = uuid::Uuid::new_v4().to_string();
        let (tx, rx) = mpsc::unbounded_channel();

        {
            let mut pending = self.pending.lock().await;
            pending.insert(task_id.clone(), tx);
        }

        let mut msg = BusMessage::new(
            source.to_string(),
            Some(target.to_string()),
            BusPayload::TaskRequest {
                task_id: task_id.clone(),
                task_name,
                payload,
            },
        );
        if let Some(epoch) = turn_epoch {
            msg = msg.with_turn_epoch(epoch);
        }

        self.bus.send(msg).await;

        Ok(TaskHandle {
            task_id,
            target_agent: target.to_string(),
            rx,
        })
    }

    /// Wait (without polling) until `target` appears in the registry.
    async fn await_target_ready(&self, target: &str, wait: Duration) -> Result<()> {
        if self.registry.get(target).await.is_some() {
            return Ok(());
        }

        // Register a watch that fires a oneshot when the agent registers.
        // The watch also fires immediately if the agent registered between
        // the check above and the watch installation, so there is no race.
        let (ready_tx, ready_rx) = oneshot::channel::<()>();
        let ready_tx = Arc::new(std::sync::Mutex::new(Some(ready_tx)));
        self.registry
            .watch(
                target,
                Arc::new(move |_info| {
                    let ready_tx = ready_tx.clone();
                    Box::pin(async move {
                        if let Some(tx) = ready_tx.lock().unwrap().take() {
                            let _ = tx.send(());
                        }
                    })
                }),
            )
            .await;

        timeout(wait, ready_rx)
            .await
            .map_err(|_| PipecatError::pipeline(format!("Target agent '{}' not ready", target)))?
            .map_err(|_| PipecatError::pipeline(format!("Target agent '{}' not ready", target)))?;
        Ok(())
    }

    /// Drain all pending task handles and registered update handlers.
    ///
    /// Dropping the senders makes every outstanding
    /// [`TaskHandle::await_completion`] / [`TaskHandle::stream_updates`]
    /// resolve with an error instead of hanging forever. Called from agent
    /// lifecycle cleanup when the owning agent ends or is cancelled.
    pub async fn fail_all_pending(&self, reason: &str) {
        let pending: HashMap<_, _> = {
            let mut guard = self.pending.lock().await;
            guard.drain().collect()
        };
        if !pending.is_empty() {
            log::debug!(
                "TaskContext: failing {} pending task(s): {}",
                pending.len(),
                reason
            );
        }
        drop(pending); // dropping the senders closes the receivers
        self.update_handlers.lock().await.clear();
    }

    /// Register a handler for task updates without awaiting completion.
    pub async fn on_update(&self, task_id: &str, handler: UpdateHandler) {
        let mut handlers = self.update_handlers.lock().await;
        handlers
            .entry(task_id.to_string())
            .or_default()
            .push(handler);
    }

    /// Route an incoming task update to pending handles and registered
    /// handlers. The `pending` lock is never held while user handlers run.
    pub async fn route_update(&self, task_id: &str, update: TaskUpdate) {
        // TODO(job-groups): group bookkeeping (cancel_on_error) hooks in here.
        // Send to pending channel.
        {
            let mut pending = self.pending.lock().await;
            if let Some(tx) = pending.get(task_id) {
                let is_final = matches!(update, TaskUpdate::Response { .. });
                let _ = tx.send(update.clone());
                if is_final {
                    pending.remove(task_id);
                }
            }
        }

        // Fire registered handlers (lock released above).
        let handlers: Vec<UpdateHandler> = {
            let h = self.update_handlers.lock().await;
            h.get(task_id).cloned().unwrap_or_default()
        };
        for handler in handlers {
            handler(update.clone()).await;
        }
    }

    /// Send the first chunk of a streaming task reply.
    pub async fn stream_start(
        &self,
        source: &str,
        target: &str,
        task_id: String,
        data: Option<Value>,
    ) {
        let msg = BusMessage::new(
            source.to_string(),
            Some(target.to_string()),
            BusPayload::TaskStreamStart { task_id, data },
        );
        self.bus.send(msg).await;
    }

    /// Send a streaming data chunk for a task.
    pub async fn stream_data(
        &self,
        source: &str,
        target: &str,
        task_id: String,
        data: Option<Value>,
    ) {
        let msg = BusMessage::new(
            source.to_string(),
            Some(target.to_string()),
            BusPayload::TaskStreamData { task_id, data },
        );
        self.bus.send(msg).await;
    }

    /// Send the last chunk of a streaming task reply.
    pub async fn stream_end(
        &self,
        source: &str,
        target: &str,
        task_id: String,
        data: Option<Value>,
    ) {
        let msg = BusMessage::new(
            source.to_string(),
            Some(target.to_string()),
            BusPayload::TaskStreamEnd { task_id, data },
        );
        self.bus.send(msg).await;
    }

    /// Mark a task as complete and send the terminal response.
    pub async fn complete_task(
        &self,
        source: &str,
        target: &str,
        task_id: String,
        status: TaskStatus,
        response: Option<Value>,
    ) {
        self.complete_task_fenced(source, target, task_id, status, response, None)
            .await;
    }

    /// Mark a task as complete, echoing a turn epoch back to the requester so
    /// the coordinator can fence a stale reply (turn-level ACID, Phase 2).
    pub async fn complete_task_fenced(
        &self,
        source: &str,
        target: &str,
        task_id: String,
        status: TaskStatus,
        response: Option<Value>,
        turn_epoch: Option<u64>,
    ) {
        let mut msg = BusMessage::new(
            source.to_string(),
            Some(target.to_string()),
            BusPayload::TaskResponse {
                task_id,
                status,
                response,
            },
        );
        if let Some(epoch) = turn_epoch {
            msg = msg.with_turn_epoch(epoch);
        }
        self.bus.send(msg).await;
    }

    /// Request cancellation of a task running on another agent.
    pub async fn cancel_task(
        &self,
        source: &str,
        target: &str,
        task_id: String,
        reason: Option<String>,
    ) {
        let msg = BusMessage::new(
            source.to_string(),
            Some(target.to_string()),
            BusPayload::TaskCancel { task_id, reason },
        );
        self.bus.send(msg).await;
    }

    /// Send an urgent terminal response (system priority).
    pub async fn urgent_response(
        &self,
        source: &str,
        target: &str,
        task_id: String,
        status: TaskStatus,
        response: Option<Value>,
    ) {
        let msg = BusMessage::new(
            source.to_string(),
            Some(target.to_string()),
            BusPayload::TaskResponseUrgent {
                task_id,
                status,
                response,
            },
        );
        self.bus.send(msg).await;
    }

    /// Send an urgent progress update (system priority).
    pub async fn urgent_update(
        &self,
        source: &str,
        target: &str,
        task_id: String,
        update: Option<Value>,
    ) {
        let msg = BusMessage::new(
            source.to_string(),
            Some(target.to_string()),
            BusPayload::TaskUpdateUrgent { task_id, update },
        );
        self.bus.send(msg).await;
    }
}
