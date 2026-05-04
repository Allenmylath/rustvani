use std::collections::HashMap;
use std::pin::Pin;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use tokio::sync::{mpsc, Mutex};
use tokio::time::timeout;

use crate::error::{PipecatError, Result};

use super::bus::{AgentBus, BusMessage, BusPayload, TaskStatus};

// ---------------------------------------------------------------------------
// TaskUpdate
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum TaskUpdate {
    Response {
        status: TaskStatus,
        response: Option<Value>,
    },
    StreamStart {
        data: Option<Value>,
    },
    StreamData {
        data: Option<Value>,
    },
    StreamEnd {
        data: Option<Value>,
    },
    Update {
        update: Option<Value>,
    },
}

// ---------------------------------------------------------------------------
// TaskResult
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TaskResult {
    pub task_id: String,
    pub status: TaskStatus,
    pub response: Option<Value>,
}

// ---------------------------------------------------------------------------
// TaskHandle
// ---------------------------------------------------------------------------

pub struct TaskHandle {
    pub task_id: String,
    pub target_agent: String,
    rx: mpsc::UnboundedReceiver<TaskUpdate>,
}

impl TaskHandle {
    /// Wait for the task to complete with an optional timeout.
    pub async fn await_completion(mut self, timeout_duration: Option<Duration>) -> Result<TaskResult> {
        let task_id = self.task_id.clone();
        let update = match timeout_duration {
            Some(d) => timeout(d, self.rx.recv())
                .await
                .map_err(|_| PipecatError::pipeline("Task timeout"))?,
            None => self.rx.recv().await,
        };

        match update {
            Some(TaskUpdate::Response { status, response }) => Ok(TaskResult {
                task_id,
                status,
                response,
            }),
            _ => Err(PipecatError::pipeline("Task did not return a response")),
        }
    }

    /// Stream all updates until a Response is received.
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

type UpdateHandler = Arc<dyn Fn(TaskUpdate) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

pub struct TaskContext {
    bus: Arc<dyn AgentBus>,
    pending: Mutex<HashMap<String, mpsc::UnboundedSender<TaskUpdate>>>,
    update_handlers: Mutex<HashMap<String, Vec<UpdateHandler>>>,
}

impl TaskContext {
    pub fn new(bus: Arc<dyn AgentBus>) -> Self {
        Self {
            bus,
            pending: Mutex::new(HashMap::new()),
            update_handlers: Mutex::new(HashMap::new()),
        }
    }

    /// Dispatch a task to a target agent.
    pub async fn dispatch(
        &self,
        source: &str,
        target: &str,
        task_name: Option<String>,
        payload: Option<Value>,
    ) -> Result<TaskHandle> {
        let task_id = uuid::Uuid::new_v4().to_string();
        let (tx, rx) = mpsc::unbounded_channel();

        {
            let mut pending = self.pending.lock().await;
            pending.insert(task_id.clone(), tx);
        }

        let msg = BusMessage {
            source: source.to_string(),
            target: Some(target.to_string()),
            payload: BusPayload::TaskRequest {
                task_id: task_id.clone(),
                task_name,
                payload,
            },
        };

        self.bus.send(msg).await;

        Ok(TaskHandle {
            task_id,
            target_agent: target.to_string(),
            rx,
        })
    }

    /// Register a handler for task updates without awaiting completion.
    pub async fn on_update(&self, task_id: &str, handler: UpdateHandler) {
        let mut handlers = self.update_handlers.lock().await;
        handlers.entry(task_id.to_string()).or_default().push(handler);
    }

    /// Route an incoming task update to pending handles and registered handlers.
    pub async fn route_update(&self, task_id: &str, update: TaskUpdate) {
        // Send to pending channel
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

        // Fire registered handlers
        let handlers: Vec<UpdateHandler> = {
            let h = self.update_handlers.lock().await;
            h.get(task_id).cloned().unwrap_or_default()
        };
        for handler in handlers {
            handler(update.clone()).await;
        }
    }

    /// Send a streaming data chunk for a task.
    pub async fn stream_data(
        &self,
        source: &str,
        target: &str,
        task_id: String,
        data: Option<Value>,
    ) {
        let msg = BusMessage {
            source: source.to_string(),
            target: Some(target.to_string()),
            payload: BusPayload::TaskStreamData { task_id, data },
        };
        self.bus.send(msg).await;
    }

    /// Mark a task as complete and send the response.
    pub async fn complete_task(
        &self,
        source: &str,
        target: &str,
        task_id: String,
        status: TaskStatus,
        response: Option<Value>,
    ) {
        let msg = BusMessage {
            source: source.to_string(),
            target: Some(target.to_string()),
            payload: BusPayload::TaskResponse {
                task_id,
                status,
                response,
            },
        };
        self.bus.send(msg).await;
    }

    /// Request cancellation of a task.
    pub async fn cancel_task(
        &self,
        source: &str,
        target: &str,
        task_id: String,
        reason: Option<String>,
    ) {
        let msg = BusMessage {
            source: source.to_string(),
            target: Some(target.to_string()),
            payload: BusPayload::TaskCancel { task_id, reason },
        };
        self.bus.send(msg).await;
    }

    /// Send an urgent response (system priority).
    pub async fn urgent_response(
        &self,
        source: &str,
        target: &str,
        task_id: String,
        status: TaskStatus,
        response: Option<Value>,
    ) {
        let msg = BusMessage {
            source: source.to_string(),
            target: Some(target.to_string()),
            payload: BusPayload::TaskResponseUrgent {
                task_id,
                status,
                response,
            },
        };
        self.bus.send(msg).await;
    }

    /// Send an urgent update (system priority).
    pub async fn urgent_update(
        &self,
        source: &str,
        target: &str,
        task_id: String,
        update: Option<Value>,
    ) {
        let msg = BusMessage {
            source: source.to_string(),
            target: Some(target.to_string()),
            payload: BusPayload::TaskUpdateUrgent { task_id, update },
        };
        self.bus.send(msg).await;
    }
}
