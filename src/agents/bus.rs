use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::{Mutex, Notify};
use tokio::task::JoinHandle;

use crate::error::Result;
use crate::frames::{Frame, FrameDirection};

// ---------------------------------------------------------------------------
// TaskStatus
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

// ---------------------------------------------------------------------------
// BusPayload
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum BusPayload {
    // --- Frame transport ---
    Frame {
        frame: Frame,
        direction: FrameDirection,
    },

    // --- Agent lifecycle ---
    Activate {
        args: Option<Value>,
    },
    Deactivate,
    End {
        reason: Option<String>,
    },
    Cancel {
        reason: Option<String>,
    },

    // --- Registry ---
    AgentReady {
        runner: String,
        parent: Option<String>,
        active: bool,
        bridged: bool,
        started_at: Option<f64>,
    },
    AgentRegistry {
        runner: String,
        agents: Vec<AgentRegistryEntry>,
    },
    AgentError {
        error: String,
    },

    // --- Task coordination ---
    TaskRequest {
        task_id: String,
        task_name: Option<String>,
        payload: Option<Value>,
    },
    TaskResponse {
        task_id: String,
        status: TaskStatus,
        response: Option<Value>,
    },
    TaskResponseUrgent {
        task_id: String,
        status: TaskStatus,
        response: Option<Value>,
    },
    TaskUpdate {
        task_id: String,
        update: Option<Value>,
    },
    TaskUpdateUrgent {
        task_id: String,
        update: Option<Value>,
    },
    TaskUpdateRequest {
        task_id: String,
    },
    TaskCancel {
        task_id: String,
        reason: Option<String>,
    },

    // --- Task streaming ---
    TaskStreamStart {
        task_id: String,
        data: Option<Value>,
    },
    TaskStreamData {
        task_id: String,
        data: Option<Value>,
    },
    TaskStreamEnd {
        task_id: String,
        data: Option<Value>,
    },
}

// ---------------------------------------------------------------------------
// BusMessage
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BusMessage {
    pub source: String,
    pub target: Option<String>,
    pub payload: BusPayload,
}

impl BusMessage {
    pub fn is_system(&self) -> bool {
        matches!(
            self.payload,
            BusPayload::End { .. }
                | BusPayload::Cancel { .. }
                | BusPayload::Activate { .. }
                | BusPayload::Deactivate
                | BusPayload::AgentReady { .. }
                | BusPayload::AgentRegistry { .. }
                | BusPayload::AgentError { .. }
                | BusPayload::TaskResponseUrgent { .. }
                | BusPayload::TaskUpdateUrgent { .. }
                | BusPayload::TaskCancel { .. }
        )
    }
}

// ---------------------------------------------------------------------------
// AgentRegistryEntry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AgentRegistryEntry {
    pub name: String,
    pub parent: Option<String>,
    pub active: bool,
    pub bridged: bool,
    pub started_at: Option<f64>,
}

// ---------------------------------------------------------------------------
// BusSubscriber
// ---------------------------------------------------------------------------

#[async_trait]
pub trait BusSubscriber: Send + Sync {
    fn name(&self) -> &str;
    async fn on_bus_message(&self, message: BusMessage);
}

// ---------------------------------------------------------------------------
// AgentBus trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait AgentBus: Send + Sync {
    async fn subscribe(&self, subscriber: Arc<dyn BusSubscriber>) -> Result<()>;
    async fn unsubscribe(&self, name: &str);
    async fn send(&self, message: BusMessage);
    async fn start(&self);
    async fn stop(&self);
}

// ---------------------------------------------------------------------------
// LocalAgentBus
// ---------------------------------------------------------------------------

struct SubscriptionState {
    name: String,
    system_queue: Arc<Mutex<VecDeque<BusMessage>>>,
    data_queue: Arc<Mutex<VecDeque<BusMessage>>>,
    notify: Arc<Notify>,
    dispatch_handle: Mutex<Option<JoinHandle<()>>>,
}

pub struct LocalAgentBus {
    subscriptions: Arc<Mutex<Vec<SubscriptionState>>>,
    running: AtomicBool,
}

impl LocalAgentBus {
    pub fn new() -> Self {
        Self {
            subscriptions: Arc::new(Mutex::new(Vec::new())),
            running: AtomicBool::new(false),
        }
    }

    // Dispatch loop is started inline in subscribe().

}

#[async_trait]
impl AgentBus for LocalAgentBus {
    async fn subscribe(&self, subscriber: Arc<dyn BusSubscriber>) -> Result<()> {
        let name = subscriber.name().to_string();
        let system_queue = Arc::new(Mutex::new(VecDeque::new()));
        let data_queue = Arc::new(Mutex::new(VecDeque::new()));
        let notify = Arc::new(Notify::new());

        let dispatch_handle = {
            let sub = subscriber.clone();
            let sys_q = system_queue.clone();
            let dat_q = data_queue.clone();
            let ntf = notify.clone();

            let handle = tokio::spawn(async move {
                loop {
                    let msg = {
                        let mut sys = sys_q.lock().await;
                        if let Some(m) = sys.pop_front() {
                            m
                        } else {
                            let mut data = dat_q.lock().await;
                            if let Some(m) = data.pop_front() {
                                m
                            } else {
                                drop(data);
                                drop(sys);
                                ntf.notified().await;
                                continue;
                            }
                        }
                    };
                    sub.on_bus_message(msg).await;
                }
            });
            Mutex::new(Some(handle))
        };

        let sub_state = SubscriptionState {
            name,
            system_queue,
            data_queue,
            notify,
            dispatch_handle,
        };

        self.subscriptions.lock().await.push(sub_state);
        Ok(())
    }

    async fn unsubscribe(&self, name: &str) {
        let mut subs = self.subscriptions.lock().await;
        if let Some(idx) = subs.iter().position(|s| s.name == name) {
            let sub = subs.swap_remove(idx);
            let handle = {
                let mut guard = sub.dispatch_handle.lock().await;
                guard.take()
            };
            if let Some(h) = handle {
                h.abort();
            }
        }
    }

    async fn send(&self, message: BusMessage) {
        let subs = self.subscriptions.lock().await;
        let mut targets = Vec::new();
        for sub in subs.iter() {
            if message.source == sub.name {
                continue;
            }
            if let Some(target) = &message.target {
                if &sub.name != target {
                    continue;
                }
            }
            targets.push((sub.system_queue.clone(), sub.data_queue.clone(), sub.notify.clone()));
        }
        drop(subs);

        let is_system = message.is_system();
        for (sys_q, data_q, notify) in targets {
            if is_system {
                sys_q.lock().await.push_back(message.clone());
            } else {
                data_q.lock().await.push_back(message.clone());
            }
            notify.notify_one();
        }
    }

    async fn start(&self) {
        // Subscribers already start their dispatch loops on subscribe.
        // This method exists for network buses that need explicit start.
        self.running.store(true, Ordering::Relaxed);
    }

    async fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
        let mut subs = self.subscriptions.lock().await;
        for sub in subs.drain(..) {
            let handle = {
                let mut guard = sub.dispatch_handle.lock().await;
                guard.take()
            };
            if let Some(h) = handle {
                h.abort();
            }
        }
    }
}
