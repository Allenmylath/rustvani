use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::{mpsc, RwLock};

use crate::clock::BaseClock;
use crate::error::Result;
use crate::frames::{Frame, FrameDirection};
use crate::observer::BaseObserver;
use crate::pipeline::PipelineTask;

use super::bus::{AgentBus, BusMessage, BusPayload, BusSubscriber};
use super::registry::AgentRegistry;

// ---------------------------------------------------------------------------
// Agent trait
// ---------------------------------------------------------------------------

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

pub struct BaseAgent {
    name: String,
    parent: Option<String>,
    pipeline_task: PipelineTask,
    push_tx: mpsc::Sender<(Frame, FrameDirection)>,
    active: AtomicBool,
    bridged: AtomicBool,
    ready: AtomicBool,
    bus: RwLock<Option<Arc<dyn AgentBus>>>,
    registry: RwLock<Option<Arc<AgentRegistry>>>,
}

impl BaseAgent {
    pub fn new(
        name: impl Into<String>,
        pipeline_task: PipelineTask,
        bridged: bool,
        active_on_start: bool,
    ) -> Self {
        let push_tx = pipeline_task.push_sender();
        Self {
            name: name.into(),
            parent: None,
            pipeline_task,
            push_tx,
            active: AtomicBool::new(active_on_start),
            bridged: AtomicBool::new(bridged),
            ready: AtomicBool::new(false),
            bus: RwLock::new(None),
            registry: RwLock::new(None),
        }
    }

    pub fn with_parent(mut self, parent: impl Into<String>) -> Self {
        self.parent = Some(parent.into());
        self
    }

    async fn announce_ready(&self) {
        let runner = {
            let registry_guard = self.registry.read().await;
            match registry_guard.as_ref() {
                Some(r) => r.runner_name().to_string(),
                None => return,
            }
        };

        let info = super::registry::AgentInfo {
            name: self.name.clone(),
            runner: runner.clone(),
            parent: self.parent.clone(),
            active: self.active.load(Ordering::Relaxed),
            bridged: self.bridged.load(Ordering::Relaxed),
            started_at: Some(crate::clock::system_clock().get_time()),
        };

        {
            let registry_guard = self.registry.read().await;
            if let Some(registry) = registry_guard.as_ref() {
                let _ = registry.register(info.clone()).await;
            }
        }

        let bus_guard = self.bus.read().await;
        if let Some(bus) = bus_guard.as_ref() {
            let msg = BusMessage {
                source: self.name.clone(),
                target: None,
                payload: BusPayload::AgentReady {
                    runner: info.runner,
                    parent: info.parent,
                    active: info.active,
                    bridged: info.bridged,
                    started_at: info.started_at,
                },
            };
            bus.send(msg).await;
        }
    }
}

#[async_trait]
impl BusSubscriber for BaseAgent {
    fn name(&self) -> &str {
        &self.name
    }

    async fn on_bus_message(&self, message: BusMessage) {
        if message.source == self.name {
            return;
        }

        if let Some(target) = &message.target {
            if target != &self.name {
                return;
            }
        }

        match message.payload {
            BusPayload::Frame { frame, direction } => {
                if self.bridged.load(Ordering::Relaxed) {
                    let _ = self.push_tx.send((frame, direction)).await;
                }
            }
            BusPayload::Activate { .. } => {
                self.active.store(true, Ordering::Relaxed);
            }
            BusPayload::Deactivate => {
                self.active.store(false, Ordering::Relaxed);
            }
            BusPayload::End { reason } => {
                let frame = match reason {
                    Some(r) => Frame::end_with(r),
                    None => Frame::end(),
                };
                let _ = self.push_tx.send((frame, FrameDirection::Downstream)).await;
            }
            BusPayload::Cancel { reason } => {
                let frame = match reason {
                    Some(r) => Frame::cancel_with(r),
                    None => Frame::cancel(),
                };
                let _ = self.push_tx.send((frame, FrameDirection::Downstream)).await;
            }
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
        *self.bus.write().await = Some(bus);
        *self.registry.write().await = Some(registry);
        Ok(())
    }

    async fn run(
        &self,
        clock: Arc<dyn BaseClock>,
        observer: Option<Arc<dyn BaseObserver>>,
    ) -> Result<()> {
        self.ready.store(true, Ordering::Relaxed);
        self.announce_ready().await;
        self.pipeline_task.run(clock, observer).await
    }

    async fn end(&self, reason: Option<String>) -> Result<()> {
        let frame = match reason {
            Some(r) => Frame::end_with(r),
            None => Frame::end(),
        };
        let _ = self.push_tx.send((frame, FrameDirection::Downstream)).await;
        Ok(())
    }

    async fn cancel(&self, reason: Option<String>) -> Result<()> {
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
        self.bridged.load(Ordering::Relaxed)
    }

    fn ready(&self) -> bool {
        self.ready.load(Ordering::Relaxed)
    }
}
