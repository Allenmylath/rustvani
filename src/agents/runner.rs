use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tokio::time::{timeout, Duration};

use crate::clock::BaseClock;
use crate::error::{PipecatError, Result};
use crate::observer::BaseObserver;

use super::base::Agent;
use super::bus::{AgentBus, AgentRegistryEntry, BusMessage, BusPayload, BusSubscriber};
use super::registry::{AgentInfo, AgentRegistry};

// ---------------------------------------------------------------------------
// AgentRunner
// ---------------------------------------------------------------------------

pub struct AgentRunner {
    name: String,
    bus: Arc<dyn AgentBus>,
    registry: Arc<AgentRegistry>,
    agents: Mutex<HashMap<String, Arc<dyn Agent>>>,
    clock: Arc<dyn BaseClock>,
    observer: Option<Arc<dyn BaseObserver>>,
    shutdown: Arc<Notify>,
    shutdown_requested: Arc<AtomicBool>,
    started: Arc<AtomicBool>,
    agent_tasks: Mutex<Vec<JoinHandle<()>>>,
}

impl AgentRunner {
    pub fn new(
        name: impl Into<String>,
        bus: Arc<dyn AgentBus>,
        clock: Arc<dyn BaseClock>,
    ) -> Self {
        let name = name.into();
        let registry = AgentRegistry::new(&name);
        Self {
            name: name.clone(),
            bus,
            registry,
            agents: Mutex::new(HashMap::new()),
            clock,
            observer: None,
            shutdown: Arc::new(Notify::new()),
            shutdown_requested: Arc::new(AtomicBool::new(false)),
            started: Arc::new(AtomicBool::new(false)),
            agent_tasks: Mutex::new(Vec::new()),
        }
    }

    pub fn with_observer(mut self, observer: Arc<dyn BaseObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    pub fn bus(&self) -> &Arc<dyn AgentBus> {
        &self.bus
    }

    pub fn registry(&self) -> &Arc<AgentRegistry> {
        &self.registry
    }

    pub async fn add_agent(&self, agent: Arc<dyn Agent>) -> Result<()> {
        let name = agent.name().to_string();
        let mut agents = self.agents.lock().await;
        if agents.contains_key(&name) {
            log::error!("Agent '{}' already exists, skipping", name);
            return Ok(());
        }
        agents.insert(name, agent);
        Ok(())
    }

    pub async fn run(&self) -> Result<()> {
        if self.started.swap(true, Ordering::Relaxed) {
            return Err(PipecatError::pipeline(
                "AgentRunner::run() called more than once",
            ));
        }

        log::debug!("AgentRunner '{}' starting", self.name);

        // Subscribe runner to bus
        let runner_sub = Arc::new(RunnerSubscriber {
            runner_name: self.name.clone(),
            shutdown: self.shutdown.clone(),
            shutdown_requested: self.shutdown_requested.clone(),
            registry: self.registry.clone(),
            bus: self.bus.clone(),
        });
        self.bus.subscribe(runner_sub).await?;
        self.bus.start().await;

        // Subscribe all agents to bus
        let agents = self.agents.lock().await.clone();
        for (_, agent) in &agents {
            let wrapper = Arc::new(AgentSubscriberWrapper(agent.clone()));
            self.bus.subscribe(wrapper).await?;
        }

        // Setup all agents
        for (_, agent) in &agents {
            agent.setup(self.bus.clone(), self.registry.clone()).await?;
        }

        // Start all agents
        let mut tasks = self.agent_tasks.lock().await;
        for (_, agent) in &agents {
            let agent = agent.clone();
            let clock = self.clock.clone();
            let observer = self.observer.clone();
            let handle = tokio::spawn(async move {
                if let Err(e) = agent.run(clock, observer).await {
                    log::error!("Agent '{}' run error: {}", agent.name(), e);
                }
            });
            tasks.push(handle);
        }
        drop(tasks);

        // Wait for shutdown
        self.shutdown.notified().await;

        // End root agents
        let agents = self.agents.lock().await;
        for (_, agent) in agents.iter() {
            if agent.parent().is_none() {
                agent.end(None).await.ok();
            }
        }

        // Wait for agent tasks to finish (with timeout)
        let mut tasks = self.agent_tasks.lock().await;
        for handle in tasks.drain(..) {
            let _ = timeout(Duration::from_secs(10), handle).await;
        }

        self.bus.stop().await;

        log::debug!("AgentRunner '{}' finished", self.name);
        Ok(())
    }

    pub async fn end(&self, reason: Option<String>) {
        if self.shutdown_requested.swap(true, Ordering::Relaxed) {
            return;
        }
        log::debug!("AgentRunner '{}' ending gracefully", self.name);

        let agents = self.agents.lock().await;
        for (_, agent) in agents.iter() {
            if agent.parent().is_none() {
                let msg = BusMessage {
                    source: self.name.clone(),
                    target: Some(agent.name().to_string()),
                    payload: BusPayload::End {
                        reason: reason.clone(),
                    },
                };
                self.bus.send(msg).await;
            }
        }
        self.shutdown.notify_one();
    }

    pub async fn cancel(&self, reason: Option<String>) {
        if self.shutdown_requested.swap(true, Ordering::Relaxed) {
            return;
        }
        log::debug!("AgentRunner '{}' cancelling", self.name);

        let agents = self.agents.lock().await;
        for (_, agent) in agents.iter() {
            if agent.parent().is_none() {
                let msg = BusMessage {
                    source: self.name.clone(),
                    target: Some(agent.name().to_string()),
                    payload: BusPayload::Cancel {
                        reason: reason.clone(),
                    },
                };
                self.bus.send(msg).await;
            }
        }
        self.shutdown.notify_one();
    }
}

// ---------------------------------------------------------------------------
// AgentSubscriberWrapper — bridges Arc<dyn Agent> into Arc<dyn BusSubscriber>
// ---------------------------------------------------------------------------

struct AgentSubscriberWrapper(Arc<dyn Agent>);

#[async_trait]
impl BusSubscriber for AgentSubscriberWrapper {
    fn name(&self) -> &str {
        self.0.name()
    }

    async fn on_bus_message(&self, message: BusMessage) {
        self.0.on_bus_message(message).await;
    }
}

// ---------------------------------------------------------------------------
// RunnerSubscriber — handles runner-level bus messages
// ---------------------------------------------------------------------------

struct RunnerSubscriber {
    runner_name: String,
    shutdown: Arc<Notify>,
    shutdown_requested: Arc<AtomicBool>,
    registry: Arc<AgentRegistry>,
    bus: Arc<dyn AgentBus>,
}

#[async_trait]
impl BusSubscriber for RunnerSubscriber {
    fn name(&self) -> &str {
        &self.runner_name
    }

    async fn on_bus_message(&self, message: BusMessage) {
        if message.source == self.runner_name {
            return;
        }

        match message.payload {
            BusPayload::End { .. } => {
                if self.shutdown_requested.swap(true, Ordering::Relaxed) {
                    return;
                }
                self.shutdown.notify_one();
            }
            BusPayload::Cancel { .. } => {
                if self.shutdown_requested.swap(true, Ordering::Relaxed) {
                    return;
                }
                self.shutdown.notify_one();
            }
            BusPayload::AgentReady {
                runner,
                parent,
                active,
                bridged,
                started_at,
            } => {
                let is_local = runner == self.runner_name;

                if !is_local {
                    let info = AgentInfo {
                        name: message.source.clone(),
                        runner,
                        parent,
                        active,
                        bridged,
                        started_at,
                    };
                    self.registry.register(info).await;
                }

                // When a local agent becomes ready, broadcast our registry
                // so remote runners can discover us.
                if is_local {
                    let entries: Vec<AgentRegistryEntry> = {
                        let local_names = self.registry.local_agents().await;
                        let mut entries = Vec::new();
                        for name in local_names {
                            if let Some(info) = self.registry.get(&name).await {
                                entries.push(AgentRegistryEntry {
                                    name: info.name,
                                    parent: info.parent,
                                    active: info.active,
                                    bridged: info.bridged,
                                    started_at: info.started_at,
                                });
                            }
                        }
                        entries
                    };

                    if !entries.is_empty() {
                        let msg = BusMessage {
                            source: self.runner_name.clone(),
                            target: None,
                            payload: BusPayload::AgentRegistry {
                                runner: self.runner_name.clone(),
                                agents: entries,
                            },
                        };
                        self.bus.send(msg).await;
                    }
                }
            }
            BusPayload::AgentRegistry { runner, agents } => {
                if runner != self.runner_name {
                    for entry in agents {
                        let info = AgentInfo {
                            name: entry.name,
                            runner: runner.clone(),
                            parent: entry.parent,
                            active: entry.active,
                            bridged: entry.bridged,
                            started_at: entry.started_at,
                        };
                        self.registry.register(info).await;
                    }
                }
            }
            _ => {}
        }
    }
}
