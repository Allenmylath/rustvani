use std::collections::HashMap;
use std::pin::Pin;
use std::future::Future;
use std::sync::Arc;
use tokio::sync::Mutex;

// ---------------------------------------------------------------------------
// AgentInfo
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AgentInfo {
    pub name: String,
    pub runner: String,
    pub parent: Option<String>,
    pub active: bool,
    pub bridged: bool,
    pub started_at: Option<f64>,
}

// ---------------------------------------------------------------------------
// WatchHandler
// ---------------------------------------------------------------------------

type WatchHandler = Arc<dyn Fn(AgentInfo) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

// ---------------------------------------------------------------------------
// AgentRegistry
// ---------------------------------------------------------------------------

pub struct AgentRegistry {
    runner_name: String,
    local: Mutex<HashMap<String, AgentInfo>>,
    remote: Mutex<HashMap<String, AgentInfo>>,
    watches: Mutex<HashMap<String, Vec<WatchHandler>>>,
}

impl AgentRegistry {
    pub fn new(runner_name: impl Into<String>) -> Arc<Self> {
        Arc::new(Self {
            runner_name: runner_name.into(),
            local: Mutex::new(HashMap::new()),
            remote: Mutex::new(HashMap::new()),
            watches: Mutex::new(HashMap::new()),
        })
    }

    pub fn runner_name(&self) -> &str {
        &self.runner_name
    }

    pub async fn local_agents(&self) -> Vec<String> {
        self.local.lock().await.keys().cloned().collect()
    }

    pub async fn remote_agents(&self) -> Vec<String> {
        self.remote.lock().await.keys().cloned().collect()
    }

    pub async fn get(&self, name: &str) -> Option<AgentInfo> {
        if let Some(info) = self.local.lock().await.get(name) {
            return Some(info.clone());
        }
        self.remote.lock().await.get(name).cloned()
    }

    pub async fn watch(&self, agent_name: &str, handler: WatchHandler) {
        let mut watches = self.watches.lock().await;
        watches.entry(agent_name.to_string()).or_default().push(handler);
        drop(watches);

        if let Some(info) = self.get(agent_name).await {
            let handlers: Vec<WatchHandler> = {
                let w = self.watches.lock().await;
                w.get(agent_name).cloned().unwrap_or_default()
            };
            for h in handlers {
                h(info.clone()).await;
            }
        }
    }

    pub async fn register(&self, info: AgentInfo) -> bool {
        let is_local = info.runner == self.runner_name;
        let mut target = if is_local {
            self.local.lock().await
        } else {
            self.remote.lock().await
        };

        if target.contains_key(&info.name) {
            return false;
        }

        target.insert(info.name.clone(), info.clone());
        drop(target);

        let handlers: Vec<WatchHandler> = {
            let w = self.watches.lock().await;
            w.get(&info.name).cloned().unwrap_or_default()
        };

        for h in handlers {
            h(info.clone()).await;
        }

        true
    }

    pub async fn unregister(&self, name: &str) {
        self.local.lock().await.remove(name);
        self.remote.lock().await.remove(name);
    }
}
