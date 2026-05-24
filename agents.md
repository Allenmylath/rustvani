# Agents — rustvani

Multi-agent coordination in rustvani sits on top of the frame pipeline. Each agent
owns a `PipelineTask`, communicates through an `AgentBus`, and is orchestrated by
an `AgentRunner`. The system is entirely async-native (tokio) with no global state.

---

## Core components

| Component | File | Role |
|---|---|---|
| `Agent` trait | `src/agents/base.rs` | Contract every agent must satisfy |
| `BaseAgent` | `src/agents/base.rs` | Concrete default implementation |
| `AgentBus` / `LocalAgentBus` | `src/agents/bus.rs` | Message broker with priority queues |
| `BusPayload` | `src/agents/bus.rs` | All message types on the bus |
| `AgentRunner` | `src/agents/runner.rs` | Orchestrates agents in parallel tokio tasks |
| `AgentRegistry` | `src/agents/registry.rs` | Local + remote agent discovery |
| `TaskContext` | `src/agents/task.rs` | Inter-agent task dispatch and streaming |

---

## Architecture

```
AgentRunner
  ├── AgentBus (LocalAgentBus)
  │     ├── per-agent dispatch loop (tokio::spawn per subscriber)
  │     │     system queue  ← lifecycle, errors, urgent responses
  │     │     data queue    ← frames, task requests, stream chunks
  │     └── priority: system queue always drains before data queue
  ├── AgentRegistry
  │     ├── local  — agents in this runner
  │     └── remote — agents discovered via AgentRegistry broadcasts
  └── Agents (tokio tasks)
        agent_a.run()  →  PipelineTask  →  frame pipeline
        agent_b.run()  →  PipelineTask  →  frame pipeline
```

Agents do not share memory. All coordination goes through `BusMessage` values sent
over the bus.

---

## The `Agent` trait

```rust
#[async_trait]
pub trait Agent: BusSubscriber {
    fn parent(&self) -> Option<&str>;

    async fn setup(&self, bus: Arc<dyn AgentBus>, registry: Arc<AgentRegistry>) -> Result<()>;
    async fn run(&self, clock: Arc<dyn BaseClock>, observer: Option<Arc<dyn BaseObserver>>) -> Result<()>;
    async fn end(&self, reason: Option<String>) -> Result<()>;
    async fn cancel(&self, reason: Option<String>) -> Result<()>;

    fn active(&self) -> bool;
    fn bridged(&self) -> bool;
    fn ready(&self) -> bool;
}
```

- `setup` — called once before `run`; store bus and registry references here
- `run` — blocks until the pipeline finishes; the tokio task lives here
- `end` — graceful shutdown; injects `EndFrame` into the pipeline
- `cancel` — hard abort; injects `CancelFrame` into the pipeline
- `bridged` — if `true`, the agent receives `BusPayload::Frame` messages injected
  from the bus into its pipeline

---

## `BaseAgent` — the default implementation

`BaseAgent` implements `Agent` in full. Extend it by wrapping it or composing it
inside your own struct.

```rust
let pipeline_task = PipelineTask::new(pipeline);

let agent = BaseAgent::new(
    "my-agent",          // name — must be unique on the runner
    pipeline_task,
    false,               // bridged: receives bus frames?
    true,                // active_on_start
);

// Optional: declare a parent for hierarchical shutdown
let child_agent = BaseAgent::new("child", child_task, false, true)
    .with_parent("my-agent");
```

`BaseAgent.run()` does three things in order:
1. Sets `ready = true`
2. Broadcasts `AgentReady` on the bus so other agents and the registry know it is live
3. Awaits `pipeline_task.run()` — blocks until the pipeline finishes

---

## Message bus (`LocalAgentBus`)

### Sending a message

```rust
bus.send(BusMessage {
    source: "sender-name".to_string(),
    target: Some("target-name".to_string()), // None = broadcast to all
    payload: BusPayload::Activate { args: None },
}).await;
```

Agents never receive their own messages (`source == self.name` is filtered out).

### Priority queues

Every subscriber has two queues. The dispatch loop always drains `system_queue`
first:

| Queue | Payloads routed here |
|---|---|
| system | `End`, `Cancel`, `Activate`, `Deactivate`, `AgentReady`, `AgentRegistry`, `AgentError`, `TaskResponseUrgent`, `TaskUpdateUrgent`, `TaskCancel` |
| data | everything else (`Frame`, `TaskRequest`, `TaskResponse`, `TaskStream*`, `TaskUpdate`) |

This guarantees lifecycle signals are never starved behind a backlog of data frames.

---

## All `BusPayload` variants

### Frame transport
| Variant | Purpose |
|---|---|
| `Frame { frame, direction }` | Inject a pipeline frame into a bridged agent |

### Agent lifecycle
| Variant | Purpose |
|---|---|
| `Activate { args }` | Mark agent active; `args` carries optional JSON config |
| `Deactivate` | Mark agent inactive |
| `End { reason }` | Graceful shutdown — sends `EndFrame` into pipeline |
| `Cancel { reason }` | Hard abort — sends `CancelFrame` into pipeline |

### Registry
| Variant | Purpose |
|---|---|
| `AgentReady { runner, parent, active, bridged, started_at }` | Broadcast when an agent finishes setup |
| `AgentRegistry { runner, agents }` | Batch registry sync — broadcast when a local agent becomes ready |
| `AgentError { error }` | Agent-level error notification |

### Task coordination
| Variant | Purpose |
|---|---|
| `TaskRequest { task_id, task_name, payload }` | Ask a target agent to perform work |
| `TaskResponse { task_id, status, response }` | Final result (normal priority) |
| `TaskResponseUrgent { task_id, status, response }` | Final result (system priority) |
| `TaskUpdate { task_id, update }` | Intermediate progress (normal priority) |
| `TaskUpdateUrgent { task_id, update }` | Intermediate progress (system priority) |
| `TaskUpdateRequest { task_id }` | Poll for current state |
| `TaskCancel { task_id, reason }` | Cancel an in-flight task (system priority) |

### Task streaming
| Variant | Purpose |
|---|---|
| `TaskStreamStart { task_id, data }` | Stream begin marker |
| `TaskStreamData { task_id, data }` | Incremental chunk |
| `TaskStreamEnd { task_id, data }` | Stream end marker |

---

## `AgentRunner`

The runner wires agents to the bus, manages parallel tokio tasks, and coordinates
shutdown.

```rust
let bus = Arc::new(LocalAgentBus::new());
let clock = Arc::new(SystemClock::new());

let runner = AgentRunner::new("main-runner", bus.clone(), clock);

runner.add_agent(Arc::new(agent_a)).await?;
runner.add_agent(Arc::new(agent_b)).await?;

// Blocks until shutdown is requested.
runner.run().await?;
```

### What `run()` does

1. Subscribes itself to the bus to handle registry and shutdown messages
2. Subscribes each agent to the bus
3. Calls `agent.setup(bus, registry)` on every agent sequentially
4. Spawns each agent's `run()` in its own tokio task (all run in parallel)
5. Waits on a `Notify` for a shutdown signal
6. Sends `End` to all root agents (no parent), then waits up to **10 seconds** per task
7. Calls `bus.stop()`

### Shutdown

```rust
runner.end(Some("session complete".to_string())).await;   // graceful
runner.cancel(Some("user hung up".to_string())).await;    // immediate
```

`end` / `cancel` are idempotent — calling them twice is safe.

### Hierarchy

Root agents are agents with no parent. On shutdown, the runner only sends `End` /
`Cancel` to root agents. Children must propagate shutdown themselves (e.g. by
watching the parent or reacting to their own pipeline end).

---

## `AgentRegistry`

The registry is built automatically by the runner. Each runner maintains:
- `local` — agents registered in this runner
- `remote` — agents discovered via `AgentRegistry` bus broadcasts from other runners

```rust
// Look up any agent (local or remote)
let info: Option<AgentInfo> = registry.get("transcription-agent").await;

// Watch for an agent to become ready (fires immediately if already registered)
registry.watch("transcription-agent", Arc::new(|info| Box::pin(async move {
    println!("Agent {} is ready (active={})", info.name, info.active);
}))).await;
```

`AgentInfo` fields:

```rust
pub struct AgentInfo {
    pub name: String,
    pub runner: String,
    pub parent: Option<String>,
    pub active: bool,
    pub bridged: bool,
    pub started_at: Option<f64>,  // unix timestamp
}
```

---

## `TaskContext` — inter-agent task dispatch

`TaskContext` wraps the bus and provides a structured request/response pattern with
streaming and cancellation.

```rust
let task_ctx = Arc::new(TaskContext::new(bus.clone()));
```

### Dispatch and await

```rust
let handle: TaskHandle = task_ctx.dispatch(
    "orchestrator",             // source agent name
    "search-agent",             // target agent name
    Some("web_search".to_string()),
    Some(json!({ "query": "rustvani architecture" })),
).await?;

let result: TaskResult = handle.await_completion(
    Some(Duration::from_secs(30))
).await?;

println!("status={:?} response={:?}", result.status, result.response);
```

### Stream updates

```rust
let (updates, final_result) = handle.stream_updates(
    Some(Duration::from_secs(30))
).await?;

for update in updates {
    match update {
        TaskUpdate::StreamData { data } => { /* incremental chunk */ }
        TaskUpdate::Update { update } => { /* progress notification */ }
        _ => {}
    }
}
```

### Fire-and-forget with callback

```rust
let task_id = handle.task_id.clone();
task_ctx.on_update(&task_id, Arc::new(|update| Box::pin(async move {
    // fires on every update without blocking the caller
    println!("task update: {:?}", update);
}))).await;
```

### Responding to a task (in the receiving agent)

```rust
// Send incremental chunks
task_ctx.stream_data("search-agent", "orchestrator", task_id.clone(), Some(json!({ "chunk": "..." }))).await;

// Finish
task_ctx.complete_task(
    "search-agent", "orchestrator", task_id,
    TaskStatus::Completed,
    Some(json!({ "results": [...] })),
).await;
```

### Urgent responses (system queue)

Use `urgent_response` / `urgent_update` when the response must skip ahead of any
pending data messages:

```rust
task_ctx.urgent_response("agent", "caller", task_id, TaskStatus::Cancelled, None).await;
```

### Cancel a task

```rust
task_ctx.cancel_task("caller", "search-agent", task_id, Some("user interrupted".to_string())).await;
```

---

## `TaskStatus`

```rust
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}
```

---

## Building a custom agent

Implement `Agent` + `BusSubscriber` directly, or compose `BaseAgent`:

```rust
use std::sync::Arc;
use async_trait::async_trait;
use rustvani::agents::{Agent, BaseAgent, AgentBus, AgentRegistry, BusMessage, BusPayload};
use rustvani::clock::BaseClock;
use rustvani::error::Result;
use rustvani::observer::BaseObserver;
use rustvani::pipeline::PipelineTask;

pub struct MyAgent {
    inner: BaseAgent,
}

impl MyAgent {
    pub fn new(pipeline: PipelineTask) -> Arc<Self> {
        Arc::new(Self {
            inner: BaseAgent::new("my-agent", pipeline, false, true),
        })
    }
}

#[async_trait]
impl rustvani::agents::BusSubscriber for MyAgent {
    fn name(&self) -> &str { self.inner.name() }

    async fn on_bus_message(&self, message: BusMessage) {
        // Handle custom payloads before delegating to BaseAgent
        if let BusPayload::TaskRequest { task_id, task_name, payload } = &message.payload {
            // ... handle the task
            return;
        }
        self.inner.on_bus_message(message).await;
    }
}

#[async_trait]
impl Agent for MyAgent {
    fn parent(&self) -> Option<&str> { self.inner.parent() }
    async fn setup(&self, bus: Arc<dyn AgentBus>, registry: Arc<AgentRegistry>) -> Result<()> {
        self.inner.setup(bus, registry).await
    }
    async fn run(&self, clock: Arc<dyn BaseClock>, observer: Option<Arc<dyn BaseObserver>>) -> Result<()> {
        self.inner.run(clock, observer).await
    }
    async fn end(&self, reason: Option<String>) -> Result<()> { self.inner.end(reason).await }
    async fn cancel(&self, reason: Option<String>) -> Result<()> { self.inner.cancel(reason).await }
    fn active(&self) -> bool { self.inner.active() }
    fn bridged(&self) -> bool { self.inner.bridged() }
    fn ready(&self) -> bool { self.inner.ready() }
}
```

---

## Shutdown flow (end-to-end)

```
runner.end()
  → sends BusPayload::End to all root agents
  → BaseAgent.on_bus_message receives End
  → injects Frame::end() into PipelineTask via push_tx
  → pipeline processes EndFrame: tools stop, connections return
  → pipeline_task.run() returns
  → agent tokio task exits
  → runner waits (up to 10s) for all tasks
  → bus.stop() — aborts all dispatch loops
```

Hard cancel follows the same path with `CancelFrame` instead, plus the
`CancellationToken` cascade in `OpenAILLMHandler` that aborts in-flight tool calls.
