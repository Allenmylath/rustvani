# Agents — rustvani

Multi-agent coordination in rustvani sits on top of the frame pipeline. Each agent
owns a `PipelineTask`, communicates through an `AgentBus`, and is orchestrated by
an `AgentRunner`. The system is entirely async-native (tokio) with no global state.

---

## Core components

| Component | File | Role |
|---|---|---|
| `Agent` trait | `src/agents/base.rs` | Contract every agent must satisfy |
| `BaseAgent` | `src/agents/base.rs` | Concrete default implementation + task router |
| `AgentBus` / `LocalAgentBus` | `src/agents/bus.rs` | Message broker with two-priority channels |
| `BusPayload` | `src/agents/bus.rs` | All message types on the bus |
| `BusOutputEdge` | `src/agents/edges.rs` | Tail-of-pipeline processor publishing frames to the bus |
| `AgentRunner` | `src/agents/runner.rs` | Orchestrates agents in parallel tokio tasks |
| `AgentRegistry` | `src/agents/registry.rs` | Local + remote agent discovery, finished signals |
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

- `setup` — called once before `run`; stores bus and registry references
  (set-once; calling twice errors)
- `run` — blocks until the pipeline finishes; the tokio task lives here
- `end` — graceful shutdown; cascades to children, cancels in-flight jobs,
  then injects `EndFrame` into the pipeline (single-shot — repeats are no-ops)
- `cancel` — hard abort; same as `end` but propagates to children without
  waiting and injects `CancelFrame`
- `bridged` — if `true`, the agent receives `BusPayload::Frame` messages injected
  from the bus into its pipeline (on `BaseAgent` this is configured as a peer
  filter, see below)

---

## `BaseAgent` — the default implementation

`BaseAgent` implements `Agent` in full. Extend it by wrapping it or composing it
inside your own struct.

```rust
let pipeline_task = PipelineTask::new(pipeline, PipelineParams::default());

let agent = BaseAgent::new(
    "my-agent",          // name — must be unique on the runner
    pipeline_task,
    None,                // bridged peer filter (see below)
    true,                // active_on_start
);

// Optional: declare a parent for hierarchical shutdown
let child_agent = BaseAgent::new("child", child_task, None, true)
    .with_parent("my-agent");
```

The `bridged` parameter is `Option<Vec<String>>`:

| Value | Meaning |
|---|---|
| `None` | not bridged — `BusPayload::Frame` messages are ignored |
| `Some(vec![])` | accept bridged frames from **any** source |
| `Some(vec!["voice"])` | accept bridged frames only from the listed peers |

Bridged input is also gated by activation: an **inactive** agent does not
inject bridged frames (several brains can bridge to one transport with
exactly one active — Pipecat-style handoff).

### Task handlers — `BaseAgent` is the task router

```rust
let agent = BaseAgent::new("search-agent", task, None, true)
    .on_task("web_search", Arc::new(|ctx: TaskRequestCtx| Box::pin(async move {
        // ctx bundles task_id / payload / source / agent_name / task_ctx
        ctx.stream_data(Some(json!({ "chunk": "..." }))).await;
        ctx.complete(TaskStatus::Completed, Some(json!({ "results": [] }))).await;
    })))
    .on_task_default(Arc::new(|ctx| Box::pin(async move {
        ctx.complete(TaskStatus::Failed, None).await;
    })));
```

- Each handler runs in its **own tokio task** — the bus dispatch loop is never
  blocked by job work.
- A request with no matching handler (and no default) is answered immediately
  with `TaskStatus::Failed` and a `"no handler"` payload.
- `TaskCancel` aborts the handler task and always sends the requester a
  terminal `Cancelled` response.
- When the agent ends/cancels (or its `run()` returns), every in-flight job is
  cancelled with a terminal response and every handle the agent itself is
  awaiting is failed — requesters are never left hanging.

`BaseAgent.run()` does four things in order:
1. Sets `ready = true`
2. Broadcasts `AgentReady` on the bus so other agents and the registry know it is live
3. Awaits `pipeline_task.run()` — blocks until the pipeline finishes
4. Cleans up jobs/pending handles and signals `finished` in the registry

---

## Message bus (`LocalAgentBus`)

### Sending a message

```rust
bus.send(BusMessage::new(
    "sender-name",
    Some("target-name".to_string()), // None = broadcast to all
    BusPayload::Activate { args: None },
)).await;
```

Agents never receive their own messages (`source == self.name` is filtered out).
The bus stamps a `seq: u64` on every message before fan-out (total-order
debugging across agents) and delivers it as `Arc<BusMessage>` — fan-out never
deep-clones the payload.

### Priority channels and the drop policy

Every subscriber has two channels drained by one dispatch task
(`biased tokio::select!` — system is always polled first):

| Channel | Payloads routed here | Behavior |
|---|---|---|
| system (unbounded) | `End`, `Cancel`, `Activate`, `Deactivate`, `AgentReady`, `AgentRegistry`, `AgentError`, `TaskResponseUrgent`, `TaskUpdateUrgent`, `TaskCancel` | never blocks, never drops |
| data (bounded, default 256) | everything else (`Frame`, `TaskRequest`, `TaskResponse`, `TaskStream*`, `TaskUpdate`) | `try_send`; dropped + counted when full |

Rationale (realtime voice): **control never drops, data never blocks**. A slow
background agent must never stall the producer; frames are droppable, control
messages are not. Drops are counted per subscriber
(`LocalAgentBus::dropped_count`) and logged (first drop, then every 100th).
The capacity is configurable via `LocalAgentBus::with_capacity`. Duplicate
subscriber names are rejected at `subscribe()`.

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
`Cancel` to root agents — propagation through the tree is automatic: a
`BaseAgent` receiving `End` first forwards `End` to each of its children
(looked up via the registry), waits up to 5 s per child for its `run()` to
return (`AgentRegistry::wait_finished`), and only then pushes its own
`EndFrame`. `Cancel` propagates the same way but without waiting. The runner's
10 s join timeout is the backstop. Dangling parent references are warned about
at `run()` time.

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
streaming and cancellation. `BaseAgent` constructs one automatically in
`setup()` — get it with `agent.task_ctx()`. To build one manually:

```rust
let task_ctx = Arc::new(TaskContext::new(bus.clone(), registry.clone()));
```

### Ready-gating

`dispatch()` checks the registry first: if the target agent has not announced
readiness yet, it waits (watch-based, no polling) up to 10 s
(`DEFAULT_READY_TIMEOUT`) and returns an error on timeout — requests to agents
that have not subscribed yet are no longer silently dropped. Use
`dispatch_with(..., ready_timeout)` to change or skip the gate (`None` sends
immediately).

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

Inside a `BaseAgent` task handler, use the `TaskRequestCtx` helpers
(`ctx.stream_data(...)`, `ctx.complete(...)`) — they thread the names and id
for you. The underlying `TaskContext` calls are:

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

(`stream_start` / `stream_end` exist for the stream boundary markers.)

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
            inner: BaseAgent::new("my-agent", pipeline, None, true),
        })
    }
}

#[async_trait]
impl rustvani::agents::BusSubscriber for MyAgent {
    fn name(&self) -> &str { self.inner.name() }

    async fn on_bus_message(&self, message: Arc<BusMessage>) {
        // Handle custom payloads before delegating to BaseAgent.
        // (For TaskRequest specifically, prefer BaseAgent::on_task handlers.)
        if let BusPayload::AgentError { error } = &message.payload {
            log::warn!("peer error: {error}");
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

## Bridging pipelines (`BusOutputEdge`)

Any pipeline becomes bridgeable by placing a `BusOutputEdge` at its tail —
no custom mid-pipeline proxy needed. The easiest wiring:

```rust
let agent = BaseAgent::bridged_pipeline(
    "brain",
    vec![my_processor],          // user processors; edge appended at the tail
    PipelineParams::default(),
    vec!["transport".into()],    // peers: accept bridged input from + publish to
    true,                        // active_on_start
);
```

The edge forwards every frame through unchanged and, when the agent is active,
publishes non-excluded frames as `BusPayload::Frame` to the peers (empty list
= broadcast). Lifecycle, task-control, processor-control, error and heartbeat
frames are never published. For manual wiring use `BusOutputEdge::new` /
`with_exclude` + `to_processor()` + `BaseAgent::with_output_edge`.

**Two-way bridges:** each side must exclude the frame kinds the peer
publishes (`BusOutputEdge::with_exclude`), or re-injected frames will
ping-pong between the pipelines forever. See `examples/bridged_agents.rs`.

---

## Shutdown flow (end-to-end)

```
runner.end()
  → sends BusPayload::End to root agents only
  → BaseAgent.on_bus_message receives End → BaseAgent.end() (single-shot)
      → forwards End to each child, waits for child run() to return (≤5s each)
      → cancels in-flight jobs (requesters get TaskResponse::Cancelled)
      → fails its own pending task handles (no awaiter hangs)
      → injects Frame::end() into PipelineTask via push_tx
  → pipeline processes EndFrame: tools stop, connections return
  → pipeline_task.run() returns → registry.mark_finished(agent)
  → agent tokio task exits
  → runner waits (up to 10s) for all tasks
  → bus.stop() — dispatch loops drain pending system messages, then exit
```

Hard cancel follows the same path with `CancelFrame` instead (children are
not waited for), plus the `CancellationToken` cascade in `OpenAILLMHandler`
that aborts in-flight tool calls.
