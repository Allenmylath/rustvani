# Changelog

## 0.3.0 — 2026-06-11

Rust-native rewrite of the agents module internals plus Pipecat-aligned
worker semantics (job routing, cancellation cleanup, ready-gating, frame
bridging via pipeline edges). **Breaking release** — all changes below are
in `rustvani::agents` unless noted.

### Bus (`agents::bus`)

- **Breaking:** `BusSubscriber::on_bus_message` now takes `Arc<BusMessage>`
  instead of `BusMessage`. Messages fan out as a single `Arc` — no deep
  clones per subscriber.
- **Breaking:** `BusMessage` gained a `seq: u64` field, stamped by the bus
  in `send()` before fan-out (total-order debugging across agents).
  Construct messages with the new `BusMessage::new(source, target, payload)`
  instead of a struct literal.
- `LocalAgentBus` internals rewritten: per-subscriber dispatch tasks built
  on an unbounded system channel + bounded data channel (default capacity
  256, configurable via `LocalAgentBus::with_capacity`), drained with a
  `biased tokio::select!`. System messages (End/Cancel/Activate/urgent task
  replies/registry) are always delivered before queued data messages and
  are never dropped; data messages are dropped (counted + rate-limited
  warn) when a subscriber's channel is full — control never drops, data
  never blocks.
- `subscribe()` now rejects duplicate names with an error (they previously
  coexisted silently).
- `unsubscribe()`/`stop()` let in-flight handlers finish (2 s grace, then
  abort) and deliver pending system messages before exiting.
- `send()` after `stop()` is a no-op. `send()` itself is lock-free
  (`arc-swap` subscriber snapshot).
- New: `LocalAgentBus::dropped_count(name)`, `DEFAULT_DATA_CAPACITY`.

### BaseAgent (`agents::base`)

- **Breaking:** `BaseAgent::new`'s `bridged` parameter changed from `bool`
  to `Option<Vec<String>>`: `None` = not bridged, `Some(vec![])` = accept
  bridged frames from any source, `Some(names)` = accept only from the
  listed peers. The `Agent::bridged() -> bool` accessor is unchanged
  (`true` when `Some`).
- Task routing implemented (Pipecat `BaseWorker` parity): register job
  handlers with `BaseAgent::on_task(name, handler)` /
  `on_task_default(handler)`. Handlers receive a `TaskRequestCtx` (new
  type) with `complete` / `stream_start` / `stream_data` / `stream_end`
  helpers; each handler runs in its own tokio task. Requests with no
  matching handler fail fast with a terminal `Failed` response.
- `TaskCancel` aborts the running handler and always sends the requester a
  terminal `Cancelled` response.
- Lifecycle cleanup: on `end()`/`cancel()`/`run()` return, all in-flight
  jobs are cancelled with terminal responses and all pending dispatched
  handles are failed (`TaskContext::fail_all_pending`) — a requester is
  never left hanging on a dead agent.
- `end()`/`cancel()` are now single-shot (repeat calls are no-ops) and
  cascade to child agents (see Runner below).
- New: `BaseAgent::bridged_pipeline(...)` convenience constructor,
  `with_output_edge`, `task_ctx()`, `pipeline()`, `active_flag()` accessors.
  New exported types: `TaskHandler`, `TaskRequestCtx`.
- Internals: bus/registry now `OnceCell` (set once in `setup()`, which now
  errors if called twice).

### TaskContext (`agents::task`)

- **Breaking:** `TaskContext::new(bus)` is now
  `TaskContext::new(bus, registry)`.
- `dispatch()` now ready-gates: if the target is not in the registry it
  waits (watch-based, no polling) up to 10 s
  (`DEFAULT_READY_TIMEOUT`) and errors on timeout instead of silently
  dropping the request. `dispatch_with(...)` exposes the timeout
  (`None` = send immediately without gating).
- New: `fail_all_pending(reason)`, `stream_start`, `stream_end`.
  `UpdateHandler` is now exported.
- `TaskHandle::await_completion` now skips non-terminal updates (stream
  chunks/progress) instead of erroring on them.

### Bus edges (`agents::edges` — new module)

- `BusOutputEdge`: a tail-of-pipeline `FrameProcessor` that forwards every
  frame through unchanged and publishes non-excluded frames to configured
  peer agents (empty = broadcast) as `BusPayload::Frame`. Lifecycle,
  task-control, processor-control, error, and heartbeat frames are never
  published. Publishing is gated by the owning agent's activation flag
  (inactive agents also skip injecting bridged input) — Pipecat-style
  handoff between several brains and one transport pipeline.

### Registry (`agents::registry`)

- New: `children_of(name)`, `mark_finished(name)`, `wait_finished(name)`
  (per-agent finished signal used by the End cascade).

### Runner (`agents::runner`)

- Shutdown now cascades through the agent tree: `End` goes to root agents
  only; an agent receiving `End` first forwards it to its children and
  waits (5 s per child, runner's 10 s join timeout as backstop) for them to
  finish before pushing its own `EndFrame`. `Cancel` propagates without
  waiting. The runner validates parent references at `run()` time.

### Frames (`crate::frames`)

- **Breaking (type change):** `AudioRawData.audio` is now `bytes::Bytes`
  instead of `Vec<u8>` — cloning an audio frame no longer copies the
  payload. `AudioRawData::new`, `Frame::input_audio`, `Frame::output_audio`
  accept `impl Into<Bytes>`, so existing `Vec<u8>` callers compile
  unchanged. Code that needs `Vec<u8>` back uses `.to_vec()` (explicit
  copy).

### Dependencies

- Added `arc-swap` (lock-free subscriber snapshot) and `bytes`
  (cheap-clone audio payloads). `tokio-util` (already a dependency) now
  also provides `CancellationToken`.

### Misc

- New example: `examples/bridged_agents.rs` — two bridged agents
  demonstrating dispatch, streaming reply, and cancel.
- Fixed stale assertions in the sixtydb STT config test and a
  non-exhaustive match in `examples/channel_pipeline.rs`.
