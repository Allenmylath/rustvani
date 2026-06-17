# The Turn as a Transaction — Turn-Level ACID (Phase 2, Foundation)

> **Status:** design doc + landed foundation. **Scope:** extend the Phase 1 turn transaction
> **across asynchronous agent dispatch over the bus** — epoch fencing on `BusMessage`, a
> coordinator that commits/rolls back the turn, and barge-in that cancels in-flight agent work.
> **Out of scope:** the *barge-in ≠ cancellation* intent split and deferred-result delivery
> (Phase 2b), and the durable side-effect tier (Phase 3). See [§7](#7-not-in-this-round).

## Why this doc exists

[Phase 1](turn-acid.md) made one turn's conversation context ACID **in-process**: `LLMContext`
([`src/context/mod.rs`](../src/context/mod.rs)) gained a monotonic `epoch` plus a staging buffer
(`begin_turn` / `stage_*` / `commit` / `rollback`), and the synchronous tool loop in
[`openai.rs`](../src/services/llm/openai.rs) stages each tool round and commits at the round
boundary. Phase 1 deliberately left a **seam**, called out in its
[scope note](turn-acid.md#scope-note--barge-in--cancellation):

- the epoch only *tags* the turn — it is not yet used to reject anything;
- tool execution is synchronous (the tool is `await`ed inside the turn), so **nothing can arrive
  late**; and
- `BusMessage` ([`src/agents/bus.rs`](../src/agents/bus.rs)) carries `seq` for ordering but **no
  turn identity**.

Phase 2 lights up that seam. Once agent work is **decoupled and async** — dispatched over the bus
and running off the turn's critical path — a slow agent's answer for turn *N* can land *after* a
barge-in has started turn *N+1*. Without fencing, that late answer contaminates the wrong turn.
This round delivers the two load-bearing pieces that make the agentic turn atomic, consistent, and
isolated across the bus, plus minimal cancellation on barge-in.

---

## 1. The transaction boundary, extended

Phase 1's boundary is unchanged: a turn opens when the user aggregator flushes
(`begin_turn()` bumps the epoch — [`llm_user_aggregator.rs`](../src/processors/llm_user_aggregator.rs))
and commits when the response finalizes. What changes is **where the work happens**. In the
agentic path the LLM/tool work is farmed to **worker agents over the bus** instead of `await`ed
inline, so the turn's body now spans asynchronous, independently-cancellable tasks. The
transaction must hold across that gap.

---

## 2. Gaps this closes (beyond Phase 1)

- **Isolation across the bus — newly broken by async.** Phase 1's isolation was *within the single
  pipeline*. The moment agent work is async, a result computed for turn *N* can be delivered during
  turn *N+1*. There was no token on the wire to detect this.
- **Atomicity across dispatch.** The staging buffer is in-process; a turn whose work lives in other
  agents needs a coordinator to decide, as one actor, whether the turn `commit()`s or `rollback()`s.
- **Cancellation vs. tagging.** Barge-in must be able to *stop* in-flight agent work (not just
  abandon a local future) — while still keeping tagging and cancellation as separate decisions
  (see [§5](#5-barge-in--cancellation)).

---

## 3. Mechanisms

### 3.1 Bus epoch fencing

`BusMessage` gains `turn_epoch: Option<u64>` next to `seq`
([`bus.rs`](../src/agents/bus.rs), `with_turn_epoch` builder). It is `None` for epoch-agnostic
traffic (lifecycle, registry, bridged frames). The coordinator stamps `Some(epoch)` on every task
it dispatches via [`TaskContext::dispatch_fenced`](../src/agents/task.rs); the executing agent
captures it on `TaskRequestCtx.turn_epoch` and **echoes it back** on its `TaskResponse`
(`complete_task_fenced`) — defense-in-depth and observable on the wire.

The **authoritative fence** lives in the coordinator: a `dispatched: HashMap<task_id, epoch>` map
recording the epoch each task was dispatched under. When a result returns, the coordinator compares
its recorded epoch against the current epoch; a mismatch means the turn was superseded.

> **The epoch tags; cancellation is separate.** Advancing the epoch records *"this is a new
> turn"* — by itself it does **not** kill the agent still computing turn *N*'s answer. That is what
> keeps the door open for Phase 2b's *defer* path. See [§5](#5-barge-in--cancellation).

### 3.2 Coordinator as transaction manager

`AgenticCoordinator` ([`src/agents/coordinator.rs`](../src/agents/coordinator.rs)) plays the
2PC/saga coordinator role for a turn. It owns the shared `Arc<Mutex<LLMContext>>`, a `TaskContext`
(its bus identity), the live epoch, and the `dispatched` map. Per turn:

1. **Open** — `open_turn()` reads the context's current `epoch()` and adopts it as live.
   `begin_turn()` itself stays upstream (the user aggregator owns it); the coordinator only reads.
2. **Dispatch fenced** — `dispatch()` sends fenced work to worker agent(s) and records
   `task_id -> epoch`.
3. **Stage / fence** — `stage_result(task_id, msg)` checks the fence *first*. Current-epoch result
   → `stage_message` into context (`FenceOutcome::Staged`). Stale or unknown task → dropped
   (`FenceOutcome::Quarantined`), context untouched.
4. **Commit** — `commit()` splices the staged round into `messages` atomically (Phase 1
   `LLMContext::commit`, which still drops orphan tool-call rounds for consistency).

This reuses the entire Phase 1 staging API unchanged; the coordinator is the cross-bus actor that
drives it.

### 3.3 Cancellation propagation on barge-in

`on_interruption(reason)` is the cross-bus version of Phase 1's single-processor rollback:

- it broadcasts `TaskCancel` to every in-flight worker (`cancel_task`). `TaskCancel` is already a
  **system-priority** message ([`bus.rs`](../src/agents/bus.rs) `is_system`), so it jumps the data
  queue; the executing `BaseAgent` aborts the job and replies `Cancelled`, and the existing
  parent/child cancel cascade ([`base.rs`](../src/agents/base.rs) `cascade_to_children`) carries it
  to sub-agents — **no new cancel primitive**; and
- it `rollback()`s the staging buffer so the shared context retains nothing from the killed turn.

The epoch advances on the next `begin_turn`; the fence in [§3.1](#31-bus-epoch-fencing) then
quarantines any late `Cancelled`/stale reply that races in after the cancel.

---

## 4. Worked scenarios

1. **Clean async turn** — open epoch `E`, dispatch fenced work, `stage_result` (epoch matches),
   `commit(E)` → context valid, response spoken.
2. **Barge-in during slow agent work** — epoch advances to `E+1`; the late `E` result fails the
   fence (`dispatched[id] == E != E+1`) → quarantined, never staged; turn `E+1` starts clean.
3. **Explicit cancel on barge-in** — `on_interruption` broadcasts `TaskCancel` for the in-flight
   `E` tasks; the worker aborts and resolves `Cancelled`; staging rolls back.
4. **Agent failure** — a worker returns `TaskStatus::Failed`; the coordinator rolls the round back
   rather than committing a half-turn.

---

## 5. Barge-in ≠ cancellation

Carried over from Phase 1's [scope note](turn-acid.md#scope-note--barge-in--cancellation), now
realized in code — but only the *tagging* half ships this round:

| Concept | What it does | Triggered by | This round |
|---|---|---|---|
| **Turn epoch** (attribution / isolation) | *Tags* every dispatch so a result is never merged into the wrong turn | every new turn (barge-in) | **shipped** — bus `turn_epoch` + coordinator fence |
| **Cancellation** of in-flight work | Actually abandons the async task | an **explicit, intent-classified** decision | **partial** — `on_interruption` cancels *all* in-flight work; the intent split is Phase 2b |

So today, barge-in fences (always) and `on_interruption` cancels (bluntly, all in-flight work).
The finer split — a *correction* ("never mind Tokyo") → `TaskCancel`, vs. *filler while waiting*
("how long?") → let the task run and surface its result as a tagged deferred answer ("by the way,
those Tokyo flights are \$800") — needs intent classification that does not exist yet, and is
**Phase 2b**.

---

## 6. ACID checklist (Phase 2 foundation)

| Property | Mechanism |
|---|---|
| **Atomicity** | coordinator stages agent results and `commit()`s / `rollback()`s the turn as a unit |
| **Consistency** | Phase 1 `commit` still validates `tool_call ↔ result` pairing before splicing |
| **Isolation** | `turn_epoch` on `BusMessage` + the coordinator's `dispatched` fence — stale results quarantined across the bus |
| **Durability** | unchanged — existing billing / transcript path (Phase 3) |

---

## 7. Not in this round

- **Deferred result delivery (Phase 2b).** A quarantined result is currently *dropped*. Re-surfacing
  it as a tagged out-of-band answer is deferred — and must be injected as a **plain context note**,
  not a raw `ToolResult` (a standalone tool result with no matching assistant `tool_calls` in the
  current history is dropped by Phase 1's orphan repair in `commit`). The hard part is the
  *decision* (was it retracted? still relevant?), which is the intent classification below.
- **The barge-in ≠ cancellation intent split (Phase 2b)** — correction → cancel vs. filler → keep.
- **Durability of real-world side effects (Phase 3)** — defer-until-commit, transactional outbox /
  durable workflow engine, and routing billing/transcript through a durable sink.

---

## 8. Open questions / risks

- **Coordinator placement** — validated here as a struct driven from tests/agents. Wiring it into
  the live voice pipeline (replacing `OpenAILLMHandler` in agentic mode, and feeding
  `SystemFrame::Interruption` into `on_interruption` the way the assistant aggregator already
  consumes it) is the follow-up integration step.
- **Echo vs. local map** — the coordinator's `dispatched` map is authoritative; the bus epoch echo
  is defense-in-depth/observability. Threading the epoch into the caller-side `TaskUpdate` enum was
  intentionally avoided to keep the public task API stable.
- **Quarantine policy** — drop+log this round; *defer-and-resurface* is Phase 2b.
