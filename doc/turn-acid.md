# The Turn as a Transaction — Turn-Level ACID (Phase 1)

> **Status:** design doc. **Scope:** Phase 1 only — make the conversation context ACID for a
> single turn on the existing, non-agentic pipeline. **No agents, no coordinator, no
> Cloacina, no billing/durability changes.** The agentic coordinator and durable side-effect
> tiers are out of scope and appear only as a one-line pointer in [§8](#8-future-work).

## Why this doc exists

The **turn** (user speaks → bot responds) is rustvani's sacrosanct unit. Its *input* half is
already close to ACID: the **TurnGate** in [`sarvam.rs`](../src/services/stt/sarvam.rs) keeps
a monotonic `epoch`, exactly-once release via `Mutex<Option>::take()`, ledger-based
transcript attribution, and linearized emission. That transactional discipline **dies at the
STT boundary**: once text enters
[`LLMUserAggregator`](../src/processors/llm_user_aggregator.rs) → LLM →
[`LLMAssistantAggregator`](../src/processors/llm_assistant_aggregator.rs) there is no epoch,
no staging, and no rollback. The conversation context is mutated in place, immediately, with
no way to undo a turn that the user interrupts.

This doc closes exactly that gap, on the current pipeline.

---

## 1. The turn as the transaction boundary

A turn **opens** at `VADUserStartedSpeaking` and **commits** when the bot response finalizes
(`LLMFullResponseEnd`). Everything between — the user transcript, LLM reasoning, tool calls,
and assistant text — is the transaction body. **Barge-in starts the *next* turn and must
abort the current one.**

The existing `active_user_turn_id` cell (shared between the aggregators and
`AudioCaptureProcessor`) is the seed of a turn identity — but today it never reaches the LLM
layer where the context is written.

---

## 2. Gaps this closes

- **Atomicity — broken.** [`add_user_message`](../src/context/mod.rs),
  [`add_assistant_message`](../src/context/mod.rs), and `add_tool_result` mutate
  `LLMContext.messages` *in place, immediately*. On interruption the assistant aggregator
  [discards its partial text](../src/processors/llm_assistant_aggregator.rs) — but the user
  message is already committed, and any tool results written mid-flight stay. The result is a
  half-mutated context.
- **Consistency — not enforced.** Nothing guarantees that every assistant `tool_calls` has a
  matching `ToolResult`. An interruption between `FunctionCallInProgress` and
  `FunctionCallResult` orphans the call, and the next request sends an invalid message
  sequence (which providers reject).
- **Isolation — within the single pipeline.** A turn's writes must not interleave with the
  next turn's after a barge-in; today there is no epoch to scope or fence them.

*(Durability is already partially covered by the existing billing/transcript path and is out
of scope for this doc.)*

---

## 3. Mechanisms

### 3.1 Turn epoch

`LLMContext` carries a monotonic `epoch` (turn identity), bumped by `begin_turn()` when the
user aggregator flushes a turn and triggers inference. The epoch travels *with* the context
(the `LLMContextFrame` carries the `Arc<Mutex<LLMContext>>`), so no separate token needs
threading. `begin_turn()` also discards any staged round left by a previously interrupted
turn — an implicit rollback at the start of each turn.

For Phase 1 (synchronous inference, one turn's work in flight at a time) the epoch is the
**turn identity seed**; full epoch *fencing* — dropping a stale result that arrives after a
barge-in — only matters once agent work is decoupled and async, which is future work.

> **The epoch tags; it does not auto-cancel.** It does **not** destroy in-flight work. See
> the [scope note](#scope-note--barge-in--cancellation) for why this distinction matters.

### 3.2 Transactional context (staging buffer)

A **staging buffer** over `LLMContext`. The LLM stages an assistant `tool_calls` message and
its `ToolResult`s (`stage_assistant_tool_calls` / `stage_tool_result`) instead of writing
`messages` directly. `commit()`:

1. **validates consistency** — drops any assistant `tool_calls` whose ids are not all
   answered by a staged `ToolResult` (and the dangling results), so a partial round can
   never reach `messages`; then
2. **splices** the surviving staged messages into `messages` **atomically**.

`rollback()` discards the staging buffer. This is the **chokepoint** that keeps `messages`
always valid — the next API request can never observe an assistant `tool_calls` without its
matching results.

**Commit granularity is the tool round, not turn end.** The Dhara transition hook
([`dhara/manager.rs`](../src/dhara/manager.rs) `apply_node_to_context`) runs *between* tool
rounds and manipulates `messages` directly (it can `messages.clear()` on a context-reset
transition). So each round's staged messages are committed into `messages` **before** the
hook runs, and the staging buffer is **not** included in `to_api_messages()`. Plain user and
assistant **text** messages are committed directly (via `add_user_message` /
`add_assistant_message`) — a single message carries no orphan risk, and rolling back a
user turn would strand follow-up references (*"…and tomorrow?"*), so only the **tool round**
is transactional.

### 3.3 Interruption → abort

When the bot is interrupted mid-response, its `run_inference` task is aborted (the future is
dropped at its current `await`), so any staged-but-uncommitted round would otherwise linger.
Two things clear it:

- the **assistant aggregator** calls `context.rollback()` in its `Interruption` handler
  (prompt cleanup), and
- the **LLM** calls `context.rollback()` at the *start* of the next `run_inference`
  (safety net — each inference starts from clean committed history).

There is **no context mutation on the abort path.** Existing UX is preserved: the assistant
partial is still recorded to the transcript as `interrupted` (billing) — it is simply never
spliced into context.

---

## 4. Touch points (existing code)

| File | Change |
|---|---|
| [`src/context/mod.rs`](../src/context/mod.rs) | Add the staging buffer + `epoch` and the `begin_turn` / `stage_*` / `commit` / `rollback` API, with orphaned-tool-call repair in `commit`. `to_api_messages` unchanged. |
| [`src/services/llm/openai.rs`](../src/services/llm/openai.rs) | `rollback()` stale staging at `run_inference` start; `stage_assistant_tool_calls` + `stage_tool_result` during a round; `commit()` at the round boundary, before the Dhara transition hook. |
| [`src/processors/llm_assistant_aggregator.rs`](../src/processors/llm_assistant_aggregator.rs) | `rollback()` on `Interruption` (prompt cleanup of an aborted round). |
| [`src/processors/llm_user_aggregator.rs`](../src/processors/llm_user_aggregator.rs) | `begin_turn()` on flush — bump the epoch and discard any leftover staged round before triggering inference. |

---

## 5. Worked scenarios

1. **Clean turn** — open epoch `E`, stage user + assistant, `commit(E)` → context valid.
2. **Barge-in mid-LLM** — `Interruption` bumps the epoch and `rollback(E)`; context
   untouched; assistant partial recorded as `interrupted` (existing behavior preserved).
3. **Interruption between tool call and result** — `rollback(E)` drops the orphaned call, so
   the next request never sees an assistant `tool_calls` without its `ToolResult` (consistency).
4. **Empty / no-op turn** — nothing staged, `commit` is a no-op (mirrors the existing
   empty-turn skip in the user aggregator).

**Outcome: Atomicity, Consistency, and Isolation for the conversation on the current
pipeline, with no change to agents, billing, or external side effects.**

### 5a. User-facing impact

On a clean, uninterrupted turn the user notices **nothing** — behavior is identical. The
entire impact is on **interruption paths**, which in voice are hit constantly. Today those
paths corrupt the conversation; Phase 1 makes them clean.

- **Interrupt during a lookup (Consistency).**
  *"Status of order 4471?" → "Let me check—" [tool call in flight] → "Sorry, 4472."*
  Today the killed turn leaves an assistant `tool_call` with no matching `ToolResult`; the
  next request is malformed → provider `400` or a confused model → the bot stalls or
  re-answers about 4471. **After:** `rollback()` drops the orphan; the bot cleanly looks up
  4472.
- **No residue from an abandoned turn (Isolation).**
  *"Tell me about refunds." → "Refunds take 14 day—" → "Never mind, store hours?"*
  Today the partial assistant text is discarded but any tool result written mid-turn stays,
  and there is no epoch scoping the staging. **After:** `rollback(E)` clears all of turn `E`'s
  staged writes atomically, so turn `E+1` starts from clean context — no fragments of the
  killed turn bleed in.
- **Rapid mid-answer correction (Atomicity).**
  *"Book the 3pm demo." → "Booking the 3pm slo—" → "No wait, 4pm."*
  Today the abandoned half-turn leaves torn assistant/tool state and the bot may re-confirm
  3pm or conflate both. **After:** the response transaction rolls back as a unit — the
  model's next view of history is exactly "user wants 4pm."

**Net:** Phase 1 removes the class of *"the bot got weird right after I interrupted it"*
bugs. No new features, happy path unchanged — the bot simply recovers from interruptions the
way a human would, which (because users interrupt voice bots constantly) is felt on a large
fraction of real conversations.

#### Scope note — barge-in ≠ cancellation

In Phase 1 the LLM tool loop in [`openai.rs`](../src/services/llm/openai.rs) is *synchronous*
(the tool is `await`ed inside the turn), so there is no background task whose result can
"land later." The case where a user does small talk while a slow lookup runs **and still
expects the answer** requires *decoupled async execution* — a Phase 2 capability.

The Phase-1 epoch must therefore **tag, not auto-cancel**. This keeps the door open for
Phase 2 to separate two concepts that are easy to conflate:

| Concept | What it does | Triggered by |
|---|---|---|
| **Turn epoch** (attribution / isolation) | *Tags* every result so it is never silently merged into the wrong turn's context | every new turn (barge-in) |
| **Cancellation** of in-flight work | Actually abandons the async task | an **explicit, intent-classified** decision — *not* barge-in by itself |

So in Phase 2: a *correction* ("never mind Tokyo") → `TaskCancel`; *filler while waiting* →
let the task run and surface its result as a tagged deferred answer ("by the way, those Tokyo
flights are $800"). **Barge-in starts a new turn; it must not by itself cancel async work.**

---

## 6. ACID checklist (Phase 1)

| Property | Mechanism |
|---|---|
| **Atomicity** | `TurnTransaction` staging + single `commit` / `rollback` |
| **Consistency** | `commit` validates `tool_call ↔ result` pairing and message ordering before splicing |
| **Isolation** | turn-epoch token stamped on the context frame; abort bumps the epoch |
| **Durability** | unchanged — existing billing / transcript path (out of scope) |

---

## 7. Open questions / risks

- **Speculative / pipelined next turn vs strict serialization** — recommend strict serialize.
- **Unifying the TurnGate `epoch` with `active_user_turn_id`** without disturbing TurnGate
  invariants — confirm the mirror direction and ownership.
- **Exact `commit` / `rollback` firing points** relative to the existing `LLMFullResponseEnd`
  / `Interruption` handling in the assistant aggregator.

---

## 8. Future work

A later, separately-scoped effort extends this turn epoch across the agent bus (coordinator
as transaction manager, stale-epoch fencing) and adds a durable side-effect tier
(transactional outbox; a durable workflow engine only for genuine multi-step side-effect
DAGs). The bus-fencing + coordinator half is now specified and landed as the Phase 2
foundation — see [`turn-acid-phase2.md`](turn-acid-phase2.md). The durable side-effect tier
remains Phase 3. See [`agents.md`](../agents.md) for the multi-agent layer.
