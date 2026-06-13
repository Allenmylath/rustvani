//! `AgenticCoordinator` — the turn transaction manager (turn-level ACID, Phase 2).
//!
//! Phase 1 (`doc/turn-acid.md`) made one turn's [`LLMContext`] transactional
//! *in-process*: a staging buffer with `begin_turn` / `stage_*` / `commit` /
//! `rollback`, committed at the synchronous tool-round boundary. Phase 2 extends
//! that transaction **across asynchronous agent dispatch over the bus**.
//!
//! The coordinator plays the 2PC/saga coordinator role for a turn:
//!
//! 1. **Open** — capture the turn epoch ([`open_turn`](AgenticCoordinator::open_turn)).
//!    `begin_turn()` itself stays upstream (the user aggregator bumps the epoch on
//!    flush); the coordinator only reads it.
//! 2. **Dispatch fenced** — [`dispatch`](AgenticCoordinator::dispatch) stamps every
//!    `TaskRequest` with the turn epoch and remembers `task_id -> epoch` in the
//!    authoritative `dispatched` map.
//! 3. **Stage / fence** — [`stage_result`](AgenticCoordinator::stage_result) checks the
//!    result's task against the current epoch *first*. A result whose turn has been
//!    superseded by a barge-in is **quarantined** (dropped, never staged); a current
//!    result is staged into the shared context.
//! 4. **Commit / rollback** — [`commit`](AgenticCoordinator::commit) splices the staged
//!    round into `messages`; [`on_interruption`](AgenticCoordinator::on_interruption)
//!    cancels in-flight work and rolls the staging buffer back.
//!
//! See `doc/turn-acid-phase2.md`. Deferred (Phase 2b): the *barge-in ≠ cancellation*
//! intent split and re-surfacing a quarantined answer as a tagged deferred reply.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::context::{LLMContext, Message};
use crate::error::Result;

use super::task::{TaskContext, TaskHandle, DEFAULT_READY_TIMEOUT};

/// Outcome of attempting to stage an async agent result into the turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FenceOutcome {
    /// The result belonged to the current turn and was staged into context.
    Staged,
    /// The result belonged to a superseded turn (or an unknown task) and was
    /// dropped without touching context.
    Quarantined,
}

/// One dispatched, not-yet-resolved task and the turn it was fenced to.
struct InFlight {
    epoch: u64,
    target: String,
}

/// Coordinates one conversation's turns as transactions over the agent bus.
///
/// Holds no LLM of its own — it farms work to worker agents and integrates
/// their results into the shared [`LLMContext`] using the Phase 1 staging API.
pub struct AgenticCoordinator {
    /// Agent name this coordinator dispatches as (its bus identity).
    source: String,
    /// The shared conversation context (same `Arc` the aggregators hold).
    context: Arc<Mutex<LLMContext>>,
    /// Task dispatch/await over the bus (the coordinator agent's context).
    task_ctx: Arc<TaskContext>,
    /// The turn epoch the coordinator currently considers live. Refreshed by
    /// [`open_turn`](Self::open_turn) from `context.epoch()`.
    current_epoch: AtomicU64,
    /// Authoritative fence: every in-flight `task_id` and the epoch it was
    /// dispatched under. A result whose recorded epoch differs from
    /// `current_epoch` is stale.
    dispatched: Mutex<HashMap<String, InFlight>>,
}

impl AgenticCoordinator {
    /// Create a coordinator bound to a shared context and a task context.
    ///
    /// `source` is the bus name the coordinator dispatches as (typically the
    /// name of the agent whose `task_ctx` this is).
    pub fn new(
        source: impl Into<String>,
        context: Arc<Mutex<LLMContext>>,
        task_ctx: Arc<TaskContext>,
    ) -> Self {
        let epoch = context.lock().unwrap().epoch();
        Self {
            source: source.into(),
            context,
            task_ctx,
            current_epoch: AtomicU64::new(epoch),
            dispatched: Mutex::new(HashMap::new()),
        }
    }

    /// The epoch the coordinator currently considers live.
    pub fn current_epoch(&self) -> u64 {
        self.current_epoch.load(Ordering::Acquire)
    }

    /// Open (or re-open) the turn: read the context's current epoch and adopt
    /// it as live. Call this after the user aggregator's `begin_turn()` (which
    /// bumps the epoch) and after a barge-in advances the turn. Returns the
    /// adopted epoch.
    pub fn open_turn(&self) -> u64 {
        let epoch = self.context.lock().unwrap().epoch();
        self.current_epoch.store(epoch, Ordering::Release);
        epoch
    }

    /// Dispatch fenced agent work for the current turn and remember the task
    /// so its result can be fenced when it returns.
    pub async fn dispatch(
        &self,
        target: &str,
        task_name: Option<String>,
        payload: Option<serde_json::Value>,
    ) -> Result<TaskHandle> {
        let epoch = self.current_epoch();
        let handle = self
            .task_ctx
            .dispatch_fenced(
                &self.source,
                target,
                task_name,
                payload,
                epoch,
                Some(DEFAULT_READY_TIMEOUT),
            )
            .await?;
        self.dispatched.lock().unwrap().insert(
            handle.task_id.clone(),
            InFlight {
                epoch,
                target: target.to_string(),
            },
        );
        Ok(handle)
    }

    /// Whether `task_id` is still in flight under the current epoch.
    pub fn is_current(&self, task_id: &str) -> bool {
        let current = self.current_epoch();
        self.dispatched
            .lock()
            .unwrap()
            .get(task_id)
            .is_some_and(|f| f.epoch == current)
    }

    /// Stage an async agent result into the turn, fencing it first.
    ///
    /// If `task_id` was dispatched under the current epoch, `msg` is staged
    /// into the shared context (commit later with [`commit`](Self::commit)) and
    /// [`FenceOutcome::Staged`] is returned. Otherwise — a barge-in advanced the
    /// epoch, or the task is unknown/already resolved — the result is dropped
    /// and [`FenceOutcome::Quarantined`] is returned. Either way the task is
    /// removed from the in-flight set.
    pub fn stage_result(&self, task_id: &str, msg: Message) -> FenceOutcome {
        let current = self.current_epoch();
        let entry = self.dispatched.lock().unwrap().remove(task_id);
        match entry {
            Some(f) if f.epoch == current => {
                self.context.lock().unwrap().stage_message(msg);
                FenceOutcome::Staged
            }
            Some(f) => {
                log::info!(
                    "coordinator '{}': quarantined result for task {} (dispatched epoch {}, current {})",
                    self.source,
                    task_id,
                    f.epoch,
                    current
                );
                FenceOutcome::Quarantined
            }
            None => {
                log::debug!(
                    "coordinator '{}': result for unknown/resolved task {} dropped",
                    self.source,
                    task_id
                );
                FenceOutcome::Quarantined
            }
        }
    }

    /// Commit the staged round into the shared context atomically (Phase 1
    /// [`LLMContext::commit`]). Returns the number of messages committed.
    pub fn commit(&self) -> usize {
        self.context.lock().unwrap().commit()
    }

    /// Number of tasks still in flight (diagnostics/tests).
    pub fn in_flight(&self) -> usize {
        self.dispatched.lock().unwrap().len()
    }

    /// Abort the turn on a barge-in / interruption.
    ///
    /// Broadcasts `TaskCancel` to every in-flight worker (reusing the existing
    /// cancel machinery — the executing `BaseAgent` aborts the job and replies
    /// `Cancelled`, cascading to its children), then rolls back any staged
    /// round so the shared context retains nothing from the killed turn. The
    /// epoch advances on the next `begin_turn`; [`stage_result`](Self::stage_result)
    /// then quarantines any late reply that races in.
    pub async fn on_interruption(&self, reason: Option<String>) {
        let in_flight: Vec<(String, String)> = {
            let mut guard = self.dispatched.lock().unwrap();
            guard
                .drain()
                .map(|(task_id, f)| (task_id, f.target))
                .collect()
        };
        for (task_id, target) in in_flight {
            self.task_ctx
                .cancel_task(&self.source, &target, task_id, reason.clone())
                .await;
        }
        self.context.lock().unwrap().rollback();
    }
}
