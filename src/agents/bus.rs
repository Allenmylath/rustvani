//! Agent message bus — two-priority, drop-aware, lock-free fan-out.
//!
//! ## Design
//!
//! Every subscriber owns **two channels**:
//!
//! - an **unbounded system channel** for control messages (`End`, `Cancel`,
//!   `Activate`, `TaskCancel`, urgent task replies, registry traffic). These
//!   never block the sender and are never dropped.
//! - a **bounded data channel** (default capacity 256) for everything else —
//!   primarily bridged pipeline frames. When the channel is full the message
//!   is dropped for that subscriber and counted.
//!
//! A single dispatch task per subscriber drains both channels with a
//! `biased` `tokio::select!`, so system messages are always polled before
//! data messages. This replaces a hand-rolled two-queue + `Notify` scheme.
//!
//! ## Drop policy (rationale: realtime voice)
//!
//! rustvani is a realtime voice framework. A slow background agent must
//! never stall the producer — so data sends use `try_send` and **drop on
//! full** (control never drops, data never blocks). Frames are droppable;
//! control messages are not, and they travel the unbounded system channel.
//! Drops are counted per subscriber and logged (first drop, then every
//! 100th).
//!
//! ## Fan-out
//!
//! Messages fan out as `Arc<BusMessage>` — one allocation per send, no
//! deep clones. The subscriber list is an [`arc_swap::ArcSwap`] snapshot,
//! so `send()` takes no lock at all.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwap;
use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::error::{PipecatError, Result};
use crate::frames::{Frame, FrameDirection};

/// How long `unsubscribe()` / `stop()` wait for a dispatch task to finish
/// its in-flight handler before aborting it.
const DISPATCH_JOIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

/// Default capacity of each subscriber's bounded data channel.
pub const DEFAULT_DATA_CAPACITY: usize = 256;

// ---------------------------------------------------------------------------
// TaskStatus
// ---------------------------------------------------------------------------

/// Terminal and intermediate states of a dispatched task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    /// Task has been dispatched but not yet picked up.
    Pending,
    /// Task handler is executing.
    Running,
    /// Task finished successfully.
    Completed,
    /// Task finished with an error.
    Failed,
    /// Task was cancelled before completing.
    Cancelled,
}

// ---------------------------------------------------------------------------
// BusPayload
// ---------------------------------------------------------------------------

/// The content of a [`BusMessage`].
#[derive(Debug, Clone)]
pub enum BusPayload {
    // --- Frame transport ---
    /// A pipeline frame bridged between agents.
    Frame {
        /// The frame being transported.
        frame: Frame,
        /// Direction the frame should be injected with at the receiver.
        direction: FrameDirection,
    },

    // --- Agent lifecycle ---
    /// Make the target agent active (it starts publishing bridged frames).
    Activate {
        /// Optional activation arguments.
        args: Option<Value>,
    },
    /// Make the target agent inactive.
    Deactivate,
    /// Gracefully end the target agent.
    End {
        /// Optional human-readable reason.
        reason: Option<String>,
    },
    /// Hard-cancel the target agent.
    Cancel {
        /// Optional human-readable reason.
        reason: Option<String>,
    },

    // --- Registry ---
    /// Announcement that an agent finished setup and is running.
    AgentReady {
        /// Name of the runner hosting the agent.
        runner: String,
        /// Parent agent name, if any.
        parent: Option<String>,
        /// Whether the agent is currently active.
        active: bool,
        /// Whether the agent accepts bridged frames.
        bridged: bool,
        /// Monotonic timestamp when the agent started.
        started_at: Option<f64>,
    },
    /// A runner broadcasting its full local registry (for remote discovery).
    AgentRegistry {
        /// Name of the broadcasting runner.
        runner: String,
        /// All agents local to that runner.
        agents: Vec<AgentRegistryEntry>,
    },
    /// An agent reporting a fatal error.
    AgentError {
        /// Error description.
        error: String,
    },

    // --- Task coordination ---
    /// Request the target agent to run a job.
    TaskRequest {
        /// Unique task id (UUID).
        task_id: String,
        /// Named handler to invoke; `None` selects the default handler.
        task_name: Option<String>,
        /// Arbitrary JSON payload.
        payload: Option<Value>,
    },
    /// Terminal reply for a task.
    TaskResponse {
        /// Id of the task being answered.
        task_id: String,
        /// Terminal status.
        status: TaskStatus,
        /// Arbitrary JSON result.
        response: Option<Value>,
    },
    /// Terminal reply delivered with system priority.
    TaskResponseUrgent {
        /// Id of the task being answered.
        task_id: String,
        /// Terminal status.
        status: TaskStatus,
        /// Arbitrary JSON result.
        response: Option<Value>,
    },
    /// Non-terminal progress update for a task.
    TaskUpdate {
        /// Id of the task being updated.
        task_id: String,
        /// Arbitrary JSON update.
        update: Option<Value>,
    },
    /// Non-terminal progress update delivered with system priority.
    TaskUpdateUrgent {
        /// Id of the task being updated.
        task_id: String,
        /// Arbitrary JSON update.
        update: Option<Value>,
    },
    /// Ask the executing agent to emit a progress update.
    TaskUpdateRequest {
        /// Id of the task.
        task_id: String,
    },
    /// Ask the executing agent to cancel a running task.
    TaskCancel {
        /// Id of the task to cancel.
        task_id: String,
        /// Optional human-readable reason.
        reason: Option<String>,
    },

    // --- Task streaming ---
    /// First chunk of a streaming task reply.
    TaskStreamStart {
        /// Id of the task.
        task_id: String,
        /// Arbitrary JSON chunk.
        data: Option<Value>,
    },
    /// Intermediate chunk of a streaming task reply.
    TaskStreamData {
        /// Id of the task.
        task_id: String,
        /// Arbitrary JSON chunk.
        data: Option<Value>,
    },
    /// Last chunk of a streaming task reply.
    TaskStreamEnd {
        /// Id of the task.
        task_id: String,
        /// Arbitrary JSON chunk.
        data: Option<Value>,
    },
}

// ---------------------------------------------------------------------------
// BusMessage
// ---------------------------------------------------------------------------

/// A message travelling on the agent bus.
#[derive(Debug, Clone)]
pub struct BusMessage {
    /// Name of the sending agent/runner. Never delivered back to the source.
    pub source: String,
    /// `Some(name)` delivers only to that subscriber; `None` broadcasts.
    pub target: Option<String>,
    /// Message content.
    pub payload: BusPayload,
    /// Bus-stamped sequence number for total-order debugging across agents.
    ///
    /// Stamped by the bus in `send()` before fan-out, so every subscriber
    /// sees the same value for the same message. The value passed in by the
    /// caller is ignored.
    pub seq: u64,
}

impl BusMessage {
    /// Create a message. `seq` is initialised to 0 and stamped by the bus.
    pub fn new(source: impl Into<String>, target: Option<String>, payload: BusPayload) -> Self {
        Self {
            source: source.into(),
            target,
            payload,
            seq: 0,
        }
    }

    /// Whether this message travels the unbounded system (priority) channel.
    ///
    /// System messages are control traffic: they are delivered before any
    /// queued data messages and are never dropped.
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

/// One agent's metadata inside an [`BusPayload::AgentRegistry`] broadcast.
#[derive(Debug, Clone)]
pub struct AgentRegistryEntry {
    /// Agent name.
    pub name: String,
    /// Parent agent name, if any.
    pub parent: Option<String>,
    /// Whether the agent is currently active.
    pub active: bool,
    /// Whether the agent accepts bridged frames.
    pub bridged: bool,
    /// Monotonic timestamp when the agent started.
    pub started_at: Option<f64>,
}

// ---------------------------------------------------------------------------
// BusSubscriber
// ---------------------------------------------------------------------------

/// A bus endpoint. Each subscriber gets its own dispatch task; messages are
/// delivered one at a time (system priority first).
#[async_trait]
pub trait BusSubscriber: Send + Sync {
    /// Unique name on the bus. Duplicate names are rejected at subscribe.
    fn name(&self) -> &str;
    /// Handle one message. Called sequentially per subscriber.
    async fn on_bus_message(&self, message: Arc<BusMessage>);
}

// ---------------------------------------------------------------------------
// AgentBus trait
// ---------------------------------------------------------------------------

/// Message transport between agents. [`LocalAgentBus`] is the in-process
/// implementation; the trait is ready for network implementations.
#[async_trait]
pub trait AgentBus: Send + Sync {
    /// Register a subscriber and start its dispatch task.
    ///
    /// Returns an error if a subscriber with the same name already exists.
    async fn subscribe(&self, subscriber: Arc<dyn BusSubscriber>) -> Result<()>;
    /// Remove a subscriber. Its in-flight handler is allowed to finish
    /// (bounded wait), then the dispatch task is stopped.
    async fn unsubscribe(&self, name: &str);
    /// Fan a message out to all matching subscribers.
    ///
    /// Filtering: the source never receives its own message; if `target` is
    /// `Some`, only the named subscriber receives it.
    async fn send(&self, message: BusMessage);
    /// Start the bus. The local bus is usable immediately after
    /// construction; this exists for network buses that need explicit start.
    async fn start(&self);
    /// Stop the bus: all dispatch tasks are stopped (in-flight handlers may
    /// finish, pending system messages are drained) and further sends become
    /// no-ops.
    async fn stop(&self);
}

// ---------------------------------------------------------------------------
// SubscriberHandle
// ---------------------------------------------------------------------------

/// Per-subscriber channel endpoints and dispatch-task bookkeeping.
struct SubscriberHandle {
    name: Arc<str>,
    /// System priority — never blocks, never drops.
    sys_tx: mpsc::UnboundedSender<Arc<BusMessage>>,
    /// Bounded data channel — `try_send`, drop on full.
    data_tx: mpsc::Sender<Arc<BusMessage>>,
    /// Count of data messages dropped for this subscriber.
    dropped: Arc<AtomicU64>,
    /// Cancels the dispatch loop.
    cancel: CancellationToken,
    /// Dispatch task handle. std Mutex is fine: never held across .await.
    join: std::sync::Mutex<Option<JoinHandle<()>>>,
}

impl SubscriberHandle {
    /// Take the join handle out (sync, lock dropped immediately) and await
    /// it with a timeout; abort on timeout. Never aborts first, so an
    /// in-flight `on_bus_message` is allowed to finish.
    async fn shutdown(&self) {
        self.cancel.cancel();
        let handle = self.join.lock().unwrap().take();
        if let Some(h) = handle {
            let abort = h.abort_handle();
            if tokio::time::timeout(DISPATCH_JOIN_TIMEOUT, h)
                .await
                .is_err()
            {
                log::warn!(
                    "bus: dispatch task for '{}' did not stop within {:?}, aborting",
                    self.name,
                    DISPATCH_JOIN_TIMEOUT
                );
                abort.abort();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch loop
// ---------------------------------------------------------------------------

/// One task per subscriber. `biased` gives the priority semantics: the
/// system channel is always polled before the data channel. On cancel,
/// remaining *system* messages are drained and delivered (End/Cancel must
/// not be lost); pending data messages are discarded.
async fn dispatch_loop(
    subscriber: Arc<dyn BusSubscriber>,
    mut sys_rx: mpsc::UnboundedReceiver<Arc<BusMessage>>,
    mut data_rx: mpsc::Receiver<Arc<BusMessage>>,
    cancel: CancellationToken,
) {
    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            Some(msg) = sys_rx.recv() => subscriber.on_bus_message(msg).await,
            Some(msg) = data_rx.recv() => subscriber.on_bus_message(msg).await,
            else => break,
        }
    }
    // Deliver any system messages still queued — End/Cancel must not be lost.
    while let Ok(msg) = sys_rx.try_recv() {
        subscriber.on_bus_message(msg).await;
    }
    log::debug!("bus: dispatch loop for '{}' exited", subscriber.name());
}

// ---------------------------------------------------------------------------
// LocalAgentBus
// ---------------------------------------------------------------------------

/// In-process [`AgentBus`] implementation.
///
/// See the module docs for the two-channel priority design and drop policy.
pub struct LocalAgentBus {
    /// Lock-free snapshot of subscriber handles. `send()` only loads.
    subscribers: ArcSwap<Vec<Arc<SubscriberHandle>>>,
    /// Serialises RCU writers (subscribe/unsubscribe/stop). Never held
    /// across an .await.
    write_lock: std::sync::Mutex<()>,
    /// Sequence counter stamped onto every message in `send()`.
    seq: AtomicU64,
    /// Capacity of each subscriber's bounded data channel.
    data_capacity: usize,
    /// Cleared by `stop()`; `send()` is a no-op once cleared.
    running: AtomicBool,
}

impl LocalAgentBus {
    /// Create a bus with the default data-channel capacity
    /// ([`DEFAULT_DATA_CAPACITY`]). The bus is usable immediately.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_DATA_CAPACITY)
    }

    /// Create a bus with a custom per-subscriber data-channel capacity.
    pub fn with_capacity(data_capacity: usize) -> Self {
        Self {
            subscribers: ArcSwap::from_pointee(Vec::new()),
            write_lock: std::sync::Mutex::new(()),
            seq: AtomicU64::new(0),
            data_capacity: data_capacity.max(1),
            running: AtomicBool::new(true),
        }
    }

    /// Number of data messages dropped so far for a subscriber, or `None`
    /// if no subscriber with that name exists.
    pub fn dropped_count(&self, name: &str) -> Option<u64> {
        self.subscribers
            .load()
            .iter()
            .find(|s| &*s.name == name)
            .map(|s| s.dropped.load(Ordering::Relaxed))
    }
}

impl Default for LocalAgentBus {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AgentBus for LocalAgentBus {
    async fn subscribe(&self, subscriber: Arc<dyn BusSubscriber>) -> Result<()> {
        let name: Arc<str> = Arc::from(subscriber.name());

        let (sys_tx, sys_rx) = mpsc::unbounded_channel();
        let (data_tx, data_rx) = mpsc::channel(self.data_capacity);
        let cancel = CancellationToken::new();

        let join = tokio::spawn(dispatch_loop(subscriber, sys_rx, data_rx, cancel.clone()));

        let handle = Arc::new(SubscriberHandle {
            name: name.clone(),
            sys_tx,
            data_tx,
            dropped: Arc::new(AtomicU64::new(0)),
            cancel,
            join: std::sync::Mutex::new(Some(join)),
        });

        // RCU insert under the writer lock; reject duplicate names.
        let inserted = {
            let _guard = self.write_lock.lock().unwrap();
            let current = self.subscribers.load_full();
            if current.iter().any(|s| s.name == name) {
                false
            } else {
                let mut next = Vec::with_capacity(current.len() + 1);
                next.extend(current.iter().cloned());
                next.push(handle.clone());
                self.subscribers.store(Arc::new(next));
                true
            }
        };

        if !inserted {
            // Tear down the dispatch task we just spawned.
            handle.shutdown().await;
            return Err(PipecatError::pipeline(format!(
                "Bus subscriber '{}' already exists",
                name
            )));
        }

        log::debug!("bus: subscribed '{}'", name);
        Ok(())
    }

    async fn unsubscribe(&self, name: &str) {
        // RCU remove under the writer lock, then shut down outside it.
        let removed = {
            let _guard = self.write_lock.lock().unwrap();
            let current = self.subscribers.load_full();
            let removed = current.iter().find(|s| &*s.name == name).cloned();
            if removed.is_some() {
                let next: Vec<Arc<SubscriberHandle>> = current
                    .iter()
                    .filter(|s| &*s.name != name)
                    .cloned()
                    .collect();
                self.subscribers.store(Arc::new(next));
            }
            removed
        };

        if let Some(handle) = removed {
            handle.shutdown().await;
            log::debug!("bus: unsubscribed '{}'", name);
        }
    }

    async fn send(&self, message: BusMessage) {
        if !self.running.load(Ordering::Acquire) {
            log::debug!(
                "bus: dropping send from '{}' — bus is stopped",
                message.source
            );
            return;
        }

        let mut message = message;
        // Stamp before fan-out so all subscribers see the same seq.
        message.seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let is_system = message.is_system();
        let msg = Arc::new(message);

        let subs = self.subscribers.load();
        for sub in subs.iter() {
            if msg.source == *sub.name {
                continue;
            }
            if let Some(target) = &msg.target {
                if target.as_str() != &*sub.name {
                    continue;
                }
            }

            if is_system {
                // Unbounded: only fails when the dispatch task is gone.
                if sub.sys_tx.send(msg.clone()).is_err() {
                    log::debug!(
                        "bus: system message to '{}' not delivered (receiver gone)",
                        sub.name
                    );
                }
            } else {
                match sub.data_tx.try_send(msg.clone()) {
                    Ok(()) => {}
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        let n = sub.dropped.fetch_add(1, Ordering::Relaxed) + 1;
                        // Rate-limited: warn on first drop, then every 100th.
                        if n == 1 || n % 100 == 0 {
                            log::warn!(
                                "bus: data channel full for '{}' — {} message(s) dropped so far",
                                sub.name,
                                n
                            );
                        }
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        log::debug!(
                            "bus: data message to '{}' not delivered (receiver gone)",
                            sub.name
                        );
                    }
                }
            }
        }
    }

    async fn start(&self) {
        self.running.store(true, Ordering::Release);
    }

    async fn stop(&self) {
        self.running.store(false, Ordering::Release);

        // Swap in an empty list, then shut every dispatch task down.
        let old = {
            let _guard = self.write_lock.lock().unwrap();
            self.subscribers.swap(Arc::new(Vec::new()))
        };
        for handle in old.iter() {
            handle.shutdown().await;
        }
        log::debug!("bus: stopped ({} subscriber(s) shut down)", old.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;
    use tokio::sync::Mutex as TokioMutex;
    use tokio::sync::Notify;

    /// Test subscriber that records the order of received payload tags and
    /// can be slowed down / blocked via an async gate.
    struct Recorder {
        name: String,
        received: TokioMutex<Vec<Arc<BusMessage>>>,
        received_count: AtomicUsize,
        /// When set, the handler waits on this before returning.
        gate: Option<Arc<Notify>>,
        /// Per-message handler delay.
        delay: Option<Duration>,
        in_flight: AtomicBool,
        notify_on_receive: Arc<Notify>,
    }

    impl Recorder {
        fn new(name: &str) -> Arc<Self> {
            Arc::new(Self {
                name: name.to_string(),
                received: TokioMutex::new(Vec::new()),
                received_count: AtomicUsize::new(0),
                gate: None,
                delay: None,
                in_flight: AtomicBool::new(false),
                notify_on_receive: Arc::new(Notify::new()),
            })
        }

        fn with_gate(name: &str, gate: Arc<Notify>) -> Arc<Self> {
            Arc::new(Self {
                name: name.to_string(),
                received: TokioMutex::new(Vec::new()),
                received_count: AtomicUsize::new(0),
                gate: Some(gate),
                delay: None,
                in_flight: AtomicBool::new(false),
                notify_on_receive: Arc::new(Notify::new()),
            })
        }

        fn with_delay(name: &str, delay: Duration) -> Arc<Self> {
            Arc::new(Self {
                name: name.to_string(),
                received: TokioMutex::new(Vec::new()),
                received_count: AtomicUsize::new(0),
                gate: None,
                delay: Some(delay),
                in_flight: AtomicBool::new(false),
                notify_on_receive: Arc::new(Notify::new()),
            })
        }

        async fn payload_names(&self) -> Vec<&'static str> {
            self.received
                .lock()
                .await
                .iter()
                .map(|m| match &m.payload {
                    BusPayload::Frame { .. } => "frame",
                    BusPayload::Cancel { .. } => "cancel",
                    BusPayload::End { .. } => "end",
                    BusPayload::TaskUpdate { .. } => "update",
                    _ => "other",
                })
                .collect()
        }
    }

    #[async_trait]
    impl BusSubscriber for Recorder {
        fn name(&self) -> &str {
            &self.name
        }

        async fn on_bus_message(&self, message: Arc<BusMessage>) {
            self.in_flight.store(true, Ordering::SeqCst);
            if let Some(gate) = &self.gate {
                gate.notified().await;
            }
            if let Some(d) = self.delay {
                tokio::time::sleep(d).await;
            }
            self.received.lock().await.push(message);
            self.received_count.fetch_add(1, Ordering::SeqCst);
            self.in_flight.store(false, Ordering::SeqCst);
            self.notify_on_receive.notify_waiters();
        }
    }

    fn data_msg(source: &str, target: Option<&str>) -> BusMessage {
        BusMessage::new(
            source,
            target.map(String::from),
            BusPayload::TaskUpdate {
                task_id: "t".into(),
                update: None,
            },
        )
    }

    fn cancel_msg(source: &str, target: Option<&str>) -> BusMessage {
        BusMessage::new(
            source,
            target.map(String::from),
            BusPayload::Cancel { reason: None },
        )
    }

    async fn wait_for<F: Fn() -> bool>(cond: F, timeout: Duration) -> bool {
        let deadline = tokio::time::Instant::now() + timeout;
        while tokio::time::Instant::now() < deadline {
            if cond() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        cond()
    }

    /// Invariant 1: system messages are handled before queued data messages.
    #[tokio::test]
    async fn system_priority_over_data() {
        let bus = LocalAgentBus::new();
        let sub = Recorder::with_delay("slow", Duration::from_millis(10));
        bus.subscribe(sub.clone()).await.unwrap();

        // Enqueue 50 data messages, then one Cancel.
        for _ in 0..50 {
            bus.send(data_msg("src", Some("slow"))).await;
        }
        bus.send(cancel_msg("src", Some("slow"))).await;

        assert!(
            wait_for(
                || sub.received_count.load(Ordering::SeqCst) >= 51,
                Duration::from_secs(10)
            )
            .await,
            "not all messages delivered"
        );

        let names = sub.payload_names().await;
        let cancel_pos = names.iter().position(|n| *n == "cancel").unwrap();
        // The handler is slow (10ms per message); the Cancel must overtake
        // almost all of the 50 queued data messages. Allow the few that were
        // already in flight when the Cancel was sent.
        assert!(
            cancel_pos < 5,
            "Cancel was handled at position {cancel_pos}, expected near the front"
        );
    }

    /// Invariant 2: bounded data channel drops on full and counts drops;
    /// a concurrently sent system message still arrives.
    #[tokio::test]
    async fn drop_accounting() {
        let bus = LocalAgentBus::with_capacity(4);
        let gate = Arc::new(Notify::new());
        let sub = Recorder::with_gate("blocked", gate.clone());
        bus.subscribe(sub.clone()).await.unwrap();

        // First message occupies the handler (waiting on the gate); the
        // next 4 fill the channel. Wait until the handler is in flight so
        // exactly one message has been popped from the channel.
        bus.send(data_msg("src", Some("blocked"))).await;
        assert!(
            wait_for(
                || sub.in_flight.load(Ordering::SeqCst),
                Duration::from_secs(2)
            )
            .await,
            "handler never started"
        );

        // 10 more data messages: 4 fit, 6 dropped.
        for _ in 0..10 {
            bus.send(data_msg("src", Some("blocked"))).await;
        }
        assert_eq!(bus.dropped_count("blocked"), Some(6));

        // A system message still arrives even though data is full.
        bus.send(cancel_msg("src", Some("blocked"))).await;

        // Unblock the handler for every queued message.
        for _ in 0..10 {
            gate.notify_waiters();
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // 1 in-flight + 4 queued data + 1 system = 6 delivered.
        assert!(
            wait_for(
                || sub.received_count.load(Ordering::SeqCst) >= 6,
                Duration::from_secs(5)
            )
            .await,
            "queued messages not delivered, got {}",
            sub.received_count.load(Ordering::SeqCst)
        );
        assert_eq!(sub.received_count.load(Ordering::SeqCst), 6);
        let names = sub.payload_names().await;
        assert!(names.contains(&"cancel"), "system message lost: {names:?}");
    }

    /// Invariant 3: broadcast reaches all except source; targeted message
    /// reaches only the target.
    #[tokio::test]
    async fn targeting_rules() {
        let bus = LocalAgentBus::new();
        let a = Recorder::new("a");
        let b = Recorder::new("b");
        let c = Recorder::new("c");
        bus.subscribe(a.clone()).await.unwrap();
        bus.subscribe(b.clone()).await.unwrap();
        bus.subscribe(c.clone()).await.unwrap();

        // Broadcast from a: b and c receive, a does not.
        bus.send(data_msg("a", None)).await;
        // Targeted from a to b: only b receives.
        bus.send(data_msg("a", Some("b"))).await;

        assert!(
            wait_for(
                || b.received_count.load(Ordering::SeqCst) == 2
                    && c.received_count.load(Ordering::SeqCst) == 1,
                Duration::from_secs(2)
            )
            .await
        );
        assert_eq!(a.received_count.load(Ordering::SeqCst), 0);
        assert_eq!(b.received_count.load(Ordering::SeqCst), 2);
        assert_eq!(c.received_count.load(Ordering::SeqCst), 1);
    }

    /// Invariant 4: concurrent sends get distinct seq values, and delivery
    /// order from a single sender is preserved.
    #[tokio::test]
    async fn seq_monotonicity_and_order() {
        let bus = Arc::new(LocalAgentBus::new());
        let sub = Recorder::new("sink");
        bus.subscribe(sub.clone()).await.unwrap();

        // 4 concurrent sender tasks, 25 messages each.
        let mut handles = Vec::new();
        for i in 0..4 {
            let bus = bus.clone();
            handles.push(tokio::spawn(async move {
                for j in 0..25 {
                    let mut m = data_msg(&format!("sender{i}"), Some("sink"));
                    if let BusPayload::TaskUpdate { task_id, .. } = &mut m.payload {
                        *task_id = format!("{i}-{j}");
                    }
                    bus.send(m).await;
                }
            }));
        }
        for h in handles {
            h.await.unwrap();
        }

        assert!(
            wait_for(
                || sub.received_count.load(Ordering::SeqCst) == 100,
                Duration::from_secs(5)
            )
            .await
        );

        let received = sub.received.lock().await;
        // All seq values distinct.
        let mut seqs: Vec<u64> = received.iter().map(|m| m.seq).collect();
        seqs.sort_unstable();
        seqs.dedup();
        assert_eq!(seqs.len(), 100, "duplicate seq values");

        // Per-sender delivery order respects send order (j increasing).
        for i in 0..4 {
            let js: Vec<usize> = received
                .iter()
                .filter_map(|m| match &m.payload {
                    BusPayload::TaskUpdate { task_id, .. } => {
                        let (s, j) = task_id.split_once('-')?;
                        (s == i.to_string()).then(|| j.parse::<usize>().unwrap())
                    }
                    _ => None,
                })
                .collect();
            assert!(
                js.windows(2).all(|w| w[0] < w[1]),
                "sender {i} delivery out of order: {js:?}"
            );
        }
    }

    /// Invariant 5: unsubscribe while the handler is mid-await — no panic,
    /// the in-flight handler completes.
    #[tokio::test]
    async fn unsubscribe_while_handler_in_flight() {
        let bus = LocalAgentBus::new();
        let gate = Arc::new(Notify::new());
        let sub = Recorder::with_gate("busy", gate.clone());
        bus.subscribe(sub.clone()).await.unwrap();

        bus.send(data_msg("src", Some("busy"))).await;
        assert!(
            wait_for(
                || sub.in_flight.load(Ordering::SeqCst),
                Duration::from_secs(2)
            )
            .await
        );

        // Unsubscribe concurrently while the handler waits on the gate;
        // release the gate shortly after.
        let g = gate.clone();
        let release = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            g.notify_waiters();
        });
        bus.unsubscribe("busy").await;
        release.await.unwrap();

        // The in-flight handler completed (message recorded).
        assert_eq!(sub.received_count.load(Ordering::SeqCst), 1);
        assert!(!sub.in_flight.load(Ordering::SeqCst));
    }

    /// Invariant 6: send after stop is a no-op (no panic, no delivery).
    #[tokio::test]
    async fn send_after_stop_is_noop() {
        let bus = LocalAgentBus::new();
        let sub = Recorder::new("a");
        bus.subscribe(sub.clone()).await.unwrap();
        bus.stop().await;

        bus.send(data_msg("src", Some("a"))).await;
        bus.send(cancel_msg("src", Some("a"))).await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(sub.received_count.load(Ordering::SeqCst), 0);
    }

    /// Duplicate subscriber names are rejected.
    #[tokio::test]
    async fn duplicate_subscribe_rejected() {
        let bus = LocalAgentBus::new();
        bus.subscribe(Recorder::new("dup")).await.unwrap();
        assert!(bus.subscribe(Recorder::new("dup")).await.is_err());
    }
}
