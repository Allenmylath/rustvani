use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering as AOrdering};
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};
use futures::future::BoxFuture;

use crate::direction::FrameDirection;
use crate::frames::Frame;

// ---------------------------------------------------------------------------
// Callback types
// ---------------------------------------------------------------------------

/// Pre-bound callback stored inside the queue.
/// Arguments are captured in the closure at the call site of `queue_frame`.
pub type QueueCallback = Box<dyn FnOnce() -> BoxFuture<'static, ()> + Send>;

/// Raw item stored in both queues.
pub type QueueItem = (Frame, FrameDirection, Option<QueueCallback>);

// ---------------------------------------------------------------------------
// FrameProcessorQueue (priority queue — system frames first)
// ---------------------------------------------------------------------------

const HIGH_PRIORITY: u8 = 1; // SystemFrame
const LOW_PRIORITY: u8 = 2;  // everything else

struct PriorityEntry {
    priority: u8,
    counter: u64,
    item: QueueItem,
}

// Manual Ord/PartialOrd: lower priority value + lower counter = first out.
// BinaryHeap is a max-heap, so we invert the comparison.
impl PartialEq for PriorityEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.counter == other.counter
    }
}
impl Eq for PriorityEntry {}

impl PartialOrd for PriorityEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse: we want smaller (priority, counter) to come out first.
        other
            .priority
            .cmp(&self.priority)
            .then_with(|| other.counter.cmp(&self.counter))
    }
}

pub struct FrameProcessorQueue {
    heap: Mutex<BinaryHeap<PriorityEntry>>,
    notify: Arc<Notify>,
    high_counter: AtomicU64,
    low_counter: AtomicU64,
}

impl FrameProcessorQueue {
    pub fn new() -> Self {
        Self {
            heap: Mutex::new(BinaryHeap::new()),
            notify: Arc::new(Notify::new()),
            high_counter: AtomicU64::new(0),
            low_counter: AtomicU64::new(0),
        }
    }

    /// Enqueue a frame. System frames get `HIGH_PRIORITY`.
    pub async fn put(&self, item: QueueItem) {
        let is_system = item.0.is_system();
        let (priority, counter) = if is_system {
            (
                HIGH_PRIORITY,
                self.high_counter.fetch_add(1, AOrdering::SeqCst),
            )
        } else {
            (
                LOW_PRIORITY,
                self.low_counter.fetch_add(1, AOrdering::SeqCst),
            )
        };
        self.heap
            .lock()
            .await
            .push(PriorityEntry { priority, counter, item });
        self.notify.notify_one();
    }

    /// Dequeue the highest-priority item, waiting if the queue is empty.
    pub async fn get(&self) -> QueueItem {
        loop {
            {
                let mut heap = self.heap.lock().await;
                if let Some(entry) = heap.pop() {
                    return entry.item;
                }
            }
            self.notify.notified().await;
        }
    }

    /// Drain non-system, non-uninterruptible frames from the priority queue.
    /// System frames are always kept. Uninterruptible frames are always kept.
    /// Called on interruption so that frames mid-transfer (still in input_queue
    /// before the input task moves them to process_queue) are also discarded.
    pub async fn drain_keep_uninterruptible(&self) {
        let mut heap = self.heap.lock().await;
        let mut kept = BinaryHeap::new();
        for entry in heap.drain() {
            let keep = entry.item.0.is_system() || entry.item.0.is_uninterruptible();
            if keep {
                kept.push(entry);
            }
            // dropped items (and their callbacks) are discarded
        }
        *heap = kept;
    }

    pub async fn len(&self) -> usize {
        self.heap.lock().await.len()
    }
}

// ---------------------------------------------------------------------------
// ProcessQueue (FIFO — supports draining while keeping UninterruptibleFrames)
// ---------------------------------------------------------------------------

pub struct ProcessQueue {
    items: Mutex<VecDeque<QueueItem>>,
    notify: Arc<Notify>,
}

impl ProcessQueue {
    pub fn new() -> Self {
        Self {
            items: Mutex::new(VecDeque::new()),
            notify: Arc::new(Notify::new()),
        }
    }

    pub async fn put(&self, item: QueueItem) {
        self.items.lock().await.push_back(item);
        self.notify.notify_one();
    }

    /// Dequeue, waiting if empty.
    pub async fn get(&self) -> QueueItem {
        loop {
            {
                let mut q = self.items.lock().await;
                if let Some(item) = q.pop_front() {
                    return item;
                }
            }
            self.notify.notified().await;
        }
    }

    /// Drain the queue, keeping only `Uninterruptible` frames.
    /// Called on interruption to discard in-flight non-critical work.
    pub async fn drain_keep_uninterruptible(&self) {
        let mut q = self.items.lock().await;
        let mut kept = VecDeque::new();
        while let Some(item) = q.pop_front() {
            if item.0.is_uninterruptible() {
                kept.push_back(item);
            }
        }
        *q = kept;
    }

    pub async fn is_empty(&self) -> bool {
        self.items.lock().await.is_empty()
    }
}
