use std::collections::VecDeque;
use std::sync::Arc;
use futures::future::BoxFuture;
use tokio::sync::Mutex;
use tokio::sync::Notify;

use crate::direction::FrameDirection;
use crate::frames::Frame;

pub type QueueCallback = Box<dyn FnOnce() -> BoxFuture<'static, ()> + Send>;
pub type QueueItem = (Frame, FrameDirection, Option<QueueCallback>);

/// Priority queue: system frames go to the front, others to the back.
pub struct FrameProcessorQueue {
    system:     Arc<Mutex<VecDeque<QueueItem>>>,
    non_system: Arc<Mutex<VecDeque<QueueItem>>>,
    notify:     Arc<Notify>,
}

impl FrameProcessorQueue {
    pub fn new() -> Self {
        Self {
            system:     Arc::new(Mutex::new(VecDeque::new())),
            non_system: Arc::new(Mutex::new(VecDeque::new())),
            notify:     Arc::new(Notify::new()),
        }
    }

    pub async fn put(&self, item: QueueItem) {
        if item.0.is_system() {
            self.system.lock().await.push_back(item);
        } else {
            self.non_system.lock().await.push_back(item);
        }
        self.notify.notify_one();
    }

    /// Get next item — system frames take priority.
    pub async fn get(&self) -> QueueItem {
        loop {
            {
                if let Some(item) = self.system.lock().await.pop_front() {
                    return item;
                }
                if let Some(item) = self.non_system.lock().await.pop_front() {
                    return item;
                }
            }
            self.notify.notified().await;
        }
    }

    /// Drain all non-uninterruptible frames from the non-system queue.
    pub async fn drain_keep_uninterruptible(&self) {
        let mut q = self.non_system.lock().await;
        q.retain(|(frame, _, _)| frame.is_uninterruptible());
    }
}

/// Simple FIFO queue for the process task.
pub struct ProcessQueue {
    inner:  Arc<Mutex<VecDeque<QueueItem>>>,
    notify: Arc<Notify>,
}

impl ProcessQueue {
    pub fn new() -> Self {
        Self {
            inner:  Arc::new(Mutex::new(VecDeque::new())),
            notify: Arc::new(Notify::new()),
        }
    }

    pub async fn put(&self, item: QueueItem) {
        self.inner.lock().await.push_back(item);
        self.notify.notify_one();
    }

    pub async fn get(&self) -> QueueItem {
        loop {
            if let Some(item) = self.inner.lock().await.pop_front() {
                return item;
            }
            self.notify.notified().await;
        }
    }

    pub async fn drain_keep_uninterruptible(&self) {
        let mut q = self.inner.lock().await;
        q.retain(|(frame, _, _)| frame.is_uninterruptible());
    }
}
