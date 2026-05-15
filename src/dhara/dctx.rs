//! DharaContext — runtime context provided to every function handler.
//!
//! Bundles pipeline access, flow state, and metadata so handlers
//! don't need to manually thread `Arc`s and push senders.

use std::any::Any;
use std::sync::{Arc, Mutex};


use crate::frames::{Frame, FrameDirection};
use crate::ravi::models as ravi_models;

// ---------------------------------------------------------------------------
// Push sender type (matches PipelineTask::push_sender)
// ---------------------------------------------------------------------------

pub type PushSender = tokio::sync::mpsc::Sender<(Frame, FrameDirection)>;

// ---------------------------------------------------------------------------
// DharaContext
// ---------------------------------------------------------------------------

/// Runtime context provided to every Dhara function handler.
///
/// Handlers receive a shared reference to this. It provides:
/// - Pipeline frame pushing (downstream, upstream)
/// - Ravi server message convenience
/// - Access to flow-specific state via `Any` downcasting
/// - Current node name for logging
///
/// # Example
///
/// ```rust,ignore
/// async fn handle_end_conversation(args: String, ctx: &DharaContext) -> String {
///     let state = ctx.state::<InterviewState>().unwrap();
///     let mut s = state.lock().unwrap();
///     s.completed = true;
///
///     ctx.push_ravi_message(json!({ "type": "interview_done" })).await;
///
///     json!({ "status": "ok" }).to_string()
/// }
/// ```
#[derive(Clone)]
pub struct DharaContext {
    /// Pipeline push sender — initialized after pipeline construction.
    push: Arc<std::sync::OnceLock<PushSender>>,

    /// Flow-specific state. Downcast to the concrete type in handlers.
    flow_state: Arc<dyn Any + Send + Sync>,

    /// Current node name (updated on transitions).
    current_node: Arc<Mutex<String>>,

    /// Connection ID for logging.
    connection_id: u64,
}

impl DharaContext {
    /// Create a new DharaContext.
    ///
    /// `flow_state` is the application-specific state (e.g. `Arc<Mutex<InterviewState>>`).
    /// Call `set_push_sender` after `PipelineTask::new()` to wire pipeline access.
    pub fn new(flow_state: Arc<dyn Any + Send + Sync>, connection_id: u64) -> Self {
        Self {
            push: Arc::new(std::sync::OnceLock::new()),
            flow_state,
            current_node: Arc::new(Mutex::new(String::new())),
            connection_id,
        }
    }

    /// Wire the pipeline push sender. Call once after `PipelineTask::new()`.
    pub fn set_push_sender(&self, sender: PushSender) {
        let _ = self.push.set(sender);
    }

    /// Get the deferred push Arc (for DharaManager internals).
    pub(crate) fn push_arc(&self) -> &Arc<std::sync::OnceLock<PushSender>> {
        &self.push
    }

    // -----------------------------------------------------------------------
    // State access
    // -----------------------------------------------------------------------

    /// Downcast the flow state to a concrete type.
    ///
    /// Returns `None` if the type doesn't match.
    ///
    /// ```rust,ignore
    /// let interview = ctx.state::<Mutex<InterviewState>>().unwrap();
    /// let mut s = interview.lock().unwrap();
    /// ```
    pub fn state<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.flow_state.downcast_ref::<T>()
    }

    /// Get the raw flow state Arc (for cloning into async blocks).
    pub fn state_arc(&self) -> &Arc<dyn Any + Send + Sync> {
        &self.flow_state
    }

    // -----------------------------------------------------------------------
    // Node info
    // -----------------------------------------------------------------------

    /// Current node name.
    pub fn current_node(&self) -> String {
        self.current_node.lock().unwrap().clone()
    }

    /// Update current node (called by DharaManager on transitions).
    pub(crate) fn set_current_node(&self, name: &str) {
        *self.current_node.lock().unwrap() = name.to_string();
    }

    /// Connection ID for logging.
    pub fn connection_id(&self) -> u64 {
        self.connection_id
    }

    // -----------------------------------------------------------------------
    // Pipeline frame pushing
    // -----------------------------------------------------------------------

    /// Push a frame into the pipeline.
    pub async fn push_frame(&self, frame: Frame, direction: FrameDirection) -> bool {
        if let Some(tx) = self.push.get() {
            tx.send((frame, direction)).await.is_ok()
        } else {
            log::warn!("[conn={}] DharaContext: push sender not yet initialized", self.connection_id);
            false
        }
    }

    /// Push a frame downstream (towards TTS/output).
    pub async fn push_downstream(&self, frame: Frame) -> bool {
        self.push_frame(frame, FrameDirection::Downstream).await
    }

    /// Push a frame upstream (towards input).
    pub async fn push_upstream(&self, frame: Frame) -> bool {
        self.push_frame(frame, FrameDirection::Upstream).await
    }

    /// Convenience: push a RAVI server message downstream.
    ///
    /// Wraps the data in the RAVI protocol envelope and sends it.
    pub async fn push_ravi_message(&self, data: serde_json::Value) -> bool {
        let payload = ravi_models::msg_server_message(data);
        let frame = Frame::ravi_server_message(payload);
        self.push_downstream(frame).await
    }
}
