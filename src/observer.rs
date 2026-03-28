use async_trait::async_trait;
use crate::direction::FrameDirection;
use crate::frames::Frame;

// ---------------------------------------------------------------------------
// Observer event data
// ---------------------------------------------------------------------------

/// Emitted whenever a processor begins processing a frame.
#[derive(Debug, Clone)]
pub struct FrameProcessed {
    pub processor_name: String,
    pub frame: Frame,
    pub direction: FrameDirection,
    pub timestamp: f64,
}

/// Emitted whenever a processor pushes a frame to an adjacent processor.
#[derive(Debug, Clone)]
pub struct FramePushed {
    pub source_name: String,
    pub destination_name: String,
    pub frame: Frame,
    pub direction: FrameDirection,
    pub timestamp: f64,
}

// ---------------------------------------------------------------------------
// BaseObserver trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait BaseObserver: Send + Sync {
    async fn on_process_frame(&self, data: FrameProcessed);
    async fn on_push_frame(&self, data: FramePushed);
}

// ---------------------------------------------------------------------------
// NoopObserver — does nothing (default when no observer is provided)
// ---------------------------------------------------------------------------

pub struct NoopObserver;

#[async_trait]
impl BaseObserver for NoopObserver {
    async fn on_process_frame(&self, _data: FrameProcessed) {}
    async fn on_push_frame(&self, _data: FramePushed) {}
}
