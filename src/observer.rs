use async_trait::async_trait;
use crate::direction::FrameDirection;
use crate::frames::Frame;

pub struct FrameProcessed {
    pub processor_name: String,
    pub frame: Frame,
    pub direction: FrameDirection,
    pub timestamp: f64,
}

pub struct FramePushed {
    pub source_name: String,
    pub destination_name: String,
    pub frame: Frame,
    pub direction: FrameDirection,
    pub timestamp: f64,
}

#[async_trait]
pub trait BaseObserver: Send + Sync {
    async fn on_process_frame(&self, event: FrameProcessed);
    async fn on_push_frame(&self, event: FramePushed);
}

pub struct NoopObserver;

#[async_trait]
impl BaseObserver for NoopObserver {
    async fn on_process_frame(&self, _event: FrameProcessed) {}
    async fn on_push_frame(&self, _event: FramePushed) {}
}
