pub mod clock;
pub mod direction;
pub mod error;
pub mod frames;
pub mod metrics;
pub mod observer;
pub mod processor;
pub mod queue;

// Convenience re-exports so users can write `use pipecat_core::*`
pub use clock::{BaseClock, SystemClock};
pub use direction::FrameDirection;
pub use error::{PipecatError, Result};
pub use frames::{
    DataFrameData, ErrorFrameData, Frame, PauseResumeFrameData, StartFrameData,
};
pub use metrics::{FrameProcessorMetrics, LLMTokenUsage};
pub use observer::{BaseObserver, FrameProcessed, FramePushed, NoopObserver};
pub use processor::{
    FrameCallback, FrameHandler, FrameProcessor, FrameProcessorSetup, PassthroughHandler,
};
pub use queue::{FrameProcessorQueue, ProcessQueue};
