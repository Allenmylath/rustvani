//! Pipeline module: chain assembly and lifecycle management.

mod pipeline;
mod task;

pub use pipeline::Pipeline;
pub use task::{FinishReason, PipelineLifecycle, PipelineParams, PipelineTask};
