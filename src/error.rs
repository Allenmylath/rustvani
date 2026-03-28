use thiserror::Error;

#[derive(Debug, Error)]
pub enum PipecatError {
    #[error("Frame processing error: {0}")]
    ProcessingError(String),

    #[error("Not initialized: {0}")]
    NotInitialized(String),

    #[error("Task error: {0}")]
    TaskError(String),

    #[error("Processor not started — StartFrame not yet received")]
    NotStarted,
}

pub type Result<T> = std::result::Result<T, PipecatError>;
