use thiserror::Error;

#[derive(Debug, Error)]
pub enum PipecatError {
    #[error("Pipeline error: {0}")]
    Pipeline(String),
    #[error("IO error: {0}")]
    Io(String),
}

impl PipecatError {
    pub fn pipeline(msg: impl Into<String>) -> Self {
        Self::Pipeline(msg.into())
    }
}

pub type Result<T> = std::result::Result<T, PipecatError>;
