pub mod collector;
pub mod encoder;
pub mod processor;
pub mod segment;
pub mod storage;

pub use collector::{AudioCaptureCollector, NoopAudioCaptureCollector, SessionAudioCapture};
pub use processor::AudioCaptureProcessor;
pub use segment::{AudioSegmentMeta, PendingAudioSegment};
pub use storage::local::LocalAudioStorage;
pub use storage::{AudioStorage, RecordedSegment};

#[cfg(feature = "db-postgres")]
pub use storage::postgres::PostgresAudioMetaStorage;
