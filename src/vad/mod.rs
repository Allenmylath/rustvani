//! Voice Activity Detection for rustvani / pipecat-rs.
//!
//! Uses the Silero VAD ONNX model via tract-onnx.
//! Model is embedded at compile time — zero file I/O, single binary.
//!
//! # Quick start
//!
//! ```text
//! let vad = VadProcessor::new(16_000, VadParams::default())?.into_processor();
//!
//! let pipeline = PipelineTask::new(vec![
//!     transport.input(),
//!     vad,               // emits VADUserStartedSpeaking / VADUserStoppedSpeaking
//!     stt,
//!     llm,
//!     tts,
//!     transport.output(),
//! ], PipelineParams::default());
//! ```
//!
//! # Enable / disable via TransportParams
//!
//! Set `TransportParams::vad_enabled = true` to have `BaseTransport`
//! insert `VadProcessor` automatically.

pub mod params;
pub mod processor;
pub mod silero;
pub mod state;

pub use params::{VadParams, VAD_CONFIDENCE, VAD_MIN_VOLUME, VAD_START_SECS, VAD_STOP_SECS};
pub use processor::VadProcessor;
pub use silero::SileroVad;
pub use state::{StateMachine, VadState, calculate_audio_volume, exp_smoothing};