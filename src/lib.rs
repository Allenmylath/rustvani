// ---------------------------------------------------------------------------
// Core modules (always available)
// ---------------------------------------------------------------------------
pub mod agents;
pub mod audio_capture;
pub mod billing;
pub mod clock;
pub mod context;
pub mod error;
pub mod frames;
pub mod metrics;
pub mod observer;
pub mod pipeline;
pub mod processors;
pub mod utils;
pub mod audio_process;
pub mod adapters;
pub mod turn;

// ---------------------------------------------------------------------------
// Feature-gated modules
// ---------------------------------------------------------------------------

pub mod vad;

#[cfg(feature = "transport-websocket")]
pub mod transport;

/// The exact `axum` this crate was built against.
///
/// [`WebSocketTransport::run_socket`](transport::websocket::WebSocketTransport::run_socket)
/// takes an [`axum::extract::ws::WebSocket`] by value, so a server that obtains
/// that socket from its *own* axum dependency only type-checks while both crates
/// resolve to the same axum. axum is pre-1.0, so 0.7 and 0.8 are separate
/// packages that Cargo will happily link side by side — and the mismatch shows
/// up as a confusing "expected `WebSocket`, found `WebSocket`" error.
///
/// Build the router off this re-export and the versions cannot drift:
///
/// ```ignore
/// use rustvani::axum::{Router, extract::WebSocketUpgrade, routing::get};
/// ```
#[cfg(feature = "transport-websocket")]
pub use axum;

#[cfg(feature = "transport-websocket")]
pub mod ravi;

/// Frame serializers — Twilio and other WebSocket wire-protocol adapters.
#[cfg(feature = "transport-websocket")]
pub mod serializers;

#[cfg(feature = "dhara")]
pub mod dhara;

/// Services — STT, LLM, TTS implementations.
///
/// Individual services are gated by their respective features:
/// `stt-sarvam`, `llm-openai`, `tts-deepgram`, `tts-sarvam`.
pub mod services;
pub mod tools;

// ---------------------------------------------------------------------------
// Core re-exports (always available)
// ---------------------------------------------------------------------------
pub use audio_capture::{AudioCaptureCollector, AudioCaptureProcessor, AudioStorage, LocalAudioStorage, NoopAudioCaptureCollector, RecordedSegment, SessionAudioCapture};
#[cfg(feature = "db-postgres")]
pub use audio_capture::PostgresAudioMetaStorage;
pub use billing::{BillingCollector, BillingEvent, LogBillingStorage, NoopBillingCollector, SessionBilling};
pub use billing::events::{TranscriptEntry, TranscriptRole};
#[cfg(feature = "db-postgres")]
pub use billing::PostgresBillingStorage;
pub use clock::{BaseClock, SystemClock, system_clock};
pub use error::{PipecatError, Result};
pub use frames::{
    AudioRawData, ControlFrame, DataFrame, DataFrameData, ErrorFrameData, Frame, FrameDirection,
    FrameHandler, FrameInner, FrameKind, FrameProcessor, FrameProcessorSetup, KeypadEntry,
    PassthroughHandler, ProcessorBusHandle, StartFrameData, SystemFrame, TranscriptionData,
    FunctionCallData, FunctionCallResultData,
};
pub use context::{shared_context, LLMContext, ToolCall};
pub use pipeline::{FinishReason, Pipeline, PipelineLifecycle, PipelineParams, PipelineTask};
pub use processors::llm_user_aggregator::LLMUserAggregator;
pub use processors::llm_assistant_aggregator::LLMAssistantAggregator;

// ---------------------------------------------------------------------------
// Feature-gated re-exports
// ---------------------------------------------------------------------------

// VAD (pure-Rust parts always available; ONNX backend gated behind vad-silero-ort)
pub use vad::{SileroVadNative, VadAnalyzer, VadBackend, VadParams, VadProcessor, VadState};
#[cfg(feature = "vad-silero-ort")]
pub use vad::SileroVadOrt;

// Transport
#[cfg(feature = "transport-websocket")]
pub use transport::{BaseInputTransport, BaseOutputTransport, BaseTransport, TransportParams};

// Serializers
#[cfg(feature = "transport-websocket")]
pub use serializers::{
    FrameSerializer, SerializedInput, SerializedOutput, TwilioFrameSerializer, TwilioInputParams,
    TwilioStart,
};

// Services — each gated by its feature
#[cfg(feature = "stt-deepgram")]
pub use services::{DeepgramSttConfig, DeepgramSttHandler};

#[cfg(feature = "stt-sarvam")]
pub use services::{NoiseBackend, SarvamSttConfig, SarvamSttHandler};

#[cfg(feature = "llm-openai")]
pub use services::{OpenAILLMConfig, OpenAILLMHandler, FunctionRegistry};

#[cfg(feature = "stt-sarvam")]
pub use services::{SarvamLLMConfig, SarvamLLMHandler};

#[cfg(feature = "tts-deepgram")]
pub use services::{DeepgramTtsConfig, DeepgramTtsHandler};

#[cfg(feature = "tts-sarvam")]
pub use services::{SarvamTtsConfig, SarvamTtsHandler};

#[cfg(feature = "tts-piper")]
pub use services::{PiperModel, PiperQuality, PiperTtsConfig, PiperTtsHandler};
