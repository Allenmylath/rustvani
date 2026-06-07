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

#[cfg(feature = "vad-silero")]
pub mod vad;

#[cfg(feature = "transport-websocket")]
pub mod transport;

#[cfg(feature = "transport-websocket")]
pub mod ravi;

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
pub use audio_capture::{AudioCaptureCollector, AudioCaptureProcessor, AudioStorage, LocalAudioStorage, NoopAudioCaptureCollector, SessionAudioCapture};
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
    FrameHandler, FrameInner, FrameKind, FrameProcessor, FrameProcessorSetup, PassthroughHandler,
    StartFrameData, SystemFrame, TranscriptionData, FunctionCallData, FunctionCallResultData,
};
pub use context::{shared_context, LLMContext, ToolCall};
pub use pipeline::{FinishReason, Pipeline, PipelineLifecycle, PipelineParams, PipelineTask};
pub use processors::llm_user_aggregator::LLMUserAggregator;
pub use processors::llm_assistant_aggregator::LLMAssistantAggregator;

// ---------------------------------------------------------------------------
// Feature-gated re-exports
// ---------------------------------------------------------------------------

// VAD
#[cfg(feature = "vad-silero")]
pub use vad::{SileroVadNative, SileroVadOrt, VadAnalyzer, VadBackend, VadParams, VadProcessor, VadState};

// Transport
#[cfg(feature = "transport-websocket")]
pub use transport::{BaseInputTransport, BaseOutputTransport, BaseTransport, TransportParams};

// Services — each gated by its feature
#[cfg(feature = "stt-deepgram")]
pub use services::{DeepgramSttConfig, DeepgramSttHandler};

#[cfg(feature = "stt-sarvam")]
pub use services::{SarvamSttConfig, SarvamSttHandler};

#[cfg(feature = "llm-openai")]
pub use services::{OpenAILLMConfig, OpenAILLMHandler, FunctionRegistry};

#[cfg(feature = "stt-sarvam")]
pub use services::{SarvamLLMConfig, SarvamLLMHandler};

#[cfg(feature = "tts-deepgram")]
pub use services::{DeepgramTtsConfig, DeepgramTtsHandler};

#[cfg(feature = "tts-sarvam")]
pub use services::{SarvamTtsConfig, SarvamTtsHandler};

pub use services::{PiperModel, PiperQuality, PiperTtsConfig, PiperTtsHandler};
