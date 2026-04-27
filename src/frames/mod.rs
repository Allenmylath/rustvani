pub mod direction;
pub mod processor;
pub mod queue;

pub use direction::FrameDirection;
pub use processor::{
    FrameCallback, FrameHandler, FrameProcessor, FrameProcessorSetup, PassthroughHandler,
    WeakFrameProcessor,
};

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Frame ID counter
// ---------------------------------------------------------------------------

static FRAME_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

pub fn next_frame_id() -> u64 {
    FRAME_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Flat FrameKind
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameKind {
    // System — lifecycle
    Start, Cancel, Error, Interruption, Stop,
    EndTask, CancelTask, StopTask, InterruptionTask,
    // System — speaking signals
    BotSpeaking, UserSpeaking,
    BotStartedSpeaking, BotStoppedSpeaking,
    UserStartedSpeaking, UserStoppedSpeaking,
    VADUserStartedSpeaking, VADUserStoppedSpeaking,
    // System — audio/video input
    InputAudioRaw,
    // System — processor control
    PauseProcessor, PauseProcessorUrgent,
    ResumeProcessor, ResumeProcessorUrgent,
    Heartbeat,
    // System — RAVI protocol
    RaviClientMessage,
    RaviServerMessage,
    RaviServerResponse,
    // Control — pipeline lifecycle
    End,
    // Control — LLM response boundaries
    LLMFullResponseStart,
    LLMFullResponseEnd,
    // Data
    Data,
    Transcription,
    LLMText,
    LLMContextFrame,
    // Data — audio output
    OutputAudioRaw,
}

// ---------------------------------------------------------------------------
// Audio payload
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AudioRawData {
    pub audio:            Vec<u8>,
    pub sample_rate:      u32,
    pub num_channels:     u16,
    pub num_frames:       usize,
    pub transport_source: Option<String>,
}

impl AudioRawData {
    pub fn new(audio: Vec<u8>, sample_rate: u32, num_channels: u16) -> Self {
        let num_frames = if num_channels > 0 {
            audio.len() / (num_channels as usize * 2)
        } else {
            0
        };
        Self { audio, sample_rate, num_channels, num_frames, transport_source: None }
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.transport_source = Some(source.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Payloads
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct StartFrameData {
    pub allow_interruptions:       bool,
    pub enable_metrics:            bool,
    pub enable_usage_metrics:      bool,
    pub report_only_initial_ttfb:  bool,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct ErrorFrameData {
    pub error:          String,
    pub fatal:          bool,
    pub processor_name: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct DataFrameData {
    pub content:  Vec<u8>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct TranscriptionData {
    pub text:      String,
    pub user_id:   String,
    pub timestamp: String,
    pub language:  Option<String>,
    pub finalized: bool,
}

impl TranscriptionData {
    pub fn new(text: impl Into<String>, user_id: impl Into<String>, timestamp: impl Into<String>) -> Self {
        Self { text: text.into(), user_id: user_id.into(), timestamp: timestamp.into(), language: None, finalized: false }
    }
    pub fn with_language(mut self, lang: impl Into<String>) -> Self { self.language = Some(lang.into()); self }
    pub fn finalized(mut self) -> Self { self.finalized = true; self }
}

// ---------------------------------------------------------------------------
// SystemFrame
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum SystemFrame {
    // ---- Lifecycle ----
    Start(StartFrameData),
    Cancel    { reason: Option<String> },
    Error(ErrorFrameData),
    Interruption,
    Stop      { reason: Option<String> },

    EndTask   { reason: Option<String> },
    CancelTask { reason: Option<String> },
    StopTask,
    InterruptionTask,

    // ---- Speaking signals ----
    BotSpeaking,
    UserSpeaking,
    BotStartedSpeaking,
    BotStoppedSpeaking,
    UserStartedSpeaking { emulated: bool },
    UserStoppedSpeaking  { emulated: bool },
    VADUserStartedSpeaking { start_secs: f32, timestamp: f64 },
    VADUserStoppedSpeaking { stop_secs: f32, timestamp: f64 },

    // ---- Audio input ----
    InputAudioRaw(AudioRawData),

    // ---- Processor control ----
    PauseProcessor        { name: String },
    PauseProcessorUrgent  { name: String },
    ResumeProcessor       { name: String },
    ResumeProcessorUrgent { name: String },

    // ---- Pipeline health probe ----
    Heartbeat(f64),

    // ---- RAVI protocol ----

    /// Parsed inbound client message from the transport.
    /// `data` is the raw JSON string of the `data` field, if present.
    RaviClientMessage {
        msg_id:   String,
        msg_type: String,
        data:     Option<String>,
    },

    /// Pre-serialised JSON to send to the client (pushed downstream to the
    /// output transport, which emits it as a WebSocket text frame).
    RaviServerMessage {
        payload: String,
    },

    /// Pre-serialised JSON response to a specific client request.
    /// Carries `client_msg_id` so the observer / transport can correlate it.
    RaviServerResponse {
        client_msg_id: String,
        payload:       String,
    },
}

// ---------------------------------------------------------------------------
// ControlFrame / DataFrame
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum ControlFrame {
    End { reason: Option<String> },
    LLMFullResponseStart,
    LLMFullResponseEnd,
}

#[derive(Debug, Clone)]
pub enum DataFrame {
    Data(DataFrameData),
    OutputAudioRaw(AudioRawData),
    Transcription(TranscriptionData),
    LLMText(String),
    LLMContextFrame(Arc<Mutex<crate::context::LLMContext>>),
}

#[derive(Debug, Clone)]
pub enum FrameInner {
    System(SystemFrame),
    Control(ControlFrame),
    Data(DataFrame),
}

// ---------------------------------------------------------------------------
// Frame
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Frame {
    pub id:         u64,
    pub sibling_id: Option<u64>,
    pub inner:      FrameInner,
}

// ---------------------------------------------------------------------------
// Core accessors
// ---------------------------------------------------------------------------

impl Frame {
    pub fn name(&self) -> &'static str {
        match &self.inner {
            FrameInner::System(s) => match s {
                SystemFrame::Start(_)                      => "StartFrame",
                SystemFrame::Cancel { .. }                 => "CancelFrame",
                SystemFrame::Error(_)                      => "ErrorFrame",
                SystemFrame::Interruption                  => "InterruptionFrame",
                SystemFrame::Stop { .. }                   => "StopFrame",
                SystemFrame::EndTask { .. }                => "EndTaskFrame",
                SystemFrame::CancelTask { .. }             => "CancelTaskFrame",
                SystemFrame::StopTask                      => "StopTaskFrame",
                SystemFrame::InterruptionTask              => "InterruptionTaskFrame",
                SystemFrame::BotSpeaking                   => "BotSpeakingFrame",
                SystemFrame::UserSpeaking                  => "UserSpeakingFrame",
                SystemFrame::BotStartedSpeaking            => "BotStartedSpeakingFrame",
                SystemFrame::BotStoppedSpeaking            => "BotStoppedSpeakingFrame",
                SystemFrame::UserStartedSpeaking { .. }    => "UserStartedSpeakingFrame",
                SystemFrame::UserStoppedSpeaking { .. }    => "UserStoppedSpeakingFrame",
                SystemFrame::VADUserStartedSpeaking { .. } => "VADUserStartedSpeakingFrame",
                SystemFrame::VADUserStoppedSpeaking { .. } => "VADUserStoppedSpeakingFrame",
                SystemFrame::InputAudioRaw(_)              => "InputAudioRawFrame",
                SystemFrame::PauseProcessor { .. }         => "PauseProcessorFrame",
                SystemFrame::PauseProcessorUrgent { .. }   => "PauseProcessorUrgentFrame",
                SystemFrame::ResumeProcessor { .. }        => "ResumeProcessorFrame",
                SystemFrame::ResumeProcessorUrgent { .. }  => "ResumeProcessorUrgentFrame",
                SystemFrame::Heartbeat(_)                  => "HeartbeatFrame",
                SystemFrame::RaviClientMessage { .. }      => "RaviClientMessageFrame",
                SystemFrame::RaviServerMessage { .. }      => "RaviServerMessageFrame",
                SystemFrame::RaviServerResponse { .. }     => "RaviServerResponseFrame",
            },
            FrameInner::Control(c) => match c {
                ControlFrame::End { .. }           => "EndFrame",
                ControlFrame::LLMFullResponseStart => "LLMFullResponseStartFrame",
                ControlFrame::LLMFullResponseEnd   => "LLMFullResponseEndFrame",
            },
            FrameInner::Data(d) => match d {
                DataFrame::Data(_)             => "DataFrame",
                DataFrame::OutputAudioRaw(_)   => "OutputAudioRawFrame",
                DataFrame::Transcription(_)    => "TranscriptionFrame",
                DataFrame::LLMText(_)          => "LLMTextFrame",
                DataFrame::LLMContextFrame(_)  => "LLMContextFrame",
            },
        }
    }

    pub fn kind(&self) -> FrameKind {
        match &self.inner {
            FrameInner::System(s) => match s {
                SystemFrame::Start(_)                      => FrameKind::Start,
                SystemFrame::Cancel { .. }                 => FrameKind::Cancel,
                SystemFrame::Error(_)                      => FrameKind::Error,
                SystemFrame::Interruption                  => FrameKind::Interruption,
                SystemFrame::Stop { .. }                   => FrameKind::Stop,
                SystemFrame::EndTask { .. }                => FrameKind::EndTask,
                SystemFrame::CancelTask { .. }             => FrameKind::CancelTask,
                SystemFrame::StopTask                      => FrameKind::StopTask,
                SystemFrame::InterruptionTask              => FrameKind::InterruptionTask,
                SystemFrame::BotSpeaking                   => FrameKind::BotSpeaking,
                SystemFrame::UserSpeaking                  => FrameKind::UserSpeaking,
                SystemFrame::BotStartedSpeaking            => FrameKind::BotStartedSpeaking,
                SystemFrame::BotStoppedSpeaking            => FrameKind::BotStoppedSpeaking,
                SystemFrame::UserStartedSpeaking { .. }    => FrameKind::UserStartedSpeaking,
                SystemFrame::UserStoppedSpeaking { .. }    => FrameKind::UserStoppedSpeaking,
                SystemFrame::VADUserStartedSpeaking { .. } => FrameKind::VADUserStartedSpeaking,
                SystemFrame::VADUserStoppedSpeaking { .. } => FrameKind::VADUserStoppedSpeaking,
                SystemFrame::InputAudioRaw(_)              => FrameKind::InputAudioRaw,
                SystemFrame::PauseProcessor { .. }         => FrameKind::PauseProcessor,
                SystemFrame::PauseProcessorUrgent { .. }   => FrameKind::PauseProcessorUrgent,
                SystemFrame::ResumeProcessor { .. }        => FrameKind::ResumeProcessor,
                SystemFrame::ResumeProcessorUrgent { .. }  => FrameKind::ResumeProcessorUrgent,
                SystemFrame::Heartbeat(_)                  => FrameKind::Heartbeat,
                SystemFrame::RaviClientMessage { .. }      => FrameKind::RaviClientMessage,
                SystemFrame::RaviServerMessage { .. }      => FrameKind::RaviServerMessage,
                SystemFrame::RaviServerResponse { .. }     => FrameKind::RaviServerResponse,
            },
            FrameInner::Control(c) => match c {
                ControlFrame::End { .. }           => FrameKind::End,
                ControlFrame::LLMFullResponseStart => FrameKind::LLMFullResponseStart,
                ControlFrame::LLMFullResponseEnd   => FrameKind::LLMFullResponseEnd,
            },
            FrameInner::Data(d) => match d {
                DataFrame::Data(_)             => FrameKind::Data,
                DataFrame::OutputAudioRaw(_)   => FrameKind::OutputAudioRaw,
                DataFrame::Transcription(_)    => FrameKind::Transcription,
                DataFrame::LLMText(_)          => FrameKind::LLMText,
                DataFrame::LLMContextFrame(_)  => FrameKind::LLMContextFrame,
            },
        }
    }

    pub fn is_system(&self) -> bool {
        matches!(self.inner, FrameInner::System(_))
    }

    pub fn is_uninterruptible(&self) -> bool {
        matches!(
            &self.inner,
            FrameInner::Control(ControlFrame::End { .. })
                | FrameInner::System(SystemFrame::EndTask { .. })
                | FrameInner::System(SystemFrame::StopTask)
                | FrameInner::System(SystemFrame::CancelTask { .. })
        )
    }
}

// ---------------------------------------------------------------------------
// Mutation helpers
// ---------------------------------------------------------------------------

impl Frame {
    pub fn with_new_id(self) -> Self { Self { id: next_frame_id(), ..self } }
    pub fn with_sibling(self, sibling_id: u64) -> Self { Self { sibling_id: Some(sibling_id), ..self } }
}

// ---------------------------------------------------------------------------
// Internal constructor
// ---------------------------------------------------------------------------

impl Frame {
    fn make(inner: FrameInner) -> Self {
        Self { id: next_frame_id(), sibling_id: None, inner }
    }
}

// ---------------------------------------------------------------------------
// Public constructors
// ---------------------------------------------------------------------------

impl Frame {
    // ---- Lifecycle ----
    pub fn start(data: StartFrameData) -> Self { Self::make(FrameInner::System(SystemFrame::Start(data))) }
    pub fn cancel() -> Self { Self::make(FrameInner::System(SystemFrame::Cancel { reason: None })) }
    pub fn cancel_with(reason: impl Into<String>) -> Self {
        Self::make(FrameInner::System(SystemFrame::Cancel { reason: Some(reason.into()) }))
    }
    pub fn error(msg: impl Into<String>, fatal: bool, processor: Option<String>) -> Self {
        Self::make(FrameInner::System(SystemFrame::Error(ErrorFrameData {
            error: msg.into(), fatal, processor_name: processor,
        })))
    }
    pub fn interruption() -> Self { Self::make(FrameInner::System(SystemFrame::Interruption)) }
    pub fn stop() -> Self { Self::make(FrameInner::System(SystemFrame::Stop { reason: None })) }
    pub fn stop_with(reason: impl Into<String>) -> Self {
        Self::make(FrameInner::System(SystemFrame::Stop { reason: Some(reason.into()) }))
    }
    pub fn end_task() -> Self { Self::make(FrameInner::System(SystemFrame::EndTask { reason: None })) }
    pub fn cancel_task() -> Self { Self::make(FrameInner::System(SystemFrame::CancelTask { reason: None })) }
    pub fn stop_task() -> Self { Self::make(FrameInner::System(SystemFrame::StopTask)) }
    pub fn interruption_task() -> Self { Self::make(FrameInner::System(SystemFrame::InterruptionTask)) }

    // ---- Speaking signals ----
    pub fn bot_speaking() -> Self { Self::make(FrameInner::System(SystemFrame::BotSpeaking)) }
    pub fn user_speaking() -> Self { Self::make(FrameInner::System(SystemFrame::UserSpeaking)) }
    pub fn bot_started_speaking() -> Self { Self::make(FrameInner::System(SystemFrame::BotStartedSpeaking)) }
    pub fn bot_stopped_speaking() -> Self { Self::make(FrameInner::System(SystemFrame::BotStoppedSpeaking)) }
    pub fn user_started_speaking() -> Self {
        Self::make(FrameInner::System(SystemFrame::UserStartedSpeaking { emulated: false }))
    }
    pub fn user_started_speaking_emulated() -> Self {
        Self::make(FrameInner::System(SystemFrame::UserStartedSpeaking { emulated: true }))
    }
    pub fn user_stopped_speaking() -> Self {
        Self::make(FrameInner::System(SystemFrame::UserStoppedSpeaking { emulated: false }))
    }
    pub fn user_stopped_speaking_emulated() -> Self {
        Self::make(FrameInner::System(SystemFrame::UserStoppedSpeaking { emulated: true }))
    }
    pub fn vad_user_started_speaking(start_secs: f32, timestamp: f64) -> Self {
        Self::make(FrameInner::System(SystemFrame::VADUserStartedSpeaking { start_secs, timestamp }))
    }
    pub fn vad_user_stopped_speaking(stop_secs: f32, timestamp: f64) -> Self {
        Self::make(FrameInner::System(SystemFrame::VADUserStoppedSpeaking { stop_secs, timestamp }))
    }

    // ---- Audio ----
    pub fn input_audio_raw(data: AudioRawData) -> Self {
        Self::make(FrameInner::System(SystemFrame::InputAudioRaw(data)))
    }
    pub fn input_audio(audio: Vec<u8>, sample_rate: u32, num_channels: u16) -> Self {
        Self::input_audio_raw(AudioRawData::new(audio, sample_rate, num_channels))
    }
    pub fn output_audio_raw(data: AudioRawData) -> Self {
        Self::make(FrameInner::Data(DataFrame::OutputAudioRaw(data)))
    }
    pub fn output_audio(audio: Vec<u8>, sample_rate: u32, num_channels: u16) -> Self {
        Self::output_audio_raw(AudioRawData::new(audio, sample_rate, num_channels))
    }

    // ---- Processor control ----
    pub fn pause_processor(name: impl Into<String>) -> Self {
        Self::make(FrameInner::System(SystemFrame::PauseProcessor { name: name.into() }))
    }
    pub fn pause_processor_urgent(name: impl Into<String>) -> Self {
        Self::make(FrameInner::System(SystemFrame::PauseProcessorUrgent { name: name.into() }))
    }
    pub fn resume_processor(name: impl Into<String>) -> Self {
        Self::make(FrameInner::System(SystemFrame::ResumeProcessor { name: name.into() }))
    }
    pub fn resume_processor_urgent(name: impl Into<String>) -> Self {
        Self::make(FrameInner::System(SystemFrame::ResumeProcessorUrgent { name: name.into() }))
    }
    pub fn heartbeat(ts: f64) -> Self {
        Self::make(FrameInner::System(SystemFrame::Heartbeat(ts)))
    }

    // ---- Control ----
    pub fn end() -> Self { Self::make(FrameInner::Control(ControlFrame::End { reason: None })) }
    pub fn end_with(reason: impl Into<String>) -> Self {
        Self::make(FrameInner::Control(ControlFrame::End { reason: Some(reason.into()) }))
    }

    // ---- Generic data ----
    pub fn transcription(data: TranscriptionData) -> Self {
        Self::make(FrameInner::Data(DataFrame::Transcription(data)))
    }
    pub fn data(content: Vec<u8>) -> Self {
        Self::make(FrameInner::Data(DataFrame::Data(DataFrameData { content, ..Default::default() })))
    }

    // ---- LLM ----
    pub fn llm_full_response_start() -> Self {
        Self::make(FrameInner::Control(ControlFrame::LLMFullResponseStart))
    }
    pub fn llm_full_response_end() -> Self {
        Self::make(FrameInner::Control(ControlFrame::LLMFullResponseEnd))
    }
    pub fn llm_text(text: String) -> Self {
        Self::make(FrameInner::Data(DataFrame::LLMText(text)))
    }
    pub fn llm_context(context: Arc<Mutex<crate::context::LLMContext>>) -> Self {
        Self::make(FrameInner::Data(DataFrame::LLMContextFrame(context)))
    }

    // ---- RAVI protocol ----

    /// Inbound: parsed client message from the WebSocket transport.
    pub fn ravi_client_message(
        msg_id:   impl Into<String>,
        msg_type: impl Into<String>,
        data:     Option<String>,
    ) -> Self {
        Self::make(FrameInner::System(SystemFrame::RaviClientMessage {
            msg_id:   msg_id.into(),
            msg_type: msg_type.into(),
            data,
        }))
    }

    /// Outbound: pre-serialised JSON to send to the client.
    pub fn ravi_server_message(payload: impl Into<String>) -> Self {
        Self::make(FrameInner::System(SystemFrame::RaviServerMessage {
            payload: payload.into(),
        }))
    }

    /// Outbound response to a specific client request.
    pub fn ravi_server_response(
        client_msg_id: impl Into<String>,
        payload:       impl Into<String>,
    ) -> Self {
        Self::make(FrameInner::System(SystemFrame::RaviServerResponse {
            client_msg_id: client_msg_id.into(),
            payload:       payload.into(),
        }))
    }
}
