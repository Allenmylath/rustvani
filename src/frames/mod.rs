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
use std::sync::{Arc, Mutex}; // added for LLMContextFrame

// ---------------------------------------------------------------------------
// Frame ID counter
// ---------------------------------------------------------------------------

static FRAME_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

pub fn next_frame_id() -> u64 {
    FRAME_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Flat FrameKind — for filter sets (HashSet<FrameKind>), no nesting tax
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
    // System — audio/video input (high-priority, bypass non-system queue)
    InputAudioRaw,
    // System — processor control
    PauseProcessor, PauseProcessorUrgent,
    ResumeProcessor, ResumeProcessorUrgent,
    Heartbeat,
    // System — LLM response boundaries
    LLMFullResponseStart,
    LLMFullResponseEnd,
    // Control
    End,
    // Data
    Data,
    Transcription,
    LLMText,
    LLMContextFrame,
    // Data — audio output (ordered, cancellable)
    OutputAudioRaw,
}

// ---------------------------------------------------------------------------
// Audio payload — shared by input and output audio frames
// ---------------------------------------------------------------------------

/// Raw PCM audio payload.
///
/// `audio` is interleaved 16-bit little-endian PCM.
/// `num_frames` = len(audio) / (num_channels * 2)
#[derive(Debug, Clone)]
pub struct AudioRawData {
    pub audio:            Vec<u8>,
    pub sample_rate:      u32,
    pub num_channels:     u16,
    /// Derived: number of PCM frames in `audio`.
    pub num_frames:       usize,
    /// Which transport track this came from / is going to.
    pub transport_source: Option<String>,
}

impl AudioRawData {
    pub fn new(audio: Vec<u8>, sample_rate: u32, num_channels: u16) -> Self {
        let num_frames = if num_channels > 0 {
            audio.len() / (num_channels as usize * 2)
        } else {
            0
        };
        Self {
            audio,
            sample_rate,
            num_channels,
            num_frames,
            transport_source: None,
        }
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
// Nested enums
// ---------------------------------------------------------------------------

/// High-priority frames — bypass the non-system queue and are processed immediately.
#[derive(Debug, Clone)]
pub enum SystemFrame {
    // ---- Lifecycle ----
    Start(StartFrameData),
    Cancel    { reason: Option<String> },
    Error(ErrorFrameData),
    Interruption,
    Stop      { reason: Option<String> },

    // Task-control: pushed upstream, converted by PipelineTask
    EndTask   { reason: Option<String> },
    CancelTask { reason: Option<String> },
    StopTask,
    InterruptionTask,

    // ---- Speaking signals ----

    /// Emitted periodically while the bot is speaking (keeps idle timer alive).
    BotSpeaking,
    /// Emitted periodically while the user is speaking (keeps idle timer alive).
    UserSpeaking,

    /// Emitted once when the bot starts producing audio output.
    BotStartedSpeaking,
    /// Emitted once when the bot stops producing audio output.
    BotStoppedSpeaking,

    /// Emitted when the user turn starts (VAD confirmed start).
    UserStartedSpeaking { emulated: bool },
    /// Emitted when the user turn ends (VAD confirmed stop).
    UserStoppedSpeaking  { emulated: bool },

    /// Emitted by the VAD layer when it definitively detects speech onset.
    /// `start_secs`: VAD start_secs threshold that triggered this event.
    VADUserStartedSpeaking { start_secs: f32, timestamp: f64 },
    /// Emitted by the VAD layer when it definitively detects speech end.
    /// `stop_secs`: VAD stop_secs silence threshold that triggered this event.
    VADUserStoppedSpeaking { stop_secs: f32, timestamp: f64 },

    // ---- Audio input (high-priority for fast VAD / STT path) ----
    /// Raw PCM audio arriving from an input transport.
    /// SystemFrame so it bypasses the non-system queue — VAD runs immediately.
    InputAudioRaw(AudioRawData),

    // ---- Processor control ----
    PauseProcessor        { name: String },
    PauseProcessorUrgent  { name: String },
    ResumeProcessor       { name: String },
    ResumeProcessorUrgent { name: String },

    // ---- Pipeline health probe ----
    Heartbeat(f64),

    // ---- LLM response boundaries ----
    /// Emitted by the LLM service before the first token arrives.
    LLMFullResponseStart,
    /// Emitted by the LLM service after [DONE] or on error — always fires.
    LLMFullResponseEnd,
}

/// Ordered frames that survive interruption drains.
#[derive(Debug, Clone)]
pub enum ControlFrame {
    End { reason: Option<String> },
}

/// Ordered frames that are cancelled by interruptions.
#[derive(Debug, Clone)]
pub enum DataFrame {
    Data(DataFrameData),
    /// Raw PCM audio heading to the output transport.
    /// DataFrame so it participates in the ordered queue and is cancelled
    /// cleanly on interruption (bot stops speaking when user interrupts).
    OutputAudioRaw(AudioRawData),
    Transcription(TranscriptionData),
    /// Individual text chunk from the LLM — forwarded to TTS as it arrives.
    LLMText(String),
    /// Carries shared conversation context — triggers LLM inference.
    /// Cancelled on interruption so a stale context does not re-run the LLM.
    LLMContextFrame(Arc<Mutex<crate::context::LLMContext>>),
}

/// Discriminant over System / Control / Data.
#[derive(Debug, Clone)]
pub enum FrameInner {
    System(SystemFrame),
    Control(ControlFrame),
    Data(DataFrame),
}

// ---------------------------------------------------------------------------
// Frame — the public type
// ---------------------------------------------------------------------------

/// Every frame in the pipeline.
///
/// `id` and `sibling_id` are universal fields on the struct so they're
/// always accessible without matching on the variant.
#[derive(Debug, Clone)]
pub struct Frame {
    pub id:         u64,
    /// Set by `broadcast_frame()` to link the two copies.
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
                SystemFrame::LLMFullResponseStart          => "LLMFullResponseStartFrame",
                SystemFrame::LLMFullResponseEnd            => "LLMFullResponseEndFrame",
            },
            FrameInner::Control(c) => match c {
                ControlFrame::End { .. } => "EndFrame",
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
                SystemFrame::LLMFullResponseStart          => FrameKind::LLMFullResponseStart,
                SystemFrame::LLMFullResponseEnd            => FrameKind::LLMFullResponseEnd,
            },
            FrameInner::Control(c) => match c {
                ControlFrame::End { .. } => FrameKind::End,
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

    /// System frames bypass the non-system queue.
    pub fn is_system(&self) -> bool {
        matches!(self.inner, FrameInner::System(_))
    }

    /// Uninterruptible frames survive interruption queue drains.
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
    pub fn with_new_id(self) -> Self {
        Self { id: next_frame_id(), ..self }
    }

    pub fn with_sibling(self, sibling_id: u64) -> Self {
        Self { sibling_id: Some(sibling_id), ..self }
    }
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

    pub fn start(data: StartFrameData) -> Self {
        Self::make(FrameInner::System(SystemFrame::Start(data)))
    }

    pub fn cancel() -> Self {
        Self::make(FrameInner::System(SystemFrame::Cancel { reason: None }))
    }

    pub fn cancel_with(reason: impl Into<String>) -> Self {
        Self::make(FrameInner::System(SystemFrame::Cancel {
            reason: Some(reason.into()),
        }))
    }

    pub fn error(msg: impl Into<String>, fatal: bool, processor: Option<String>) -> Self {
        Self::make(FrameInner::System(SystemFrame::Error(ErrorFrameData {
            error: msg.into(),
            fatal,
            processor_name: processor,
        })))
    }

    pub fn interruption() -> Self {
        Self::make(FrameInner::System(SystemFrame::Interruption))
    }

    pub fn stop() -> Self {
        Self::make(FrameInner::System(SystemFrame::Stop { reason: None }))
    }

    pub fn stop_with(reason: impl Into<String>) -> Self {
        Self::make(FrameInner::System(SystemFrame::Stop {
            reason: Some(reason.into()),
        }))
    }

    pub fn end_task() -> Self {
        Self::make(FrameInner::System(SystemFrame::EndTask { reason: None }))
    }

    pub fn cancel_task() -> Self {
        Self::make(FrameInner::System(SystemFrame::CancelTask { reason: None }))
    }

    pub fn stop_task() -> Self {
        Self::make(FrameInner::System(SystemFrame::StopTask))
    }

    pub fn interruption_task() -> Self {
        Self::make(FrameInner::System(SystemFrame::InterruptionTask))
    }

    // ---- Speaking signals ----

    pub fn bot_speaking() -> Self {
        Self::make(FrameInner::System(SystemFrame::BotSpeaking))
    }

    pub fn user_speaking() -> Self {
        Self::make(FrameInner::System(SystemFrame::UserSpeaking))
    }

    pub fn bot_started_speaking() -> Self {
        Self::make(FrameInner::System(SystemFrame::BotStartedSpeaking))
    }

    pub fn bot_stopped_speaking() -> Self {
        Self::make(FrameInner::System(SystemFrame::BotStoppedSpeaking))
    }

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
        Self::make(FrameInner::System(SystemFrame::VADUserStartedSpeaking {
            start_secs,
            timestamp,
        }))
    }

    pub fn vad_user_stopped_speaking(stop_secs: f32, timestamp: f64) -> Self {
        Self::make(FrameInner::System(SystemFrame::VADUserStoppedSpeaking {
            stop_secs,
            timestamp,
        }))
    }

    // ---- Audio ----

    /// Construct an input audio frame. High-priority (SystemFrame).
    pub fn input_audio_raw(data: AudioRawData) -> Self {
        Self::make(FrameInner::System(SystemFrame::InputAudioRaw(data)))
    }

    /// Convenience: build input audio from raw bytes.
    pub fn input_audio(
        audio: Vec<u8>,
        sample_rate: u32,
        num_channels: u16,
    ) -> Self {
        Self::input_audio_raw(AudioRawData::new(audio, sample_rate, num_channels))
    }

    /// Construct an output audio frame. Ordered + cancellable (DataFrame).
    pub fn output_audio_raw(data: AudioRawData) -> Self {
        Self::make(FrameInner::Data(DataFrame::OutputAudioRaw(data)))
    }

    /// Convenience: build output audio from raw bytes.
    pub fn output_audio(
        audio: Vec<u8>,
        sample_rate: u32,
        num_channels: u16,
    ) -> Self {
        Self::output_audio_raw(AudioRawData::new(audio, sample_rate, num_channels))
    }

    // ---- Processor control ----

    pub fn pause_processor(name: impl Into<String>) -> Self {
        Self::make(FrameInner::System(SystemFrame::PauseProcessor {
            name: name.into(),
        }))
    }

    pub fn pause_processor_urgent(name: impl Into<String>) -> Self {
        Self::make(FrameInner::System(SystemFrame::PauseProcessorUrgent {
            name: name.into(),
        }))
    }

    pub fn resume_processor(name: impl Into<String>) -> Self {
        Self::make(FrameInner::System(SystemFrame::ResumeProcessor {
            name: name.into(),
        }))
    }

    pub fn resume_processor_urgent(name: impl Into<String>) -> Self {
        Self::make(FrameInner::System(SystemFrame::ResumeProcessorUrgent {
            name: name.into(),
        }))
    }

    pub fn heartbeat(ts: f64) -> Self {
        Self::make(FrameInner::System(SystemFrame::Heartbeat(ts)))
    }

    // ---- Control ----

    pub fn end() -> Self {
        Self::make(FrameInner::Control(ControlFrame::End { reason: None }))
    }

    pub fn end_with(reason: impl Into<String>) -> Self {
        Self::make(FrameInner::Control(ControlFrame::End {
            reason: Some(reason.into()),
        }))
    }

    // ---- Generic data ----

    pub fn transcription(data: TranscriptionData) -> Self {
        Self::make(FrameInner::Data(DataFrame::Transcription(data)))
    }

    pub fn data(content: Vec<u8>) -> Self {
        Self::make(FrameInner::Data(DataFrame::Data(DataFrameData {
            content,
            ..Default::default()
        })))
    }

    // ---- LLM ----

    /// Signal start of LLM streaming response.
    pub fn llm_full_response_start() -> Self {
        Self::make(FrameInner::System(SystemFrame::LLMFullResponseStart))
    }

    /// Signal end of LLM streaming response — always emitted, even on error.
    pub fn llm_full_response_end() -> Self {
        Self::make(FrameInner::System(SystemFrame::LLMFullResponseEnd))
    }

    /// One SSE content chunk from the LLM.
    pub fn llm_text(text: String) -> Self {
        Self::make(FrameInner::Data(DataFrame::LLMText(text)))
    }

    /// Triggers LLM inference with the current conversation context.
    pub fn llm_context(context: Arc<Mutex<crate::context::LLMContext>>) -> Self {
        Self::make(FrameInner::Data(DataFrame::LLMContextFrame(context)))
    }
}
