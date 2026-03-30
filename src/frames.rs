use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

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
    // System
    Start, Cancel, Error, Interruption, Stop,
    EndTask, CancelTask, StopTask, InterruptionTask,
    BotSpeaking, UserSpeaking,
    PauseProcessor, PauseProcessorUrgent,
    ResumeProcessor, ResumeProcessorUrgent,
    Heartbeat,
    // Control
    End,
    // Data
    Data,
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

// ---------------------------------------------------------------------------
// Nested enums
// ---------------------------------------------------------------------------

/// High-priority frames — bypass the non-system queue and are processed immediately.
#[derive(Debug, Clone)]
pub enum SystemFrame {
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

    // Speaking signals (also reset idle timer)
    BotSpeaking,
    UserSpeaking,

    // Pause / resume a named processor
    PauseProcessor        { name: String },
    PauseProcessorUrgent  { name: String },
    ResumeProcessor       { name: String },
    ResumeProcessorUrgent { name: String },

    // Pipeline health probe
    Heartbeat(f64),
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
                SystemFrame::Start(_)                  => "StartFrame",
                SystemFrame::Cancel { .. }             => "CancelFrame",
                SystemFrame::Error(_)                  => "ErrorFrame",
                SystemFrame::Interruption              => "InterruptionFrame",
                SystemFrame::Stop { .. }               => "StopFrame",
                SystemFrame::EndTask { .. }            => "EndTaskFrame",
                SystemFrame::CancelTask { .. }         => "CancelTaskFrame",
                SystemFrame::StopTask                  => "StopTaskFrame",
                SystemFrame::InterruptionTask          => "InterruptionTaskFrame",
                SystemFrame::BotSpeaking               => "BotSpeakingFrame",
                SystemFrame::UserSpeaking              => "UserSpeakingFrame",
                SystemFrame::PauseProcessor { .. }     => "PauseProcessorFrame",
                SystemFrame::PauseProcessorUrgent{..}  => "PauseProcessorUrgentFrame",
                SystemFrame::ResumeProcessor { .. }    => "ResumeProcessorFrame",
                SystemFrame::ResumeProcessorUrgent{..} => "ResumeProcessorUrgentFrame",
                SystemFrame::Heartbeat(_)              => "HeartbeatFrame",
            },
            FrameInner::Control(c) => match c {
                ControlFrame::End { .. } => "EndFrame",
            },
            FrameInner::Data(d) => match d {
                DataFrame::Data(_) => "DataFrame",
            },
        }
    }

    pub fn kind(&self) -> FrameKind {
        match &self.inner {
            FrameInner::System(s) => match s {
                SystemFrame::Start(_)                  => FrameKind::Start,
                SystemFrame::Cancel { .. }             => FrameKind::Cancel,
                SystemFrame::Error(_)                  => FrameKind::Error,
                SystemFrame::Interruption              => FrameKind::Interruption,
                SystemFrame::Stop { .. }               => FrameKind::Stop,
                SystemFrame::EndTask { .. }            => FrameKind::EndTask,
                SystemFrame::CancelTask { .. }         => FrameKind::CancelTask,
                SystemFrame::StopTask                  => FrameKind::StopTask,
                SystemFrame::InterruptionTask          => FrameKind::InterruptionTask,
                SystemFrame::BotSpeaking               => FrameKind::BotSpeaking,
                SystemFrame::UserSpeaking              => FrameKind::UserSpeaking,
                SystemFrame::PauseProcessor { .. }     => FrameKind::PauseProcessor,
                SystemFrame::PauseProcessorUrgent{..}  => FrameKind::PauseProcessorUrgent,
                SystemFrame::ResumeProcessor { .. }    => FrameKind::ResumeProcessor,
                SystemFrame::ResumeProcessorUrgent{..} => FrameKind::ResumeProcessorUrgent,
                SystemFrame::Heartbeat(_)              => FrameKind::Heartbeat,
            },
            FrameInner::Control(c) => match c {
                ControlFrame::End { .. } => FrameKind::End,
            },
            FrameInner::Data(d) => match d {
                DataFrame::Data(_) => FrameKind::Data,
            },
        }
    }

    /// System frames bypass the non-system queue.
    pub fn is_system(&self) -> bool {
        matches!(self.inner, FrameInner::System(_))
    }

    /// Uninterruptible frames survive interruption queue drains.
    /// Mirrors Python's `UninterruptibleFrame` mixin on specific types.
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
// Mutation helpers — one-liners thanks to struct layout
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
// Internal constructors
// ---------------------------------------------------------------------------

impl Frame {
    fn make(inner: FrameInner) -> Self {
        Self { id: next_frame_id(), sibling_id: None, inner }
    }
}

// ---------------------------------------------------------------------------
// Public constructors — no `new_` prefix
// ---------------------------------------------------------------------------

impl Frame {
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

    pub fn bot_speaking() -> Self {
        Self::make(FrameInner::System(SystemFrame::BotSpeaking))
    }

    pub fn user_speaking() -> Self {
        Self::make(FrameInner::System(SystemFrame::UserSpeaking))
    }

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

    pub fn end() -> Self {
        Self::make(FrameInner::Control(ControlFrame::End { reason: None }))
    }

    pub fn end_with(reason: impl Into<String>) -> Self {
        Self::make(FrameInner::Control(ControlFrame::End {
            reason: Some(reason.into()),
        }))
    }

    pub fn data(content: Vec<u8>) -> Self {
        Self::make(FrameInner::Data(DataFrame::Data(DataFrameData {
            content,
            ..Default::default()
        })))
    }
}
