use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Global frame ID counter
// ---------------------------------------------------------------------------

static FRAME_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

pub fn next_frame_id() -> u64 {
    FRAME_ID_COUNTER.fetch_add(1, Ordering::SeqCst)
}

// ---------------------------------------------------------------------------
// Frame data payloads
// ---------------------------------------------------------------------------

/// Payload for StartFrame — carries pipeline-wide configuration.
#[derive(Debug, Clone, Default)]
pub struct StartFrameData {
    pub allow_interruptions: bool,
    pub enable_metrics: bool,
    pub enable_usage_metrics: bool,
    pub report_only_initial_ttfb: bool,
    /// Arbitrary key/value metadata (e.g. "deprecated_openaillmcontext").
    pub metadata: HashMap<String, String>,
    pub broadcast_sibling_id: Option<u64>,
}

/// Payload for ErrorFrame.
#[derive(Debug, Clone)]
pub struct ErrorFrameData {
    pub error: String,
    pub fatal: bool,
    /// Name of the processor that raised the error.
    pub processor_name: Option<String>,
    pub broadcast_sibling_id: Option<u64>,
}

/// Payload for Pause/Resume frames — targets a specific processor by name.
#[derive(Debug, Clone)]
pub struct PauseResumeFrameData {
    pub processor_name: String,
    pub broadcast_sibling_id: Option<u64>,
}

/// Generic data frame payload.
#[derive(Debug, Clone, Default)]
pub struct DataFrameData {
    pub content: Vec<u8>,
    pub metadata: HashMap<String, String>,
    pub broadcast_sibling_id: Option<u64>,
}

// ---------------------------------------------------------------------------
// The Frame enum
// ---------------------------------------------------------------------------

/// All frame types in the pipeline.
///
/// **System frames** (high-priority, processed before any non-system frame):
/// - `Start`, `Cancel`, `Error`, `Interruption`, `End`
/// - `PauseProcessing`, `PauseProcessingUrgent`
/// - `ResumeProcessing`, `ResumeProcessingUrgent`
///
/// **Non-system frames**:
/// - `Data` — generic data frame
/// - `Uninterruptible(id, Box<Frame>)` — wraps any frame, immune to interruption draining
#[derive(Debug, Clone)]
pub enum Frame {
    // ---- System frames ----
    Start(u64, StartFrameData),
    Cancel(u64, Option<u64>),                    // (id, broadcast_sibling_id)
    Error(u64, ErrorFrameData),
    Interruption(u64, Option<u64>),              // (id, broadcast_sibling_id)
    End(u64, Option<u64>),                       // (id, broadcast_sibling_id)
    PauseProcessing(u64, PauseResumeFrameData),
    PauseProcessingUrgent(u64, PauseResumeFrameData),
    ResumeProcessing(u64, PauseResumeFrameData),
    ResumeProcessingUrgent(u64, PauseResumeFrameData),

    // ---- Non-system frames ----
    Data(u64, DataFrameData),

    /// Wraps any frame to make it immune to interruption-queue draining.
    Uninterruptible(u64, Box<Frame>),
}

// ---------------------------------------------------------------------------
// Core accessors
// ---------------------------------------------------------------------------

impl Frame {
    pub fn id(&self) -> u64 {
        match self {
            Frame::Start(id, _)
            | Frame::Cancel(id, _)
            | Frame::Error(id, _)
            | Frame::Interruption(id, _)
            | Frame::End(id, _)
            | Frame::PauseProcessing(id, _)
            | Frame::PauseProcessingUrgent(id, _)
            | Frame::ResumeProcessing(id, _)
            | Frame::ResumeProcessingUrgent(id, _)
            | Frame::Data(id, _)
            | Frame::Uninterruptible(id, _) => *id,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Frame::Start(_, _) => "StartFrame",
            Frame::Cancel(_, _) => "CancelFrame",
            Frame::Error(_, _) => "ErrorFrame",
            Frame::Interruption(_, _) => "InterruptionFrame",
            Frame::End(_, _) => "EndFrame",
            Frame::PauseProcessing(_, _) => "FrameProcessorPauseFrame",
            Frame::PauseProcessingUrgent(_, _) => "FrameProcessorPauseUrgentFrame",
            Frame::ResumeProcessing(_, _) => "FrameProcessorResumeFrame",
            Frame::ResumeProcessingUrgent(_, _) => "FrameProcessorResumeUrgentFrame",
            Frame::Data(_, _) => "DataFrame",
            Frame::Uninterruptible(_, _) => "UninterruptibleFrame",
        }
    }

    /// Returns `true` for frames that receive high-priority queue processing.
    pub fn is_system(&self) -> bool {
        matches!(
            self,
            Frame::Start(_, _)
                | Frame::Cancel(_, _)
                | Frame::Error(_, _)
                | Frame::Interruption(_, _)
                | Frame::End(_, _)
                | Frame::PauseProcessing(_, _)
                | Frame::PauseProcessingUrgent(_, _)
                | Frame::ResumeProcessing(_, _)
                | Frame::ResumeProcessingUrgent(_, _)
        )
    }

    /// Returns `true` if this frame must not be discarded during an interruption drain.
    pub fn is_uninterruptible(&self) -> bool {
        matches!(self, Frame::Uninterruptible(_, _))
    }

    pub fn broadcast_sibling_id(&self) -> Option<u64> {
        match self {
            Frame::Start(_, d) => d.broadcast_sibling_id,
            Frame::Cancel(_, s) => *s,
            Frame::Error(_, d) => d.broadcast_sibling_id,
            Frame::Interruption(_, s) => *s,
            Frame::End(_, s) => *s,
            Frame::PauseProcessing(_, d) => d.broadcast_sibling_id,
            Frame::PauseProcessingUrgent(_, d) => d.broadcast_sibling_id,
            Frame::ResumeProcessing(_, d) => d.broadcast_sibling_id,
            Frame::ResumeProcessingUrgent(_, d) => d.broadcast_sibling_id,
            Frame::Data(_, d) => d.broadcast_sibling_id,
            Frame::Uninterruptible(_, _) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Mutation helpers (for broadcast)
// ---------------------------------------------------------------------------

impl Frame {
    /// Return this frame with a fresh ID (all other data preserved).
    pub fn with_new_id(self) -> Self {
        let new_id = next_frame_id();
        match self {
            Frame::Start(_, d) => Frame::Start(new_id, d),
            Frame::Cancel(_, s) => Frame::Cancel(new_id, s),
            Frame::Error(_, d) => Frame::Error(new_id, d),
            Frame::Interruption(_, s) => Frame::Interruption(new_id, s),
            Frame::End(_, s) => Frame::End(new_id, s),
            Frame::PauseProcessing(_, d) => Frame::PauseProcessing(new_id, d),
            Frame::PauseProcessingUrgent(_, d) => Frame::PauseProcessingUrgent(new_id, d),
            Frame::ResumeProcessing(_, d) => Frame::ResumeProcessing(new_id, d),
            Frame::ResumeProcessingUrgent(_, d) => Frame::ResumeProcessingUrgent(new_id, d),
            Frame::Data(_, d) => Frame::Data(new_id, d),
            Frame::Uninterruptible(_, inner) => Frame::Uninterruptible(new_id, inner),
        }
    }

    /// Return this frame with `broadcast_sibling_id` set.
    pub fn with_sibling(self, sibling_id: u64) -> Self {
        match self {
            Frame::Start(id, mut d) => {
                d.broadcast_sibling_id = Some(sibling_id);
                Frame::Start(id, d)
            }
            Frame::Cancel(id, _) => Frame::Cancel(id, Some(sibling_id)),
            Frame::Error(id, mut d) => {
                d.broadcast_sibling_id = Some(sibling_id);
                Frame::Error(id, d)
            }
            Frame::Interruption(id, _) => Frame::Interruption(id, Some(sibling_id)),
            Frame::End(id, _) => Frame::End(id, Some(sibling_id)),
            Frame::PauseProcessing(id, mut d) => {
                d.broadcast_sibling_id = Some(sibling_id);
                Frame::PauseProcessing(id, d)
            }
            Frame::PauseProcessingUrgent(id, mut d) => {
                d.broadcast_sibling_id = Some(sibling_id);
                Frame::PauseProcessingUrgent(id, d)
            }
            Frame::ResumeProcessing(id, mut d) => {
                d.broadcast_sibling_id = Some(sibling_id);
                Frame::ResumeProcessing(id, d)
            }
            Frame::ResumeProcessingUrgent(id, mut d) => {
                d.broadcast_sibling_id = Some(sibling_id);
                Frame::ResumeProcessingUrgent(id, d)
            }
            Frame::Data(id, mut d) => {
                d.broadcast_sibling_id = Some(sibling_id);
                Frame::Data(id, d)
            }
            other => other, // Uninterruptible has no sibling field
        }
    }
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

impl Frame {
    pub fn new_start(data: StartFrameData) -> Self {
        Frame::Start(next_frame_id(), data)
    }

    pub fn new_cancel() -> Self {
        Frame::Cancel(next_frame_id(), None)
    }

    pub fn new_error(
        error: impl Into<String>,
        fatal: bool,
        processor_name: Option<String>,
    ) -> Self {
        Frame::Error(
            next_frame_id(),
            ErrorFrameData {
                error: error.into(),
                fatal,
                processor_name,
                broadcast_sibling_id: None,
            },
        )
    }

    pub fn new_interruption() -> Self {
        Frame::Interruption(next_frame_id(), None)
    }

    pub fn new_end() -> Self {
        Frame::End(next_frame_id(), None)
    }

    pub fn new_data(content: Vec<u8>) -> Self {
        Frame::Data(
            next_frame_id(),
            DataFrameData {
                content,
                ..Default::default()
            },
        )
    }

    pub fn new_pause(processor_name: impl Into<String>) -> Self {
        Frame::PauseProcessing(
            next_frame_id(),
            PauseResumeFrameData {
                processor_name: processor_name.into(),
                broadcast_sibling_id: None,
            },
        )
    }

    pub fn new_pause_urgent(processor_name: impl Into<String>) -> Self {
        Frame::PauseProcessingUrgent(
            next_frame_id(),
            PauseResumeFrameData {
                processor_name: processor_name.into(),
                broadcast_sibling_id: None,
            },
        )
    }

    pub fn new_resume(processor_name: impl Into<String>) -> Self {
        Frame::ResumeProcessing(
            next_frame_id(),
            PauseResumeFrameData {
                processor_name: processor_name.into(),
                broadcast_sibling_id: None,
            },
        )
    }

    pub fn new_resume_urgent(processor_name: impl Into<String>) -> Self {
        Frame::ResumeProcessingUrgent(
            next_frame_id(),
            PauseResumeFrameData {
                processor_name: processor_name.into(),
                broadcast_sibling_id: None,
            },
        )
    }

    pub fn new_uninterruptible(inner: Frame) -> Self {
        Frame::Uninterruptible(next_frame_id(), Box::new(inner))
    }
}
