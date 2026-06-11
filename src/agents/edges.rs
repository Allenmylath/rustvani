//! Bus edge processors — bridge any pipeline onto the agent bus without a
//! custom mid-pipeline proxy.
//!
//! [`BusOutputEdge`] sits at the **tail** of a bridged pipeline: every frame
//! is forwarded through unchanged (downstream stays intact) and, when the
//! activation gate is open and the frame kind is not excluded, published to
//! the bus as a [`BusPayload::Frame`] for each configured target.
//!
//! The **input** side needs no processor: bridged frames already enter the
//! owning agent's pipeline head via its push channel, filtered by the
//! agent's bridged peer list (`BaseAgent`'s `bridged: Option<Vec<String>>`).
//! Frames re-entering at the pipeline head are acceptable because rustvani
//! processors pass through frames they don't handle — but beware of
//! two-way bridges: exclude the frame kinds that the peer publishes, or
//! re-injected frames will ping-pong between the pipelines forever.
//!
//! Lifecycle frames (`Start`/`End`/`Cancel`/`Stop`) are never published —
//! lifecycle travels as bus control payloads, not frames.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::frames::{Frame, FrameDirection, FrameHandler, FrameKind, FrameProcessor};

use super::bus::{AgentBus, BusMessage, BusPayload};

/// Frame kinds that are never published, regardless of configuration:
/// pipeline lifecycle, task-control, processor-control, and health probes.
/// These are internal to the owning pipeline; lifecycle crosses the bus as
/// control payloads (`End`/`Cancel`), not as frames.
fn default_excluded(kind: FrameKind) -> bool {
    matches!(
        kind,
        FrameKind::Start
            | FrameKind::End
            | FrameKind::Cancel
            | FrameKind::Stop
            | FrameKind::Error
            | FrameKind::Heartbeat
            | FrameKind::EndTask
            | FrameKind::CancelTask
            | FrameKind::StopTask
            | FrameKind::InterruptionTask
            | FrameKind::PauseProcessor
            | FrameKind::PauseProcessorUrgent
            | FrameKind::ResumeProcessor
            | FrameKind::ResumeProcessorUrgent
    )
}

struct EdgeInner {
    source_name: String,
    /// Peer agents to publish to. Empty = broadcast.
    targets: Vec<String>,
    /// Additional frame kinds excluded from publishing.
    exclude: HashSet<FrameKind>,
    /// Activation gate, normally shared with the owning agent. When unset
    /// the edge is always open.
    active: tokio::sync::OnceCell<Arc<AtomicBool>>,
    /// Bus handle, injected by the agent during `setup()`.
    bus: tokio::sync::OnceCell<Arc<dyn AgentBus>>,
}

impl EdgeInner {
    fn gate_open(&self) -> bool {
        self.active
            .get()
            .map(|a| a.load(Ordering::Relaxed))
            .unwrap_or(true)
    }

    fn should_publish(&self, frame: &Frame) -> bool {
        let kind = frame.kind();
        !default_excluded(kind) && !self.exclude.contains(&kind)
    }
}

/// Tail-of-pipeline processor that publishes passing frames onto the bus.
///
/// Cloneable handle: keep a clone for [`BusOutputEdge::set_bus`] /
/// [`BusOutputEdge::bind_activation`] and convert one into the actual
/// pipeline processor with [`BusOutputEdge::to_processor`]. The easiest
/// wiring is [`super::BaseAgent::bridged_pipeline`], which does all of this.
#[derive(Clone)]
pub struct BusOutputEdge {
    inner: Arc<EdgeInner>,
}

impl BusOutputEdge {
    /// Create an edge publishing as `source_name` to `targets`
    /// (empty = broadcast to all subscribers).
    pub fn new(source_name: impl Into<String>, targets: Vec<String>) -> Self {
        Self {
            inner: Arc::new(EdgeInner {
                source_name: source_name.into(),
                targets,
                exclude: HashSet::new(),
                active: tokio::sync::OnceCell::new(),
                bus: tokio::sync::OnceCell::new(),
            }),
        }
    }

    /// Create an edge with additional excluded frame kinds (on top of the
    /// always-excluded lifecycle/control/probe kinds).
    pub fn with_exclude(
        source_name: impl Into<String>,
        targets: Vec<String>,
        exclude: HashSet<FrameKind>,
    ) -> Self {
        Self {
            inner: Arc::new(EdgeInner {
                source_name: source_name.into(),
                targets,
                exclude,
                active: tokio::sync::OnceCell::new(),
                bus: tokio::sync::OnceCell::new(),
            }),
        }
    }

    /// Inject the bus handle. Called by the owning agent during `setup()`.
    /// Until set, the edge forwards frames locally but publishes nothing.
    /// Setting twice is a no-op.
    pub fn set_bus(&self, bus: Arc<dyn AgentBus>) {
        let _ = self.inner.bus.set(bus);
    }

    /// Share an activation flag (normally the owning agent's). When the
    /// flag is `false` the edge forwards frames locally but publishes
    /// nothing. If never bound, the edge is always open. Binding twice is
    /// a no-op.
    pub fn bind_activation(&self, active: Arc<AtomicBool>) {
        let _ = self.inner.active.set(active);
    }

    /// Build the pipeline processor for this edge. Place it at the tail of
    /// the bridged pipeline.
    pub fn to_processor(&self) -> FrameProcessor {
        FrameProcessor::new(
            format!("BusOutputEdge({})", self.inner.source_name),
            Box::new(BusOutputEdgeHandler {
                inner: self.inner.clone(),
            }),
            true, // direct mode — forwards inline, no extra task loops
        )
    }
}

struct BusOutputEdgeHandler {
    inner: Arc<EdgeInner>,
}

#[async_trait]
impl FrameHandler for BusOutputEdgeHandler {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        if self.inner.should_publish(&frame) && self.inner.gate_open() {
            if let Some(bus) = self.inner.bus.get() {
                if self.inner.targets.is_empty() {
                    bus.send(BusMessage::new(
                        self.inner.source_name.clone(),
                        None,
                        BusPayload::Frame {
                            frame: frame.clone(),
                            direction,
                        },
                    ))
                    .await;
                } else {
                    for target in &self.inner.targets {
                        bus.send(BusMessage::new(
                            self.inner.source_name.clone(),
                            Some(target.clone()),
                            BusPayload::Frame {
                                frame: frame.clone(),
                                direction,
                            },
                        ))
                        .await;
                    }
                }
            }
        }

        // Always forward through unchanged — downstream stays intact.
        processor.push_frame(frame, direction).await
    }
}
