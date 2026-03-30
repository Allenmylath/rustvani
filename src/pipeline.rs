//! Pipeline — chains FrameProcessors into a directed graph.
//!
//! `Pipeline::new(processors)` returns a plain `FrameProcessor` whose
//! handler (`PipelineHandler`) routes frames to an internal source/sink pair.
//! This means a Pipeline IS a FrameProcessor and can be nested inside another
//! Pipeline or placed in any processor list — no trait objects needed.
//!
//! Internal topology:
//!
//! ```text
//!  ┌─────────────────────────────────────────────────┐
//!  │  PipelineHandler (outer FrameProcessor)         │
//!  │                                                 │
//!  │  Downstream ──► source ──► p1 ──► p2 ──► sink  │
//!  │  Upstream   ◄── source ◄── p1 ◄── p2 ◄── sink  │
//!  └─────────────────────────────────────────────────┘
//! ```
//!
//! Frames escaping upstream past source (or downstream past sink) are
//! forwarded to the outer pipeline processor's neighbours.

use std::sync::{Arc, RwLock};

use async_trait::async_trait;

use crate::direction::FrameDirection;
use crate::error::Result;
use crate::frames::Frame;
use crate::processor::{FrameHandler, FrameProcessor, WeakFrameProcessor};

// ---------------------------------------------------------------------------
// PipelineSourceHandler
// ---------------------------------------------------------------------------

/// Installed on the first processor in the internal chain.
///
/// - **Downstream** — normal pass-through (the frame is entering the pipeline).
/// - **Upstream** — the frame has escaped the pipeline going left.
///   Forward it to the outer pipeline's upstream neighbour.
pub(crate) struct PipelineSourceHandler {
    outer: Arc<RwLock<Option<WeakFrameProcessor>>>,
}

#[async_trait]
impl FrameHandler for PipelineSourceHandler {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match direction {
            FrameDirection::Downstream => {
                // Normal path: send into the pipeline.
                processor.push_frame(frame, FrameDirection::Downstream).await
            }
            FrameDirection::Upstream => {
                // Frame escaped past the entry point going upstream.
                // Forward to whatever is upstream of the outer pipeline.
                let outer_opt = {
                    let g = self.outer.read().unwrap();
                    g.as_ref().and_then(|w| w.upgrade())
                };
                if let Some(outer) = outer_opt {
                    outer.push_frame(frame, FrameDirection::Upstream).await?;
                }
                Ok(())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PipelineSinkHandler
// ---------------------------------------------------------------------------

/// Installed on the last processor in the internal chain.
///
/// - **Upstream** — normal pass-through (the frame is traversing back up).
/// - **Downstream** — the frame has escaped the pipeline going right.
///   Forward it to the outer pipeline's downstream neighbour.
pub(crate) struct PipelineSinkHandler {
    outer: Arc<RwLock<Option<WeakFrameProcessor>>>,
}

#[async_trait]
impl FrameHandler for PipelineSinkHandler {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match direction {
            FrameDirection::Upstream => {
                // Normal path: propagate upstream through the pipeline.
                processor.push_frame(frame, FrameDirection::Upstream).await
            }
            FrameDirection::Downstream => {
                // Frame escaped past the exit point going downstream.
                // Forward to whatever is downstream of the outer pipeline.
                let outer_opt = {
                    let g = self.outer.read().unwrap();
                    g.as_ref().and_then(|w| w.upgrade())
                };
                if let Some(outer) = outer_opt {
                    outer.push_frame(frame, FrameDirection::Downstream).await?;
                }
                Ok(())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PipelineHandler  (installed on the outer FrameProcessor)
// ---------------------------------------------------------------------------

/// Routes incoming frames to the correct end of the internal chain.
///
/// - Downstream frames enter at `source`.
/// - Upstream frames enter at `sink`.
struct PipelineHandler {
    source: FrameProcessor,
    sink: FrameProcessor,
}

#[async_trait]
impl FrameHandler for PipelineHandler {
    async fn on_process_frame(
        &self,
        _processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match direction {
            FrameDirection::Downstream => {
                self.source
                    .queue_frame(frame, FrameDirection::Downstream, None)
                    .await
            }
            FrameDirection::Upstream => {
                self.sink
                    .queue_frame(frame, FrameDirection::Upstream, None)
                    .await
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline factory
// ---------------------------------------------------------------------------

/// Factory that assembles a list of processors into a single `FrameProcessor`.
///
/// The returned value IS a `FrameProcessor` so it can be placed inside any
/// other processor list — including another `Pipeline` or a `PipelineTask`.
///
/// ```text
/// let pipeline = Pipeline::new(vec![tts, llm, stt]);
/// // `pipeline` is a FrameProcessor — treat it like any other processor.
/// ```text
pub struct Pipeline;

impl Pipeline {
    /// Build a pipeline from an ordered list of processors.
    ///
    /// Internal chain: `[source] → processors... → [sink]`
    ///
    /// The outer `FrameProcessor` owns the chain as sub-processors, so
    /// `setup()` / `cleanup()` propagate automatically.
    pub fn new(processors: Vec<FrameProcessor>) -> FrameProcessor {
        // Shared slot for the outer processor reference.
        // Written once after the outer is constructed; read by source and sink.
        let outer_slot: Arc<RwLock<Option<WeakFrameProcessor>>> =
            Arc::new(RwLock::new(None));

        let source = FrameProcessor::new(
            "PipelineSource",
            Box::new(PipelineSourceHandler {
                outer: outer_slot.clone(),
            }),
            true, // direct mode: no task loops, processed inline
        );

        let sink = FrameProcessor::new(
            "PipelineSink",
            Box::new(PipelineSinkHandler {
                outer: outer_slot.clone(),
            }),
            true, // direct mode
        );

        // Build the full ordered list: source → …processors… → sink
        let all: Vec<FrameProcessor> = std::iter::once(source.clone())
            .chain(processors.iter().cloned())
            .chain(std::iter::once(sink.clone()))
            .collect();

        // Link sequentially: each processor knows its next and prev.
        for window in all.windows(2) {
            window[0].link(&window[1]);
        }

        // Create the outer wrapper processor.
        let outer = FrameProcessor::new(
            "Pipeline",
            Box::new(PipelineHandler {
                source: source.clone(),
                sink: sink.clone(),
            }),
            false, // has its own input/process task loops
        );

        // Register the internal chain so setup()/cleanup() propagate and
        // processors_with_metrics() can walk them.
        outer.set_sub_processors(all.clone());
        outer.set_entry_processors(vec![source.clone()]);

        // Backfill the weak outer ref into source and sink handlers.
        *outer_slot.write().unwrap() = Some(outer.downgrade());

        outer
    }
}
