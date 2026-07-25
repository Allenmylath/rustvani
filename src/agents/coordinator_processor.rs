//! `CoordinatorProcessor` — a frame processor that can dispatch to other agents.
//!
//! This is the additive bridge between the two planes (see the module design
//! notes): a normal [`FrameProcessor`] you drop into a pipeline `vec![]`, which
//! — because it runs inside a [`BaseAgent`] — can reach the agent bus and
//! `dispatch` work to peer agents from inside `on_process_frame`.
//!
//! It sits in the LLM's slot. When an [`LLMContextFrame`] arrives (the same
//! trigger a real LLM fires on), it runs your closure — which decides what to
//! do, calls whatever peer agents it needs, and returns the answer text — then
//! emits the standard `LLMFullResponseStart → LLMText → LLMFullResponseEnd`
//! triple downstream. To everything after it (assistant aggregator, TTS) it is
//! indistinguishable from an LLM.
//!
//! ```ignore
//! let coordinator = CoordinatorProcessor::new("voicebot", |cx, _context| {
//!     Box::pin(async move {
//!         let w = cx.call("weather", "ask", json!({"city": "Kochi"})).await;
//!         format!("It is {w:?} today.")
//!     })
//! });
//! let task = PipelineTask::new(
//!     vec![transport.input(), stt, user_agg, coordinator.into_processor(),
//!          assistant_agg, tts, transport.output()],
//!     params,
//! );
//! // Wrap in an agent named EXACTLY "voicebot" (matches the source name above).
//! ```

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::future::BoxFuture;
use serde_json::Value;

use crate::context::LLMContext;
use crate::error::{PipecatError, Result};
use crate::frames::{
    DataFrame, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor, ProcessorBusHandle,
};

use super::task::{TaskContext, TaskResult};

/// Default time a coordinator waits for a dispatched peer to answer.
pub const DEFAULT_CALL_TIMEOUT: Duration = Duration::from_secs(90);

// ---------------------------------------------------------------------------
// CoordinatorCtx — the dispatch handle handed to the coordinator closure.
// ---------------------------------------------------------------------------

/// Passed to a [`CoordinatorProcessor`] closure: dispatch to peer agents from
/// inside the pipeline.
pub struct CoordinatorCtx {
    source: String,
    bus: Option<Arc<dyn ProcessorBusHandle>>,
}

impl CoordinatorCtx {
    /// Dispatch a task to `target`'s `on_task(task)` handler and await its
    /// terminal result (with [`DEFAULT_CALL_TIMEOUT`]).
    pub async fn call(&self, target: &str, task: &str, payload: Value) -> Result<TaskResult> {
        self.call_with_timeout(target, task, payload, DEFAULT_CALL_TIMEOUT)
            .await
    }

    /// Like [`call`](Self::call) with an explicit timeout.
    pub async fn call_with_timeout(
        &self,
        target: &str,
        task: &str,
        payload: Value,
        timeout: Duration,
    ) -> Result<TaskResult> {
        let task_ctx = self.task_ctx()?;
        let handle = task_ctx
            .dispatch(&self.source, target, Some(task.to_string()), Some(payload))
            .await?;
        handle.await_completion(Some(timeout)).await
    }

    /// Recover the `TaskContext` from the injected bus handle.
    fn task_ctx(&self) -> Result<&TaskContext> {
        let bus = self.bus.as_ref().ok_or_else(|| {
            PipecatError::pipeline(
                "CoordinatorProcessor: no bus handle — is it inside a BaseAgent pipeline?",
            )
        })?;
        bus.as_any().downcast_ref::<TaskContext>().ok_or_else(|| {
            PipecatError::pipeline("CoordinatorProcessor: bus handle is not a TaskContext")
        })
    }
}

// ---------------------------------------------------------------------------
// CoordinatorProcessor
// ---------------------------------------------------------------------------

/// User closure: given a dispatch handle and the triggering shared context,
/// produce the assistant's answer text.
pub type CoordinatorFn = Arc<
    dyn Fn(CoordinatorCtx, Arc<Mutex<LLMContext>>) -> BoxFuture<'static, String> + Send + Sync,
>;

/// A frame processor that coordinates peer agents in place of an LLM.
pub struct CoordinatorProcessor {
    agent_name: String,
    handler: CoordinatorFn,
}

impl CoordinatorProcessor {
    /// Create a coordinator. `agent_name` MUST match the [`BaseAgent`] name this
    /// pipeline belongs to — it is used as the dispatch source, so peer replies
    /// route back to this agent's `TaskContext`.
    pub fn new<F>(agent_name: impl Into<String>, handler: F) -> Self
    where
        F: Fn(CoordinatorCtx, Arc<Mutex<LLMContext>>) -> BoxFuture<'static, String>
            + Send
            + Sync
            + 'static,
    {
        Self {
            agent_name: agent_name.into(),
            handler: Arc::new(handler),
        }
    }

    /// Turn this into a `FrameProcessor` to place in a pipeline `vec![]`.
    pub fn into_processor(self) -> FrameProcessor {
        FrameProcessor::new("CoordinatorProcessor", Box::new(self), false)
    }
}

#[async_trait]
impl FrameHandler for CoordinatorProcessor {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match &frame.inner {
            // The LLM's inference trigger — run the coordinator instead.
            FrameInner::Data(DataFrame::LLMContextFrame(context)) => {
                let cx = CoordinatorCtx {
                    source: self.agent_name.clone(),
                    bus: processor.bus_handle(),
                };
                let answer = (self.handler)(cx, context.clone()).await;

                processor
                    .push_frame(Frame::llm_full_response_start(), FrameDirection::Downstream)
                    .await?;
                if !answer.trim().is_empty() {
                    processor
                        .push_frame(Frame::llm_text(answer), FrameDirection::Downstream)
                        .await?;
                }
                processor
                    .push_frame(Frame::llm_full_response_end(), FrameDirection::Downstream)
                    .await?;
            }
            _ => {
                processor.push_frame(frame, direction).await?;
            }
        }
        Ok(())
    }
}
