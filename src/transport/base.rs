//! Base transport.
//!
//! `BaseTransport` owns a paired `BaseInputTransport` and `BaseOutputTransport`,
//! each installed on its own `FrameProcessor`. Concrete transports (WebRTC,
//! WebSocket, etc.) embed this and expose `input()` / `output()` to the pipeline.
//!
//! The output transport can be wired to a concrete sink via `set_audio_out_tx()`.
//! `WebSocketTransport` calls this after construction to connect `OutputAudioRaw`
//! bytes to its socket loop.

use std::sync::Arc;

use tokio::sync::mpsc;

use crate::frames::{FrameDirection, FrameHandler, FrameProcessor};
use super::input::BaseInputTransport;
use super::output::BaseOutputTransport;
use super::params::TransportParams;

// ---------------------------------------------------------------------------
// BaseTransport
// ---------------------------------------------------------------------------

pub struct BaseTransport {
    input_processor:   FrameProcessor,
    output_processor:  FrameProcessor,
    input_transport:   Arc<BaseInputTransport>,
    /// Kept so callers can reach `set_audio_out_tx` after construction.
    output_transport:  Arc<BaseOutputTransport>,
}

impl BaseTransport {
    /// Create a new base transport with the given name and params.
    ///
    /// The name is used as a prefix for the internal processor names
    /// (e.g. `"WebRtcInput"`, `"WebRtcOutput"`).
    pub fn new(name: &str, params: TransportParams) -> Self {
        let input_transport  = Arc::new(BaseInputTransport::new(params.clone()));
        let output_transport = Arc::new(BaseOutputTransport::new(params));

        let input_processor = FrameProcessor::new(
            format!("{}Input", name),
            Box::new(InputHandlerWrapper(input_transport.clone())),
            false,
        );

        let output_processor = FrameProcessor::new(
            format!("{}Output", name),
            Box::new(OutputHandlerWrapper(output_transport.clone())),
            false,
        );

        Self {
            input_processor,
            output_processor,
            input_transport,
            output_transport,
        }
    }

    /// The input `FrameProcessor` — place first in the pipeline.
    pub fn input(&self) -> FrameProcessor {
        self.input_processor.clone()
    }

    /// The output `FrameProcessor` — place last in the pipeline.
    pub fn output(&self) -> FrameProcessor {
        self.output_processor.clone()
    }

    /// Wire up audio output to a channel.
    ///
    /// The concrete transport calls this after creating the channel so that
    /// `OutputAudioRaw` bytes are forwarded to the socket loop.
    pub fn set_audio_out_tx(&self, tx: mpsc::Sender<Vec<u8>>) {
        self.output_transport.set_audio_out_tx(tx);
    }

    /// Push a raw audio chunk into the input transport's audio queue.
    ///
    /// Call this from your network/device callback.
    pub async fn push_audio_frame(
        &self,
        data: crate::frames::AudioRawData,
    ) -> bool {
        self.input_transport.push_audio_frame(data).await
    }

    /// Clone the audio sender for transports that need to own it.
    pub fn audio_sender(&self) -> tokio::sync::mpsc::Sender<crate::frames::AudioRawData> {
        self.input_transport.audio_sender()
    }
}

// ---------------------------------------------------------------------------
// Thin wrappers: Arc<Transport> → Box<dyn FrameHandler>
// ---------------------------------------------------------------------------

struct InputHandlerWrapper(Arc<BaseInputTransport>);

#[async_trait::async_trait]
impl FrameHandler for InputHandlerWrapper {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: crate::frames::Frame,
        direction: crate::frames::FrameDirection,
    ) -> crate::error::Result<()> {
        self.0.on_process_frame(processor, frame, direction).await
    }
}

struct OutputHandlerWrapper(Arc<BaseOutputTransport>);

#[async_trait::async_trait]
impl FrameHandler for OutputHandlerWrapper {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: crate::frames::Frame,
        direction: FrameDirection,
    ) -> crate::error::Result<()> {
        self.0.on_process_frame(processor, frame, direction).await
    }
}