//! Base transport.
//!
//! `BaseTransport` owns a paired `BaseInputTransport` and `BaseOutputTransport`,
//! each installed on its own `FrameProcessor`. Concrete transports (WebRTC,
//! WebSocket, etc.) embed this and expose `input()` / `output()` to the pipeline.
//!
//! # Usage
//!
//! ```text
//! let transport = BaseTransport::new("WebRtc", params);
//!
//! // Wire into a pipeline.
//! let pipeline = Pipeline::new(vec![
//!     transport.input(),
//!     stt,
//!     llm,
//!     tts,
//!     transport.output(),
//! ]);
//!
//! // Push audio from the network thread.
//! transport.push_audio_frame(audio_data).await;
//! ```

use crate::frames::{FrameProcessor, FrameHandler};
use super::params::TransportParams;
use super::input::BaseInputTransport;
use super::output::BaseOutputTransport;

/// Wraps a paired input + output transport as two `FrameProcessor`s.
pub struct BaseTransport {
    input_processor:  FrameProcessor,
    output_processor: FrameProcessor,
    input_transport:  std::sync::Arc<BaseInputTransport>,
}

impl BaseTransport {
    /// Create a new base transport with the given name and params.
    ///
    /// The name is used as a prefix for the internal processor names
    /// (e.g. `"WebRtcInput"`, `"WebRtcOutput"`).
    pub fn new(name: &str, params: TransportParams) -> Self {
        let input_transport = std::sync::Arc::new(
            BaseInputTransport::new(params.clone())
        );

        // We need a FrameHandler that delegates to our Arc<BaseInputTransport>.
        // We wrap it in a thin newtype so we can pass Box<dyn FrameHandler>.
        let input_handler = InputHandlerWrapper(input_transport.clone());
        let output_handler = BaseOutputTransport::new(params);

        let input_processor = FrameProcessor::new(
            format!("{}Input", name),
            Box::new(input_handler),
            false,
        );

        let output_processor = FrameProcessor::new(
            format!("{}Output", name),
            Box::new(output_handler),
            false,
        );

        Self {
            input_processor,
            output_processor,
            input_transport,
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

    /// Push a raw audio chunk into the input transport's audio queue.
    ///
    /// Call this from your network/device callback.
    pub async fn push_audio_frame(
        &self,
        data: crate::frames::AudioRawData,
    ) -> bool {
        self.input_transport.push_audio_frame(data).await
    }

    /// Clone the audio sender for transports that need to own it
    /// (e.g. move into a network callback closure).
    pub fn audio_sender(&self) -> tokio::sync::mpsc::Sender<crate::frames::AudioRawData> {
        self.input_transport.audio_sender()
    }
}

// ---------------------------------------------------------------------------
// Thin wrapper so Arc<BaseInputTransport> satisfies Box<dyn FrameHandler>
// ---------------------------------------------------------------------------

struct InputHandlerWrapper(std::sync::Arc<BaseInputTransport>);

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