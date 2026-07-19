//! Frame serializers — convert [`Frame`]s to/from provider wire formats for
//! WebSocket transports.
//!
//! Ported from pipecat's `pipecat.serializers` subsystem. A serializer sits
//! between [`WebSocketTransport`](crate::transport) and an external telephony
//! provider: outgoing frames become provider messages via [`serialize`], and
//! incoming provider messages become frames via [`deserialize`].
//!
//! [`serialize`]: FrameSerializer::serialize
//! [`deserialize`]: FrameSerializer::deserialize

pub mod g711;
pub mod twilio;

pub use twilio::{TwilioFrameSerializer, TwilioInputParams, TwilioStart};

use async_trait::async_trait;

use crate::frames::Frame;

// ---------------------------------------------------------------------------
// Serialized payloads
// ---------------------------------------------------------------------------

/// A serialized frame ready to send to the provider.
///
/// Mirrors pipecat's `str | bytes` serialize return: JSON-protocol providers
/// (Twilio, Plivo) yield [`Text`](SerializedOutput::Text); raw-binary providers
/// (Vonage) yield [`Binary`](SerializedOutput::Binary).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SerializedOutput {
    Text(String),
    Binary(Vec<u8>),
}

/// A raw message received from the provider, awaiting deserialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SerializedInput {
    Text(String),
    Binary(Vec<u8>),
}

// ---------------------------------------------------------------------------
// FrameSerializer trait
// ---------------------------------------------------------------------------

/// Converts frames to/from a provider's wire format.
///
/// Mirrors pipecat's abstract `FrameSerializer` (`base_serializer.py`). Unlike
/// pipecat, [`setup`](FrameSerializer::setup) receives the pipeline sample rates
/// directly rather than a `StartFrame`, since rustvani transports already hold
/// them in [`TransportParams`](crate::transport::TransportParams).
#[async_trait]
pub trait FrameSerializer: Send {
    /// Initialize with the pipeline's input/output sample rates. Called once
    /// before the first (de)serialize.
    async fn setup(&mut self, audio_in_sample_rate: u32, audio_out_sample_rate: u32);

    /// Convert an outgoing frame to a provider message, or `None` if the frame
    /// is not handled / produced no output.
    async fn serialize(&mut self, frame: &Frame) -> Option<SerializedOutput>;

    /// Convert an incoming provider message to a frame, or `None` if the
    /// message is not handled.
    async fn deserialize(&mut self, data: &SerializedInput) -> Option<Frame>;
}
