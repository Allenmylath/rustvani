//! Transport layer for rustvani / pipecat-rs.
//!
//! This module provides the base transport infrastructure that all concrete
//! transports (WebRTC, WebSocket, etc.) build on.
//!
//! # Structure
//!
//! - [`params::TransportParams`] — shared configuration.
//! - [`input::BaseInputTransport`] — audio input queue, VAD stub, speaking state.
//! - [`output::BaseOutputTransport`] — audio output, bot speaking signals (stub).
//!
//! # Concrete transports
//!
//! Concrete transports live in their own crates and depend on `pipecat-core`.
//! They create a `BaseInputTransport` / `BaseOutputTransport`, install them as
//! `FrameHandler`s on `FrameProcessor`s, and call `push_audio_frame()` from
//! their network callbacks.

pub mod base;
pub mod input;
pub mod output;
pub mod params;

pub use base::BaseTransport;
pub use input::BaseInputTransport;
pub use output::BaseOutputTransport;
pub use params::TransportParams;