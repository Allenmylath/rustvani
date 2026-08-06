//! The base STT trait: a *wire-protocol adapter*, nothing more.
//!
//! A provider answers four questions and owns no state machine:
//!
//!   1. Where do I connect, and with what headers?  ([`SttProvider::handshake`])
//!   2. How do I frame a chunk of PCM for this API?  ([`SttProvider::encode_audio`])
//!   3. How do I ask for a final transcript?         ([`SttProvider::finalize_msg`])
//!   4. What does this server message mean?          ([`SttProvider::parse`])
//!
//! Everything else — the WebSocket tasks, the turn gate, audio enhancement,
//! billing, and the single [`FrameHandler`](crate::frames::FrameHandler) impl —
//! lives in [`SttService`](super::driver::SttService), which is generic over
//! this trait. Adding a provider is therefore a protocol exercise, not a
//! concurrency one.
//!
//! Provider-specific configuration (Sarvam's `mode`, Deepgram's `endpointing`,
//! 60db's language list) stays in the provider's own config struct. None of it
//! belongs here.

use std::time::Duration;

// ---------------------------------------------------------------------------
// Vocabulary types
// ---------------------------------------------------------------------------

/// The PCM format a provider wants on the wire.
///
/// [`AudioFrontend`](super::frontend::AudioFrontend) resamples incoming audio
/// to `sample_rate` before the provider ever sees it, so a provider never has
/// to care what rate the transport happens to run at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioSpec {
    /// Sample rate in Hz the provider expects.
    pub sample_rate: u32,
    /// Channel count. Everything in the pipeline is mono today.
    pub channels: u16,
    /// Provider-specific encoding label (e.g. `"wav"`, `"linear16"`, `"mulaw"`).
    /// The core always hands over PCM i16 LE; this is for the provider's own
    /// URL params / JSON fields.
    pub encoding: String,
}

impl AudioSpec {
    pub fn new(sample_rate: u32, encoding: impl Into<String>) -> Self {
        Self { sample_rate, channels: 1, encoding: encoding.into() }
    }
}

/// One message to put on the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outgoing {
    Text(String),
    Binary(Vec<u8>),
}

/// One message off the wire, borrowed for the duration of [`SttProvider::parse`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WsMessage<'a> {
    Text(&'a str),
    Binary(&'a [u8]),
}

/// Connection parameters. `headers` excludes the WebSocket upgrade headers —
/// [`ws::connect`](super::ws::connect) adds those.
#[derive(Debug, Clone)]
pub struct Handshake {
    pub url: String,
    pub headers: Vec<(String, String)>,
}

impl Handshake {
    pub fn new(url: impl Into<String>) -> Self {
        Self { url: url.into(), headers: Vec::new() }
    }

    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }
}

/// A server message, normalised across providers.
///
/// The distinction that matters to the core is `Final`/`EmptyFinal` versus
/// everything else: only those two claim the turn gate's stashed
/// `VADUserStoppedSpeaking` and close the turn.
#[derive(Debug, Clone, PartialEq)]
pub enum SttEvent {
    /// Non-final hypothesis. Disposition is decided by
    /// [`InterimPolicy`](super::driver::InterimPolicy).
    Partial {
        text: String,
        language: Option<String>,
    },
    /// Final transcript for a segment.
    ///
    /// `audio_ms` is the provider-reported duration of audio this transcript
    /// consumed (Sarvam's `metrics.audio_duration`). It drives turn attribution
    /// and is the preferred billing source; `None` falls back to the gate's own
    /// ledger accounting.
    Final {
        text: String,
        language: Option<String>,
        audio_ms: Option<f64>,
    },
    /// The provider answered our finalize with no text. This still closes the
    /// turn — otherwise the gate would sit on the stop until its timeout.
    EmptyFinal { audio_ms: Option<f64> },
    /// Server-side VAD said speech began. Advisory only: local VAD owns turns.
    SpeechStarted,
    /// Server-side VAD said speech ended. Advisory only.
    SpeechEnded,
    /// Provider reported an error. Surfaced as a non-fatal `ErrorFrame`.
    Error(String),
    /// Keepalive acks, metadata, anything the core should not act on.
    Ignore,
}

// ---------------------------------------------------------------------------
// The trait
// ---------------------------------------------------------------------------

/// A speech-to-text wire protocol.
///
/// Implementors are shared across the pipeline task and the WebSocket receive
/// task, hence `Send + Sync + 'static`. All methods take `&self`: a provider
/// holds configuration, not mutable session state. Session state belongs to
/// [`SttService`](super::driver::SttService).
pub trait SttProvider: Send + Sync + 'static {
    /// Short lowercase identifier, used as the `provider` field on
    /// [`BillingEvent::SttUsage`](crate::billing::BillingEvent::SttUsage) and
    /// as the log prefix.
    fn name(&self) -> &'static str;

    /// PCM format this provider wants. The front-end resamples to it.
    fn audio(&self) -> &AudioSpec;

    /// Where to connect and with what headers.
    fn handshake(&self) -> Handshake;

    /// Messages to send immediately after the socket opens — for providers
    /// that configure the session in-band rather than in the URL.
    fn on_connected(&self) -> Vec<Outgoing> {
        Vec::new()
    }

    /// Wrap a chunk of PCM i16 LE (already at [`Self::audio`] rate) for the wire.
    fn encode_audio(&self, pcm_le: &[u8]) -> Outgoing;

    /// Ask the server to finalise the current segment. Sent when local VAD
    /// reports end of turn. `None` for providers that only finalise on their
    /// own endpointing.
    fn finalize_msg(&self) -> Option<Outgoing> {
        None
    }

    /// Graceful close message, sent before the socket is torn down.
    fn close_msg(&self) -> Option<Outgoing> {
        None
    }

    /// Idle keepalive: how often, and what to send. `None` disables it.
    ///
    /// Relevant even for providers that stream continuously — with
    /// [`audio_gating`](super::driver::SttCoreConfig::audio_gating) on, a quiet
    /// caller means literally zero bytes on the socket between turns.
    fn keepalive(&self) -> Option<(Duration, Outgoing)> {
        None
    }

    /// Interpret one server message.
    fn parse(&self, msg: WsMessage<'_>) -> SttEvent;
}
