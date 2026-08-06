//! Reusable STT core.
//!
//! Every speech-to-text backend in this crate is the same program with a
//! different wire format. The core owns the program; a provider supplies the
//! wire format.
//!
//! ```text
//!                 ┌─────────────────── SttService<P> ───────────────────┐
//!  InputAudioRaw ─┤ AudioFrontend → TurnGate → P::encode_audio → socket │
//!                 │        ▲                                      │     │
//!  Transcription ─┤  P::parse ── SttEvent ────────────────────────┘     │
//!                 └────────────────────────────────────────────────────┘
//! ```
//!
//! - [`SttProvider`] — the base trait a service implements: handshake, audio
//!   framing, finalize message, and message parsing. Nothing else.
//! - [`SttService`] — generic over `SttProvider`, and the only
//!   [`FrameHandler`](crate::frames::FrameHandler) in the STT subsystem. Owns
//!   the WebSocket tasks, the turn gate, the audio front-end and billing.
//! - [`TurnGate`] — audio gating, pre-roll, stashed VAD stop, duration ledger.
//!   Guarantees the transcript-before-stop ordering that
//!   [`LLMUserAggregator`](crate::processors::llm_user_aggregator::LLMUserAggregator)
//!   depends on.
//! - [`AudioFrontend`] — resample → high-pass → denoise → AGC + limiter.
//!
//! ## Adding a provider
//!
//! ```ignore
//! struct MyProvider { cfg: MyConfig }
//!
//! impl SttProvider for MyProvider {
//!     fn name(&self) -> &'static str { "myprovider" }
//!     fn audio(&self) -> &AudioSpec { &self.cfg.audio }
//!     fn handshake(&self) -> Handshake {
//!         Handshake::new(self.cfg.url()).header("Authorization", &self.cfg.key)
//!     }
//!     fn encode_audio(&self, pcm_le: &[u8]) -> Outgoing {
//!         Outgoing::Binary(pcm_le.to_vec())
//!     }
//!     fn finalize_msg(&self) -> Option<Outgoing> {
//!         Some(Outgoing::Text(r#"{"type":"Finalize"}"#.into()))
//!     }
//!     fn parse(&self, msg: WsMessage<'_>) -> SttEvent { /* … */ }
//! }
//!
//! let stt = SttService::new(MyProvider { cfg }, SttCoreConfig::default())
//!     .into_processor();
//! ```
//!
//! See `services/stt/sarvam.rs` for the worked example.

pub mod driver;
pub mod frontend;
pub mod provider;
pub mod turn_gate;
pub mod util;
pub mod ws;

pub use driver::{InterimPolicy, SttCoreConfig, SttService};
pub use frontend::{AudioFrontend, FrontendConfig, NoiseBackend};
pub use provider::{AudioSpec, Handshake, Outgoing, SttEvent, SttProvider, WsMessage};
pub use turn_gate::{TranscriptOutcome, TurnGate};
