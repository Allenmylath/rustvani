//! `SttService<P>` — the one `FrameHandler` every STT provider runs on.
//!
//! Owns the WebSocket session, the [`TurnGate`], the [`AudioFrontend`] and the
//! billing hook. A [`SttProvider`] contributes only its wire protocol.
//!
//! Frames consumed:
//!   - `StartFrame`             → connects WebSocket, resets gate
//!   - `InputAudioRaw`          → front-end → gate (send now / pre-roll buffer)
//!   - `VADUserStartedSpeaking` → drops any pending stop (barge-in), bumps
//!     epoch, forwards frame, sends pre-roll
//!   - `VADUserStoppedSpeaking` → CONSUMED: stashed in the gate; front-end
//!     tail + finalize sent; release timeout armed
//!   - `EndFrame` / `CancelFrame` → resets gate, disconnects, forwards
//!
//! Frames produced:
//!   - `TranscriptionFrame` (downstream) for mid-turn and interim transcripts
//!   - `VADUserStoppedSpeaking` (downstream, transcript bundled on) when the
//!     gate releases it
//!   - `ErrorFrame` (upstream) on connection / provider errors

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::Message;

use crate::audio_process::agc::AgcConfig;
use crate::billing::{BillingCollector, BillingEvent};
use crate::error::Result;
use crate::frames::{
    ControlFrame, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor, SystemFrame,
    TranscriptionData,
};

use super::frontend::{AudioFrontend, FrontendConfig, NoiseBackend};
use super::provider::{Outgoing, SttEvent, SttProvider, WsMessage};
use super::turn_gate::TurnGate;
use super::util::{bytes_to_i16, i16_to_bytes, timestamp};
use super::ws;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// What the core does with a provider's non-final hypotheses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterimPolicy {
    /// Discard them. The default — the LLM turn only ever sees final text.
    #[default]
    Drop,
    /// Push them downstream as `TranscriptionFrame`s with `finalized: false`,
    /// for live-caption UI. Safe because
    /// [`LLMUserAggregator`](crate::processors::llm_user_aggregator::LLMUserAggregator)
    /// only aggregates finalized transcripts.
    Emit,
}

/// Provider-independent behaviour of the STT stage.
#[derive(Debug, Clone)]
pub struct SttCoreConfig {
    /// Rate the provider is configured for. The front-end resamples to it.
    pub sample_rate: u32,

    /// Enable noise suppression before sending audio. Default: true.
    pub noise_reduction: bool,
    /// Which suppressor to use. Default: [`NoiseBackend::Rnnoise`].
    pub noise_backend: NoiseBackend,
    /// Enable high-pass (before the denoiser) plus AGC and soft limiter
    /// (after it), so the provider receives consistently-levelled, clip-free
    /// audio. Default: true.
    pub agc: bool,
    /// AGC tuning. Ignored when `agc` is false.
    pub agc_config: AgcConfig,

    /// Forward audio only during local-VAD-attested turns (plus pre-roll).
    /// Eliminates spurious server-VAD transcripts by construction and cuts
    /// STT cost. Default: true.
    ///
    /// When false, the legacy continuous-streaming behaviour is used — note
    /// that spurious transcripts then become possible again, and the
    /// aggregator's late-transcript policy should be set to `Discard`.
    pub audio_gating: bool,

    /// How much audio (ms) to retain while the user is NOT speaking, sent as
    /// pre-roll when local VAD confirms speech. Covers VAD detection latency
    /// so the first syllable isn't clipped. Default: 500.
    pub pre_roll_ms: u32,

    /// How long (ms) to hold a gated `VADUserStoppedSpeaking` waiting for the
    /// provider's transcript before releasing it anyway. Default: 1200.
    pub stop_release_timeout_ms: u64,

    /// Disposition of non-final transcripts. Default: [`InterimPolicy::Drop`].
    pub interim_policy: InterimPolicy,
}

impl Default for SttCoreConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            noise_reduction: true,
            noise_backend: NoiseBackend::default(),
            agc: true,
            agc_config: AgcConfig::default(),
            audio_gating: true,
            pre_roll_ms: 500,
            stop_release_timeout_ms: 1_200,
            interim_policy: InterimPolicy::default(),
        }
    }
}

impl SttCoreConfig {
    fn frontend(&self) -> FrontendConfig {
        FrontendConfig {
            target_sample_rate: self.sample_rate,
            noise_reduction: self.noise_reduction,
            noise_backend: self.noise_backend,
            agc: self.agc,
            agc_config: self.agc_config.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Session state
// ---------------------------------------------------------------------------

struct Session {
    ws_tx: Option<mpsc::Sender<Outgoing>>,
    send_task: Option<JoinHandle<()>>,
    receive_task: Option<JoinHandle<()>>,
    keepalive_task: Option<JoinHandle<()>>,
}

impl Session {
    fn new() -> Self {
        Self { ws_tx: None, send_task: None, receive_task: None, keepalive_task: None }
    }
}

/// Everything the receive task needs.
struct RxCtx<P: SttProvider> {
    provider: Arc<P>,
    processor: FrameProcessor,
    gate: Arc<TurnGate>,
    frontend: Arc<Mutex<AudioFrontend>>,
    billing: Option<Arc<dyn BillingCollector>>,
    interim_policy: InterimPolicy,
}

// ---------------------------------------------------------------------------
// SttService
// ---------------------------------------------------------------------------

/// The generic STT frame handler. Wrap a [`SttProvider`] in one of these and
/// call [`into_processor`](Self::into_processor).
pub struct SttService<P: SttProvider> {
    provider: Arc<P>,
    config: SttCoreConfig,
    session: Arc<Mutex<Session>>,
    frontend: Arc<Mutex<AudioFrontend>>,
    gate: Arc<TurnGate>,
    billing: Option<Arc<dyn BillingCollector>>,
}

impl<P: SttProvider> SttService<P> {
    pub fn new(provider: P, config: SttCoreConfig) -> Self {
        let name = provider.name();

        if config.sample_rate != provider.audio().sample_rate {
            log::warn!(
                "{}: core sample_rate ({}) differs from the provider's ({}) — \
                 the front-end will emit at the core rate",
                name,
                config.sample_rate,
                provider.audio().sample_rate,
            );
        }

        if config.audio_gating {
            log::info!(
                "{}: turn-gated audio enabled (pre_roll={}ms, stop_release_timeout={}ms)",
                name, config.pre_roll_ms, config.stop_release_timeout_ms
            );
        } else {
            log::warn!(
                "{}: audio gating DISABLED — continuous streaming; \
                 spurious server-VAD transcripts are possible",
                name
            );
        }

        let gate = TurnGate::new(config.sample_rate, config.pre_roll_ms);
        let frontend = Arc::new(Mutex::new(AudioFrontend::new(&config.frontend())));

        Self {
            provider: Arc::new(provider),
            config,
            session: Arc::new(Mutex::new(Session::new())),
            frontend,
            gate,
            billing: None,
        }
    }

    pub fn with_billing(mut self, billing: Arc<dyn BillingCollector>) -> Self {
        self.billing = Some(billing);
        self
    }

    pub fn into_processor(self) -> FrameProcessor {
        let name = self.provider.name();
        FrameProcessor::new(name, Box::new(self), false)
    }

    // ---- connection ------------------------------------------------------

    async fn connect(&self, processor: FrameProcessor) {
        let name = self.provider.name();
        let handshake = self.provider.handshake();
        log::info!("{}: connecting to {}", name, handshake.url);

        let (sink, stream) = match ws::connect(&handshake).await {
            Ok(pair) => pair,
            Err(e) => {
                let _ = processor.push_error(format!("{}: {}", name, e), false).await;
                return;
            }
        };

        let (ws_tx, ws_rx) = mpsc::channel::<Outgoing>(128);
        let send_task = tokio::spawn(ws::run_send_task(sink, ws_rx, name));

        let ctx = Arc::new(RxCtx {
            provider: self.provider.clone(),
            processor,
            gate: self.gate.clone(),
            frontend: self.frontend.clone(),
            billing: self.billing.clone(),
            interim_policy: self.config.interim_policy,
        });
        let receive_task = tokio::spawn(run_receive_task(stream, ctx));

        let keepalive_task = self.provider.keepalive().map(|(interval, msg)| {
            tokio::spawn(ws::run_keepalive_task(ws_tx.clone(), interval, msg, name))
        });

        {
            let mut session = self.session.lock().await;
            session.ws_tx = Some(ws_tx);
            session.send_task = Some(send_task);
            session.receive_task = Some(receive_task);
            session.keepalive_task = keepalive_task;
        }

        // In-band session configuration, for providers that need it.
        for msg in self.provider.on_connected() {
            self.send(msg).await;
        }

        log::info!("{}: connected", name);
    }

    async fn disconnect(&self) {
        if let Some(msg) = self.provider.close_msg() {
            self.send(msg).await;
        }

        let mut session = self.session.lock().await;
        if let Some(h) = session.keepalive_task.take() { h.abort(); }
        if let Some(h) = session.receive_task.take()   { h.abort(); }
        if let Some(h) = session.send_task.take()      { h.abort(); }
        session.ws_tx = None;

        log::info!("{}: disconnected", self.provider.name());
    }

    async fn send(&self, msg: Outgoing) {
        let tx = { self.session.lock().await.ws_tx.clone() };
        if let Some(tx) = tx {
            let _ = tx.send(msg).await;
        }
    }

    /// Frame and send PCM samples through the provider's encoder.
    async fn send_audio(&self, samples: &[i16]) {
        if samples.is_empty() {
            return;
        }
        let encoded = self.provider.encode_audio(&i16_to_bytes(samples));
        self.send(encoded).await;
    }
}

// ---------------------------------------------------------------------------
// FrameHandler
// ---------------------------------------------------------------------------

#[async_trait]
impl<P: SttProvider> FrameHandler for SttService<P> {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match &frame.inner {
            FrameInner::System(SystemFrame::Start(_)) => {
                self.gate.reset();
                self.frontend.lock().await.reset();
                processor.push_frame(frame, direction).await?;
                self.connect(processor.clone()).await;
            }

            FrameInner::System(SystemFrame::InputAudioRaw(ref audio)) => {
                processor.push_frame(frame.clone(), direction).await?;

                let pcm = bytes_to_i16(&audio.audio);
                let samples = self.frontend.lock().await.process(&pcm, audio.sample_rate);
                if samples.is_empty() {
                    // Front-end is buffering — nothing to admit yet.
                    return Ok(());
                }

                if self.gate.admit_audio(&samples, self.config.audio_gating) {
                    self.send_audio(&samples).await;
                }
                // else: buffered into the pre-roll ring; sent on VadStart.
            }

            FrameInner::System(SystemFrame::VADUserStartedSpeaking { .. }) => {
                // Forwards the frame downstream (under the emit lock) and
                // returns the pre-roll captured during VAD detection latency.
                let pre_roll = self.gate.on_vad_start(processor, frame, direction).await?;
                if !pre_roll.is_empty() {
                    log::debug!(
                        "{}: sending {:.0}ms pre-roll",
                        self.provider.name(),
                        self.gate.ms_of(pre_roll.len())
                    );
                    self.send_audio(&pre_roll).await;
                }
            }

            FrameInner::System(SystemFrame::VADUserStoppedSpeaking { .. }) => {
                // CONSUMED here — the gate releases it downstream after the
                // transcript arrives (or the timeout fires). Never forwarded
                // directly: that is the whole ordering guarantee.

                // Flush the front-end tail and send it before the finalize.
                let tail = self.frontend.lock().await.flush();
                let tail_ms = self.gate.ms_of(tail.len());
                self.send_audio(&tail).await;

                let gen = self.gate.on_vad_stop(frame, tail_ms);

                if let Some(msg) = self.provider.finalize_msg() {
                    self.send(msg).await;
                }

                let gate = self.gate.clone();
                let proc = processor.clone();
                let after = Duration::from_millis(self.config.stop_release_timeout_ms);
                tokio::spawn(gate.release_pending_after(proc, gen, after));
            }

            FrameInner::Control(ControlFrame::End { .. })
            | FrameInner::System(SystemFrame::Cancel { .. }) => {
                self.gate.reset();
                self.disconnect().await;
                processor.push_frame(frame, direction).await?;
            }

            _ => {
                processor.push_frame(frame, direction).await?;
            }
        }
        Ok(())
    }

    fn can_generate_metrics(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Receive task
// ---------------------------------------------------------------------------

async fn run_receive_task<P: SttProvider>(mut stream: ws::WsStream, ctx: Arc<RxCtx<P>>) {
    use futures::StreamExt;

    let name = ctx.provider.name();
    log::debug!("{}: receive task started", name);

    while let Some(result) = stream.next().await {
        match result {
            Ok(Message::Text(text)) => {
                handle_message(WsMessage::Text(text.as_str()), &ctx).await;
            }
            Ok(Message::Binary(bytes)) => {
                handle_message(WsMessage::Binary(&bytes), &ctx).await;
            }
            Ok(Message::Close(_)) => {
                log::info!("{}: server closed WebSocket", name);
                break;
            }
            Err(e) => {
                let _ = ctx
                    .processor
                    .push_error(format!("{}: receive error: {}", name, e), false)
                    .await;
                break;
            }
            _ => {}
        }
    }

    log::debug!("{}: receive task exited", name);
}

async fn handle_message<P: SttProvider>(msg: WsMessage<'_>, ctx: &Arc<RxCtx<P>>) {
    let name = ctx.provider.name();

    if let WsMessage::Text(t) = msg {
        log::debug!("{}: raw message: {}", name, t);
    }

    match ctx.provider.parse(msg) {
        SttEvent::Final { text, language, audio_ms } => {
            let text = text.trim().to_string();
            if text.is_empty() {
                // Whitespace-only counts as an empty answer to our finalize:
                // it must still claim the pending stop so the turn closes
                // promptly instead of waiting for the release timeout.
                finalize(None, language, audio_ms, ctx).await;
            } else {
                finalize(Some(text), language, audio_ms, ctx).await;
            }
        }

        SttEvent::EmptyFinal { audio_ms } => {
            finalize(None, None, audio_ms, ctx).await;
        }

        SttEvent::Partial { text, language } => {
            if ctx.interim_policy == InterimPolicy::Emit {
                let text = text.trim();
                if !text.is_empty() {
                    let mut td = TranscriptionData::new(text, "", timestamp());
                    td.language = language;
                    td.finalized = false;
                    let _ = ctx
                        .processor
                        .push_frame(Frame::transcription(td), FrameDirection::Downstream)
                        .await;
                }
            }
        }

        // Server-side VAD is advisory: local VAD owns turn boundaries.
        SttEvent::SpeechStarted => log::debug!("{}: server VAD start", name),
        SttEvent::SpeechEnded => log::debug!("{}: server VAD end", name),

        SttEvent::Error(e) => {
            log::warn!("{}: server error: {}", name, e);
            let _ = ctx
                .processor
                .push_error(format!("{}: server error: {}", name, e), false)
                .await;
        }

        SttEvent::Ignore => {}
    }
}

/// Push a final transcript through the gate, bill it, and reset the front-end.
async fn finalize<P: SttProvider>(
    text: Option<String>,
    language: Option<String>,
    audio_ms: Option<f64>,
    ctx: &Arc<RxCtx<P>>,
) {
    let name = ctx.provider.name();

    let data = text.as_ref().map(|txt| {
        let mut td = TranscriptionData::new(txt.clone(), "", timestamp());
        td.language = language.clone();
        td.finalized = true;
        td
    });

    let outcome = match ctx.gate.on_transcript(&ctx.processor, data, audio_ms).await {
        Ok(o) => o,
        Err(e) => {
            log::error!("{}: gate emission failed: {}", name, e);
            return;
        }
    };

    log::info!(
        "{}: transcript='{}' lang={:?} epoch={:?} dur={:.0}ms released_stop={}",
        name,
        text.as_deref().unwrap_or("<empty>"),
        language,
        outcome.father_epoch,
        outcome.billed_ms,
        outcome.released_stop,
    );

    // Prefer the provider-reported duration (what it will bill); fall back to
    // the gate's ledger estimate for providers without duration metrics.
    if outcome.billed_ms > 0.0 {
        if let Some(bc) = &ctx.billing {
            bc.record(BillingEvent::SttUsage {
                session_id: bc.session_id(),
                provider: name.to_string(),
                audio_duration_ms: outcome.billed_ms,
                occurred_at: Utc::now(),
            });
        }
    }

    // Start the next utterance from clean filter state.
    ctx.frontend.lock().await.reset();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::billing::NoopBillingCollector;
    use crate::frames::{DataFrame, PassthroughHandler};
    use std::sync::Mutex as StdMutex;

    /// A provider that returns scripted events, so the driver's turn handling
    /// can be exercised without a socket.
    struct MockProvider {
        audio: super::super::provider::AudioSpec,
    }

    impl MockProvider {
        fn new() -> Self {
            Self { audio: super::super::provider::AudioSpec::new(16_000, "linear16") }
        }
    }

    impl SttProvider for MockProvider {
        fn name(&self) -> &'static str {
            "MockStt"
        }
        fn audio(&self) -> &super::super::provider::AudioSpec {
            &self.audio
        }
        fn handshake(&self) -> super::super::provider::Handshake {
            super::super::provider::Handshake::new("wss://mock.invalid/stt")
        }
        fn encode_audio(&self, pcm_le: &[u8]) -> Outgoing {
            Outgoing::Binary(pcm_le.to_vec())
        }
        fn finalize_msg(&self) -> Option<Outgoing> {
            Some(Outgoing::Text("finalize".into()))
        }

        /// Scripted wire format so tests drive the real `handle_message` path:
        /// `partial:<text>`, `final:<text>`, `empty`, `error:<msg>`.
        fn parse(&self, msg: WsMessage<'_>) -> SttEvent {
            let WsMessage::Text(t) = msg else {
                return SttEvent::Ignore;
            };
            match t.split_once(':') {
                Some(("partial", text)) => SttEvent::Partial {
                    text: text.to_string(),
                    language: Some("en-IN".into()),
                },
                Some(("final", text)) => SttEvent::Final {
                    text: text.to_string(),
                    language: Some("en-IN".into()),
                    audio_ms: Some(500.0),
                },
                Some(("error", m)) => SttEvent::Error(m.to_string()),
                _ if t == "empty" => SttEvent::EmptyFinal { audio_ms: Some(40.0) },
                _ => SttEvent::Ignore,
            }
        }
    }

    /// A started processor — `push_frame` is a no-op until a StartFrame has
    /// been through, so every test needs this.
    async fn started_proc() -> FrameProcessor {
        let proc = FrameProcessor::new("test", Box::new(PassthroughHandler), false);
        let _ = proc
            .process_frame(
                Frame::start(crate::frames::StartFrameData::default()),
                FrameDirection::Downstream,
            )
            .await;
        proc
    }

    /// Build an `RxCtx` wired to a processor that records everything pushed
    /// downstream, plus the gate so tests can stash a stop first.
    async fn ctx(
        interim_policy: InterimPolicy,
    ) -> (Arc<RxCtx<MockProvider>>, Arc<TurnGate>, Arc<StdMutex<Vec<Frame>>>) {
        let gate = TurnGate::new(16_000, 100);

        let processor = started_proc().await;
        let captured = Arc::new(StdMutex::new(Vec::new()));
        let cap = captured.clone();
        processor.on_after_push_frame(move |f| cap.lock().unwrap().push(f.clone()));

        let frontend = Arc::new(Mutex::new(AudioFrontend::new(&FrontendConfig {
            target_sample_rate: 16_000,
            noise_reduction: false,
            agc: false,
            ..Default::default()
        })));

        let c = Arc::new(RxCtx {
            provider: Arc::new(MockProvider::new()),
            processor,
            gate: gate.clone(),
            frontend,
            billing: Some(Arc::new(NoopBillingCollector)),
            interim_policy,
        });
        (c, gate, captured)
    }

    /// Feed one wire message through the exact path the receive task uses.
    async fn recv(text: &str, c: &Arc<RxCtx<MockProvider>>) {
        handle_message(WsMessage::Text(text), c).await;
    }

    #[tokio::test]
    async fn final_transcript_releases_the_stop_with_the_text_bundled_on() {
        let (c, gate, captured) = ctx(InterimPolicy::Drop).await;
        gate.on_vad_stop(Frame::vad_user_stopped_speaking(0.0, 0.0), 0.0);

        recv("final:hello", &c).await;

        let frames = captured.lock().unwrap();
        assert_eq!(frames.len(), 1, "exactly one frame — no separate transcript");
        match &frames[0].inner {
            FrameInner::System(SystemFrame::VADUserStoppedSpeaking { transcript, .. }) => {
                let td = transcript.as_ref().expect("transcript must ride the stop");
                assert_eq!(td.text, "hello");
                assert!(td.finalized);
                assert_eq!(td.language.as_deref(), Some("en-IN"));
            }
            other => panic!("expected a bundled VadStop, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn empty_final_still_closes_the_turn() {
        let (c, gate, captured) = ctx(InterimPolicy::Drop).await;
        gate.on_vad_stop(Frame::vad_user_stopped_speaking(0.0, 0.0), 0.0);

        recv("empty", &c).await;

        assert!(
            gate.inner.lock().unwrap().pending_stop.is_none(),
            "an empty answer must still release the stop"
        );
        assert_eq!(captured.lock().unwrap().len(), 1, "the bare stop is still released");
    }

    #[tokio::test]
    async fn whitespace_only_final_is_treated_as_empty() {
        let (c, gate, captured) = ctx(InterimPolicy::Drop).await;
        gate.on_vad_stop(Frame::vad_user_stopped_speaking(0.0, 0.0), 0.0);

        recv("final:   ", &c).await;

        let frames = captured.lock().unwrap();
        assert_eq!(frames.len(), 1);
        match &frames[0].inner {
            FrameInner::System(SystemFrame::VADUserStoppedSpeaking { transcript, .. }) => {
                assert!(transcript.is_none(), "blank text must not ride the stop");
            }
            other => panic!("expected a VadStop, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn interim_is_dropped_under_the_default_policy() {
        let (c, _gate, captured) = ctx(InterimPolicy::Drop).await;
        recv("partial:hel", &c).await;
        assert!(captured.lock().unwrap().is_empty(), "Drop must emit nothing");
    }

    #[tokio::test]
    async fn interim_is_emitted_as_a_non_finalized_transcript_under_emit() {
        let (c, _gate, captured) = ctx(InterimPolicy::Emit).await;
        recv("partial:hel", &c).await;

        let frames = captured.lock().unwrap();
        assert_eq!(frames.len(), 1);
        match &frames[0].inner {
            FrameInner::Data(DataFrame::Transcription(td)) => {
                assert_eq!(td.text, "hel");
                assert!(!td.finalized, "interims must be marked non-final");
                assert_eq!(td.language.as_deref(), Some("en-IN"));
            }
            other => panic!("expected a Transcription frame, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn mid_turn_final_without_a_pending_stop_is_a_standalone_transcript() {
        let (c, _gate, captured) = ctx(InterimPolicy::Drop).await;
        // No on_vad_stop: the turn is still open.
        recv("final:hello", &c).await;

        let frames = captured.lock().unwrap();
        assert_eq!(frames.len(), 1);
        assert!(
            matches!(frames[0].inner, FrameInner::Data(DataFrame::Transcription(_))),
            "with no stashed stop the transcript stands alone"
        );
    }
}
