//! Sarvam AI Speech-to-Text service — turn-gated streaming.
//!
//! Connects to Sarvam's WebSocket streaming API and pushes TranscriptionFrames
//! downstream when transcripts arrive.
//!
//! Pipeline position:
//!   transport.input() → SarvamSttHandler → LLMUserAggregator → llm → tts → out
//!
//! ── Ordering & attribution invariants (the TurnGate) ──────────────────────
//!
//! 1. AUDIO GATING (fatherhood by construction).
//!    Audio is forwarded to Sarvam only while the local VAD says the user is
//!    speaking, plus a pre-roll buffer (the ~pre_roll_ms of audio captured
//!    while the VAD was still confirming speech) and the denoiser tail.
//!    Between turns nothing is sent, so Sarvam's server-side VAD has no noise
//!    to hallucinate from: every transcript it can possibly return descends
//!    from a local-VAD-attested turn. Spurious "fatherless" transcripts are
//!    impossible by construction, not by bookkeeping.
//!
//! 2. STOP-FRAME GATING (transcript bundled onto the released stop).
//!    VADUserStoppedSpeaking is never forwarded directly. It is stashed in
//!    the gate; the WebSocket receive task releases it downstream immediately
//!    AFTER pushing the transcript it was waiting for. The transcript is
//!    bundled onto the stop frame itself so that a single frame carries both
//!    the turn boundary and the text. Separate frames cannot guarantee order
//!    across the system/data lanes, so one frame is the only structural fix.
//!
//! 3. EMISSION LINEARIZATION (the `emit` mutex).
//!    Turn frames (VadStart, Transcription, released VadStop) can originate
//!    on two tasks: the pipeline task (VadStart passthrough) and the WS
//!    receive task (transcript + released stop). All such emissions acquire
//!    the gate's async `emit` lock, and the claim/drop of the pending stop
//!    happens inside the same critical section. Every interleaving therefore
//!    collapses to one of two valid serial orders:
//!      …Transcript, VadStop, VadStart…   (turn flushed, then new turn), or
//!      …VadStart[pending dropped], Transcript…  (barge-in merge).
//!    The premature-flush and lost-text interleavings cannot occur.
//!
//! 4. EXACTLY-ONCE RELEASE (atomic claim).
//!    The stashed stop lives in a Mutex<Option<Frame>>; `take()` is the CAS.
//!    Three parties race for it — transcript arrival (release), timeout
//!    (release), barge-in VadStart (drop). Exactly one wins. Timeouts are
//!    additionally generation-guarded so a stale timer can never steal a
//!    newer turn's pending stop.
//!
//! 5. DURATION LEDGER (turn attribution + billing).
//!    The gate keeps a FIFO of (epoch, ms-of-audio-sent). Sarvam reports
//!    `metrics.audio_duration` per transcript; since the server consumes the
//!    stream in order, consuming that duration from the ledger head
//!    identifies which turn (epoch) fathered the transcript, with a small
//!    tolerance for server-side silence trimming. The server-reported
//!    duration is also the billing source of truth (it replaces the old
//!    byte-swap heuristic, which misattributed inter-utterance audio).
//!
//! Wiring:
//!   let stt = SarvamSttHandler::new(SarvamSttConfig {
//!       api_key: std::env::var("SARVAM_API_KEY").unwrap(),
//!       ..Default::default()
//!   })
//!   .into_processor();
//!
//! Frames consumed:
//!   - StartFrame             → connects WebSocket, resets gate
//!   - InputAudioRaw          → denoise → gate (send now / pre-roll buffer)
//!   - VADUserStartedSpeaking → drops any pending stop (barge-in), bumps
//!                              epoch, forwards frame, sends pre-roll
//!   - VADUserStoppedSpeaking → CONSUMED: stashed in gate; denoiser tail +
//!                              flush signal sent; release timeout armed
//!   - EndFrame / CancelFrame → resets gate, disconnects, forwards
//!
//! Frames produced:
//!   - TranscriptionFrame (downstream) on transcript
//!   - VADUserStoppedSpeaking (downstream) when the gate releases it
//!   - ErrorFrame (upstream) on connection / parse errors
//!
//! Auth: api-subscription-key header (lowercase), per SDK source.
//! URL:  wss://api.sarvam.ai/speech-to-text/ws
//! Lang: language-code param (hyphen, not underscore)

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use chrono::Utc;
use futures::{SinkExt, StreamExt};
use log;
use serde::Deserialize;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::http::Request;
use tokio_tungstenite::tungstenite::Message;

use crate::audio_process::agc::AudioEnhancer;
use crate::audio_process::noisefilter::RNNoiseFilter;
use crate::billing::{BillingCollector, BillingEvent};
use crate::error::Result;
use crate::frames::{
    ControlFrame, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor,
    SystemFrame, TranscriptionData,
};

// ---------------------------------------------------------------------------
// Constants — verified against SDK source and AsyncAPI spec
// ---------------------------------------------------------------------------

const SARVAM_BASE_WSS: &str     = "wss://api.sarvam.ai";
const STT_PATH: &str            = "/speech-to-text/ws";
const STT_TRANSLATE_PATH: &str  = "/speech-to-text-translate/ws";

// saaras:v2.5 uses the translate endpoint
const TRANSLATE_MODELS: &[&str] = &["saaras:v2.5"];
// saaras:v3 supports the mode param
const MODE_MODELS: &[&str]      = &["saaras:v3"];

/// Server-side VAD may trim leading/trailing silence, so the audio duration
/// it reports per transcript can be slightly less than what we sent. Ledger
/// consumption treats remainders at or below this as fully consumed.
const LEDGER_TOLERANCE_MS: f64 = 120.0;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for SarvamSttHandler.
///
/// Verified against SDK source (sarvamai==0.1.27):
/// - Auth goes in `api-subscription-key` header (lowercase)
/// - Language param is `language-code` (hyphen, not underscore)
/// - Base URL: wss://api.sarvam.ai
/// - Path: /speech-to-text/ws
#[derive(Debug, Clone)]
pub struct SarvamSttConfig {
    /// Sarvam API subscription key.
    pub api_key: String,

    /// Model to use.
    /// "saaras:v3"    — recommended, supports mode param
    /// "saarika:v2.5" — legacy transcription
    /// "saaras:v2.5"  — legacy translation to English
    pub model: String,

    /// BCP-47 language code e.g. "ml-IN", "hi-IN", "en-IN".
    /// "unknown" = auto-detect (where supported).
    pub language: Option<String>,

    /// Output mode — saaras:v3 only.
    /// "transcribe" (default), "translate", "verbatim", "translit", "codemix"
    pub mode: Option<String>,

    /// Audio sample rate. Must match what the transport sends.
    pub sample_rate: u32,

    /// Audio encoding. "wav" or "pcm_s16le" / "pcm_l16" / "pcm_raw".
    pub encoding: String,

    /// Enable high VAD sensitivity (shorter silence before flush).
    pub high_vad_sensitivity: bool,

    /// Receive VAD signals from server (speech_start / speech_end events).
    pub vad_signals: bool,

    /// Enable RNNoise noise suppression before sending audio to Sarvam.
    /// Default: true.
    pub noise_reduction: bool,

    /// Enable the speech enhancement chain: high-pass filter (before the
    /// denoiser) plus AGC and soft limiter (after it), so Sarvam receives
    /// consistently-levelled, clip-free audio. Default: true.
    pub agc: bool,

    /// Forward audio to Sarvam only during local-VAD-attested turns
    /// (plus pre-roll). Eliminates spurious server-VAD transcripts by
    /// construction and cuts STT cost. Default: true.
    /// When false, the legacy continuous-streaming behavior is used —
    /// note that spurious transcripts then become possible again, and the
    /// aggregator's late-transcript policy should be set to Discard.
    pub audio_gating: bool,

    /// How much audio (ms) to retain while the user is NOT speaking, sent as
    /// pre-roll when local VAD confirms speech. Covers VAD detection latency
    /// so the first syllable isn't clipped. Default: 500.
    pub pre_roll_ms: u32,

    /// How long (ms) to hold a gated VADUserStoppedSpeaking waiting for
    /// Sarvam's transcript before releasing it anyway. Default: 1200.
    pub stop_release_timeout_ms: u64,
}

impl Default for SarvamSttConfig {
    fn default() -> Self {
        Self {
            api_key:                 String::new(),
            model:                   "saaras:v3".to_string(),
            language:                Some("unknown".to_string()),
            mode:                    Some("transcribe".to_string()),
            sample_rate:             16_000,
            encoding:                "wav".to_string(),
            high_vad_sensitivity:    false,
            vad_signals:             false,
            noise_reduction:         true,
            agc:                     true,
            audio_gating:            true,
            pre_roll_ms:             500,
            stop_release_timeout_ms: 1_200,
        }
    }
}

impl SarvamSttConfig {
    fn ws_path(&self) -> &'static str {
        if TRANSLATE_MODELS.contains(&self.model.as_str()) {
            STT_TRANSLATE_PATH
        } else {
            STT_PATH
        }
    }

    fn ws_url(&self) -> String {
        let mut params = vec![
            format!("model={}", urlencoding(&self.model)),
            format!("sample_rate={}", self.sample_rate),
            format!("input_audio_codec={}", urlencoding(&self.encoding)),
            "flush_signal=true".to_string(),
        ];

        // NOTE: language param uses hyphen: language-code (not language_code)
        if let Some(lang) = &self.language {
            if !TRANSLATE_MODELS.contains(&self.model.as_str()) {
                params.push(format!("language-code={}", urlencoding(lang)));
            }
        }

        if let Some(mode) = &self.mode {
            if MODE_MODELS.contains(&self.model.as_str()) {
                params.push(format!("mode={}", urlencoding(mode)));
            }
        }

        if self.high_vad_sensitivity {
            params.push("high_vad_sensitivity=true".to_string());
        }

        if self.vad_signals {
            params.push("vad_signals=true".to_string());
        }

        format!("{}{}?{}", SARVAM_BASE_WSS, self.ws_path(), params.join("&"))
    }
}

// ---------------------------------------------------------------------------
// Sarvam WebSocket response types — per AsyncAPI spec
// ---------------------------------------------------------------------------

/// Top-level envelope. type ∈ {"data", "error", "events"}
#[derive(Debug, Deserialize)]
struct SarvamMessage {
    #[serde(rename = "type")]
    msg_type: String,
    data: Option<serde_json::Value>,
}

/// Per-transcript server metrics. `audio_duration` (seconds) is how much of
/// the audio stream this transcript consumed — the attribution & billing
/// source of truth.
#[derive(Debug, Deserialize)]
struct SarvamMetrics {
    audio_duration: Option<f64>,
    #[allow(dead_code)]
    processing_latency: Option<f64>,
}

/// Transcript payload inside a "data" message.
/// Field is "transcript" for both saarika:v2.5 and saaras:v3.
#[derive(Debug, Deserialize)]
struct SarvamTranscript {
    transcript:    Option<String>,
    language_code: Option<String>,
    request_id:    Option<String>,
    metrics:       Option<SarvamMetrics>,
}

/// VAD event payload inside an "events" message.
#[derive(Debug, Deserialize)]
struct SarvamEvent {
    signal_type: Option<String>,
}

// ---------------------------------------------------------------------------
// TurnGate — ordering, attribution, and audio gating
// ---------------------------------------------------------------------------

struct LedgerEntry {
    epoch: u64,
    ms:    f64,
}

struct GateInner {
    /// Turn counter. Bumped on every VADUserStartedSpeaking.
    epoch: u64,
    /// Local VAD state, as seen by this gate.
    speaking: bool,
    /// The stashed VADUserStoppedSpeaking frame, waiting for its transcript.
    pending_stop:  Option<Frame>,
    pending_epoch: u64,
    /// Generation counter guarding release timeouts. Bumped on every stash,
    /// claim, and drop, so a stale timer always finds a mismatch.
    timeout_gen: u64,
    /// Ring buffer of recent denoised samples captured while NOT speaking;
    /// drained and sent as pre-roll on VadStart.
    pre_roll:     VecDeque<i16>,
    pre_roll_cap: usize,
    /// FIFO of (epoch, ms of audio sent) for closed turns.
    ledger: VecDeque<LedgerEntry>,
    /// Audio ms sent so far for the currently open turn.
    current_sent_ms: f64,
}

/// Outcome of processing one Sarvam data message through the gate.
struct TranscriptOutcome {
    released_stop: bool,
    father_epoch:  Option<u64>,
    billed_ms:     f64,
}

pub(crate) struct TurnGate {
    /// Linearizes downstream emission of turn frames (VadStart, Transcript,
    /// released VadStop) across the pipeline task and the WS receive task.
    /// The pending-stop claim/drop always happens inside this critical
    /// section, which is what makes the emission order provably serial.
    emit:        Mutex<()>,
    inner:       std::sync::Mutex<GateInner>,
    sample_rate: u32,
}

impl TurnGate {
    fn new(sample_rate: u32, pre_roll_ms: u32) -> Arc<Self> {
        let cap = (sample_rate as u64 * pre_roll_ms as u64 / 1000) as usize;
        Arc::new(Self {
            emit: Mutex::new(()),
            inner: std::sync::Mutex::new(GateInner {
                epoch:           0,
                speaking:        false,
                pending_stop:    None,
                pending_epoch:   0,
                timeout_gen:     0,
                pre_roll:        VecDeque::with_capacity(cap),
                pre_roll_cap:    cap,
                ledger:          VecDeque::new(),
                current_sent_ms: 0.0,
            }),
            sample_rate,
        })
    }

    fn ms_of(&self, samples: usize) -> f64 {
        samples as f64 * 1000.0 / self.sample_rate as f64
    }

    /// Local VAD start. Atomically drops any pending stop (barge-in: the
    /// user resumed, so that turn boundary is stale), bumps the epoch,
    /// forwards the VadStart frame downstream under the emit lock, and
    /// returns the pre-roll samples the caller should send to Sarvam.
    async fn on_vad_start(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<Vec<i16>> {
        let _emit = self.emit.lock().await;

        let (dropped, epoch, pre_roll) = {
            let mut s = self.inner.lock().unwrap();
            let dropped = s.pending_stop.take().is_some();
            if dropped {
                // Disarm the timer that was guarding the dropped stop.
                s.timeout_gen = s.timeout_gen.wrapping_add(1);
            }
            s.epoch += 1;
            s.speaking = true;
            let pre: Vec<i16> = s.pre_roll.drain(..).collect();
            s.current_sent_ms = self.ms_of(pre.len());
            (dropped, s.epoch, pre)
        };

        if dropped {
            log::info!(
                "TurnGate: barge-in — pending VadStop dropped (atomic take), \
                 turn continues as epoch {}",
                epoch
            );
        } else {
            log::debug!("TurnGate: VadStart — opening epoch {}", epoch);
        }

        processor.push_frame(frame, direction).await?;
        Ok(pre_roll)
    }

    /// Local VAD stop. Finalizes the open turn's ledger entry, stashes the
    /// stop frame (NOT forwarded), and returns the timeout generation the
    /// caller should arm a release timer with. `tail_ms` is the duration of
    /// the denoiser tail the caller already sent for this turn.
    fn on_vad_stop(&self, frame: Frame, tail_ms: f64) -> u64 {
        let mut s = self.inner.lock().unwrap();
        if s.pending_stop.is_some() {
            // Should be unreachable (transport CAS prevents double-stop),
            // but never silently lose a frame boundary.
            log::warn!("TurnGate: replacing an unreleased pending VadStop");
        }
        s.speaking = false;
        s.current_sent_ms += tail_ms;
        let epoch = s.epoch;
        let ms = s.current_sent_ms;
        s.ledger.push_back(LedgerEntry { epoch, ms });
        s.current_sent_ms = 0.0;
        s.pending_stop = Some(frame);
        s.pending_epoch = epoch;
        s.timeout_gen = s.timeout_gen.wrapping_add(1);
        log::debug!(
            "TurnGate: VadStop gated for epoch {} ({:.0}ms sent), awaiting transcript",
            epoch, ms
        );
        s.timeout_gen
    }

    /// Audio admission. While speaking: account the duration and tell the
    /// caller to send. While quiet: buffer into the pre-roll ring instead.
    /// When `gated` is false (legacy continuous streaming), always send but
    /// still account the duration against the current epoch.
    fn admit_audio(&self, samples: &[i16], gated: bool) -> bool {
        let mut s = self.inner.lock().unwrap();
        if s.speaking || !gated {
            s.current_sent_ms += self.ms_of(samples.len());
            return true;
        }
        for &v in samples {
            if s.pre_roll.len() == s.pre_roll_cap {
                s.pre_roll.pop_front();
            }
            s.pre_roll.push_back(v);
        }
        false
    }

    /// A Sarvam data message arrived on the receive task. If a pending stop
    /// is present, the transcript (if any) is bundled onto it and the single
    /// combined frame is released downstream. If there is no pending stop,
    /// the transcript is pushed as a standalone data frame (mid-turn partial).
    ///
    /// `server_ms` is metrics.audio_duration × 1000 when present; it drives
    /// ledger attribution and is the preferred billing duration.
    async fn on_transcript(
        &self,
        processor: &FrameProcessor,
        data: Option<TranscriptionData>,
        server_ms: Option<f64>,
    ) -> Result<TranscriptOutcome> {
        let _emit = self.emit.lock().await;

        let (stop, father, consumed_ms) = {
            let mut s = self.inner.lock().unwrap();
            let stop = s.pending_stop.take();
            if stop.is_some() {
                // Disarm the release timer for this stash.
                s.timeout_gen = s.timeout_gen.wrapping_add(1);
            }
            let (father, consumed) = consume_ledger(&mut s, server_ms);
            (stop, father, consumed)
        };

        let released = stop.is_some();
        match stop {
            Some(stop_frame) => {
                // Closing transcript rides the stop — single frame, no lane race.
                let stop_frame = match data {
                    Some(td) => stop_frame.with_vad_stop_transcript(td),
                    None     => stop_frame, // empty answer still closes the turn
                };
                log::debug!(
                    "TurnGate: releasing VadStop (+transcript) for epoch {:?}",
                    father
                );
                processor.push_frame(stop_frame, FrameDirection::Downstream).await?;
            }
            None => {
                // Mid-turn transcript: turn still open, standalone data frame is fine.
                if let Some(td) = data {
                    processor.push_frame(Frame::transcription(td), FrameDirection::Downstream).await?;
                }
            }
        }

        Ok(TranscriptOutcome {
            released_stop: released,
            father_epoch:  father,
            billed_ms:     server_ms.unwrap_or(consumed_ms),
        })
    }

    /// Timeout fallback: release the pending stop if — and only if — it is
    /// still the same stash this timer was armed for (generation guard).
    async fn release_pending_after(
        self: Arc<Self>,
        processor: FrameProcessor,
        gen: u64,
        after: Duration,
    ) {
        tokio::time::sleep(after).await;
        let _emit = self.emit.lock().await;

        let stop = {
            let mut s = self.inner.lock().unwrap();
            if s.timeout_gen == gen {
                s.pending_stop.take()
            } else {
                None
            }
        };

        match stop {
            Some(frame) => {
                log::warn!(
                    "TurnGate: no transcript within timeout — releasing VadStop anyway"
                );
                let _ = processor
                    .push_frame(frame, FrameDirection::Downstream)
                    .await;
            }
            None => {
                log::debug!("TurnGate: release timer fired but lost the race — no-op");
            }
        }
    }

    /// Full reset on End/Cancel/Start. Any pending stop is dropped (the
    /// pipeline is going away or restarting; releasing it would be noise).
    fn reset(&self) {
        let mut s = self.inner.lock().unwrap();
        s.pending_stop = None;
        s.timeout_gen = s.timeout_gen.wrapping_add(1);
        s.speaking = false;
        s.pre_roll.clear();
        s.ledger.clear();
        s.current_sent_ms = 0.0;
    }
}

/// Consume `server_ms` of audio from the front of the ledger, returning the
/// epoch the consumption ends in (the transcript's father) and the ms
/// consumed. With `server_ms == None` (legacy models without metrics), the
/// oldest closed turn is consumed whole; if none exists, the open turn is.
fn consume_ledger(s: &mut GateInner, server_ms: Option<f64>) -> (Option<u64>, f64) {
    match server_ms {
        Some(ms) => {
            let mut remaining = ms;
            let mut father = None;
            while remaining > LEDGER_TOLERANCE_MS {
                match s.ledger.front_mut() {
                    Some(e) if e.ms > 0.0 => {
                        let take = e.ms.min(remaining);
                        e.ms -= take;
                        remaining -= take;
                        father = Some(e.epoch);
                        if e.ms <= LEDGER_TOLERANCE_MS {
                            s.ledger.pop_front();
                        }
                    }
                    Some(_) => {
                        s.ledger.pop_front();
                    }
                    None => {
                        // Mid-turn transcript: the server finalized part of
                        // the still-open turn. Charge the open epoch.
                        if s.speaking {
                            let take = s.current_sent_ms.min(remaining);
                            s.current_sent_ms -= take;
                            remaining -= take;
                            father = Some(s.epoch);
                        }
                        break;
                    }
                }
            }
            (father, ms)
        }
        None => {
            if let Some(e) = s.ledger.pop_front() {
                (Some(e.epoch), e.ms)
            } else {
                let ms = s.current_sent_ms;
                s.current_sent_ms = 0.0;
                (s.speaking.then_some(s.epoch), ms)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal connection state
// ---------------------------------------------------------------------------

struct SarvamSttState {
    ws_tx:        Option<mpsc::Sender<String>>,
    send_task:    Option<JoinHandle<()>>,
    receive_task: Option<JoinHandle<()>>,
}

impl SarvamSttState {
    fn new() -> Self {
        Self { ws_tx: None, send_task: None, receive_task: None }
    }
}

/// Context shared with the WebSocket receive task.
struct RxCtx {
    processor:         FrameProcessor,
    gate:              Arc<TurnGate>,
    language_fallback: Option<String>,
    noise_filter:      Option<Arc<Mutex<RNNoiseFilter>>>,
    enhancer:          Option<Arc<Mutex<AudioEnhancer>>>,
    billing:           Option<Arc<dyn BillingCollector>>,
}

// ---------------------------------------------------------------------------
// SarvamSttHandler
// ---------------------------------------------------------------------------

pub struct SarvamSttHandler {
    config: SarvamSttConfig,
    state:  Arc<Mutex<SarvamSttState>>,
    /// Noise filter shared with the receive task (for reset on transcript).
    noise_filter: Option<Arc<Mutex<RNNoiseFilter>>>,
    /// Speech enhancer (HPF + AGC + limiter) shared with the receive task.
    enhancer: Option<Arc<Mutex<AudioEnhancer>>>,
    /// Optional billing collector — records audio duration per transcript.
    billing: Option<Arc<dyn BillingCollector>>,
    /// Turn gate — ordering, attribution, audio gating.
    gate: Arc<TurnGate>,
}

impl SarvamSttHandler {
    pub fn new(config: SarvamSttConfig) -> Self {
        let noise_filter = if config.noise_reduction {
            log::info!(
                "SarvamStt: noise reduction enabled (sample_rate={})",
                config.sample_rate
            );
            Some(Arc::new(Mutex::new(RNNoiseFilter::new(config.sample_rate))))
        } else {
            None
        };

        let enhancer = if config.agc {
            log::info!(
                "SarvamStt: speech enhancement enabled — HPF + AGC + limiter \
                 (sample_rate={})",
                config.sample_rate
            );
            Some(Arc::new(Mutex::new(AudioEnhancer::new(config.sample_rate))))
        } else {
            None
        };

        let gate = TurnGate::new(config.sample_rate, config.pre_roll_ms);

        if config.audio_gating {
            log::info!(
                "SarvamStt: turn-gated audio enabled (pre_roll={}ms, stop_release_timeout={}ms)",
                config.pre_roll_ms, config.stop_release_timeout_ms
            );
        } else {
            log::warn!(
                "SarvamStt: audio gating DISABLED — continuous streaming; \
                 spurious server-VAD transcripts are possible"
            );
        }

        Self {
            config,
            state: Arc::new(Mutex::new(SarvamSttState::new())),
            noise_filter,
            enhancer,
            billing: None,
            gate,
        }
    }

    pub fn with_billing(mut self, billing: Arc<dyn BillingCollector>) -> Self {
        self.billing = Some(billing);
        self
    }

    pub fn into_processor(self) -> FrameProcessor {
        FrameProcessor::new("SarvamStt", Box::new(self), false)
    }
}

// ---------------------------------------------------------------------------
// Connection / disconnection
// ---------------------------------------------------------------------------

impl SarvamSttHandler {
    async fn connect(&self, processor: FrameProcessor) {
        let url = self.config.ws_url();
        log::info!("SarvamStt: connecting to {}", url);

        // Auth: api-subscription-key header (lowercase), per SDK source.
        let request = match Request::builder()
            .uri(&url)
            .header("Host", "api.sarvam.ai")
            .header("api-subscription-key", &self.config.api_key)
            .header("Connection", "Upgrade")
            .header("Upgrade", "websocket")
            .header("Sec-WebSocket-Version", "13")
            .header(
                "Sec-WebSocket-Key",
                tokio_tungstenite::tungstenite::handshake::client::generate_key(),
            )
            .body(())
        {
            Ok(r) => r,
            Err(e) => {
                let _ = processor
                    .push_error(format!("SarvamStt: request build failed: {}", e), false)
                    .await;
                return;
            }
        };

        let ws_stream = match tokio_tungstenite::connect_async(request).await {
            Ok((stream, _)) => stream,
            Err(e) => {
                let _ = processor
                    .push_error(format!("SarvamStt: connect failed: {}", e), false)
                    .await;
                return;
            }
        };

        let (sink, stream) = ws_stream.split();

        // Channel carries plain String — Message::Text wrapping happens in send_task.
        let (ws_tx, ws_rx) = mpsc::channel::<String>(64);

        let send_task = tokio::spawn(run_send_task(sink, ws_rx));

        let ctx = Arc::new(RxCtx {
            processor,
            gate:              self.gate.clone(),
            language_fallback: self.config.language.clone(),
            noise_filter:      self.noise_filter.clone(),
            enhancer:          self.enhancer.clone(),
            billing:           self.billing.clone(),
        });
        let receive_task = tokio::spawn(run_receive_task(stream, ctx));

        let mut state      = self.state.lock().await;
        state.ws_tx        = Some(ws_tx);
        state.send_task    = Some(send_task);
        state.receive_task = Some(receive_task);

        log::info!("SarvamStt: connected");
    }

    async fn disconnect(&self) {
        let mut state = self.state.lock().await;
        if let Some(h) = state.receive_task.take() { h.abort(); }
        if let Some(h) = state.send_task.take()    { h.abort(); }
        state.ws_tx = None;
        log::info!("SarvamStt: disconnected");
    }

    async fn send_json(&self, json: String) {
        let tx = { self.state.lock().await.ws_tx.clone() };
        if let Some(tx) = tx {
            let _ = tx.send(json).await;
        }
    }

    /// Send audio per AsyncAPI spec:
    /// {"audio": {"data": <base64>, "sample_rate": "<rate>", "encoding": "audio/wav"}}
    async fn send_audio(&self, audio: &[u8]) {
        if audio.is_empty() {
            return;
        }
        let msg = serde_json::json!({
            "audio": {
                "data":        BASE64.encode(audio),
                "sample_rate": self.config.sample_rate.to_string(),
                "encoding":    format!("audio/{}", self.config.encoding),
            }
        });
        self.send_json(serde_json::to_string(&msg).unwrap_or_default()).await;
    }

    /// Flush signal per AsyncAPI spec: {"type": "flush"}
    async fn send_flush(&self) {
        let msg = serde_json::json!({ "type": "flush" });
        self.send_json(serde_json::to_string(&msg).unwrap_or_default()).await;
    }

    /// Run one InputAudioRaw payload through the enhancement chain:
    /// high-pass → RNNoise → AGC + limiter. Returns the samples ready to be
    /// admitted by the gate (may be empty while the denoiser is buffering).
    async fn denoise(&self, raw: &[u8]) -> Vec<i16> {
        let mut pcm = bytes_to_i16(raw);

        // 1. DC removal + high-pass (before the denoiser).
        if let Some(enh) = &self.enhancer {
            pcm = enh.lock().await.pre_filter(&pcm);
        }

        // 2. RNNoise (may buffer — empty output is normal here).
        if let Some(nf) = &self.noise_filter {
            pcm = nf.lock().await.filter(&pcm);
        }

        // 3. AGC + soft limiter (after the denoiser).
        if !pcm.is_empty() {
            if let Some(enh) = &self.enhancer {
                pcm = enh.lock().await.post_filter(&pcm);
            }
        }

        pcm
    }
}

// ---------------------------------------------------------------------------
// Audio byte ↔ i16 helpers
// ---------------------------------------------------------------------------

fn bytes_to_i16(audio: &[u8]) -> Vec<i16> {
    audio
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn i16_to_bytes(samples: &[i16]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

// ---------------------------------------------------------------------------
// FrameHandler impl
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameHandler for SarvamSttHandler {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match &frame.inner {
            FrameInner::System(SystemFrame::Start(_)) => {
                self.gate.reset();
                processor.push_frame(frame, direction).await?;
                self.connect(processor.clone()).await;
            }

            FrameInner::System(SystemFrame::InputAudioRaw(ref audio)) => {
                processor.push_frame(frame.clone(), direction).await?;

                let samples = self.denoise(&audio.audio).await;
                if samples.is_empty() {
                    // Filter is buffering — nothing to admit yet.
                    return Ok(());
                }

                if self.gate.admit_audio(&samples, self.config.audio_gating) {
                    self.send_audio(&i16_to_bytes(&samples)).await;
                }
                // else: buffered into the pre-roll ring; sent on VadStart.
            }

            FrameInner::System(SystemFrame::VADUserStartedSpeaking { .. }) => {
                // Forwards the frame downstream (under the emit lock) and
                // returns the pre-roll captured during VAD detection latency.
                let pre_roll = self
                    .gate
                    .on_vad_start(processor, frame, direction)
                    .await?;
                if !pre_roll.is_empty() {
                    log::debug!(
                        "SarvamStt: sending {:.0}ms pre-roll",
                        self.gate.ms_of(pre_roll.len())
                    );
                    self.send_audio(&i16_to_bytes(&pre_roll)).await;
                }
            }

            FrameInner::System(SystemFrame::VADUserStoppedSpeaking { .. }) => {
                // CONSUMED here — the gate releases it downstream after the
                // transcript arrives (or the timeout fires). Never forwarded
                // directly: that is the whole ordering guarantee.

                // Flush the denoiser tail and send it before the flush signal.
                let mut tail = match &self.noise_filter {
                    Some(nf) => nf.lock().await.flush(),
                    None     => Vec::new(),
                };
                if !tail.is_empty() {
                    if let Some(enh) = &self.enhancer {
                        tail = enh.lock().await.post_filter(&tail);
                    }
                }
                let tail_ms = self.gate.ms_of(tail.len());
                if !tail.is_empty() {
                    self.send_audio(&i16_to_bytes(&tail)).await;
                }

                let gen = self.gate.on_vad_stop(frame, tail_ms);
                self.send_flush().await;

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

    fn can_generate_metrics(&self) -> bool { true }
}

// ---------------------------------------------------------------------------
// Background tasks
// ---------------------------------------------------------------------------

type WsSink = futures::stream::SplitSink<
    tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
    Message,
>;

type WsStream = futures::stream::SplitStream<
    tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
>;

async fn run_send_task(mut sink: WsSink, mut rx: mpsc::Receiver<String>) {
    while let Some(text) = rx.recv().await {
        let msg = Message::Text(text.into());
        if sink.send(msg).await.is_err() {
            log::warn!("SarvamStt: send failed — closing send task");
            break;
        }
    }
    let _ = sink.close().await;
    log::debug!("SarvamStt: send task exited");
}

async fn run_receive_task(mut stream: WsStream, ctx: Arc<RxCtx>) {
    log::debug!("SarvamStt: receive task started");

    while let Some(result) = stream.next().await {
        match result {
            Ok(Message::Text(text)) => {
                handle_message(text.as_str(), &ctx).await;
            }
            Ok(Message::Close(_)) => {
                log::info!("SarvamStt: server closed WebSocket");
                break;
            }
            Err(e) => {
                let _ = ctx
                    .processor
                    .push_error(format!("SarvamStt: receive error: {}", e), false)
                    .await;
                break;
            }
            _ => {}
        }
    }

    log::debug!("SarvamStt: receive task exited");
}

async fn handle_message(text: &str, ctx: &Arc<RxCtx>) {
    log::debug!("SarvamStt: raw message: {}", text);

    let msg: SarvamMessage = match serde_json::from_str(text) {
        Ok(m) => m,
        Err(e) => {
            log::warn!("SarvamStt: parse error: {} — raw: {}", e, text);
            return;
        }
    };

    match msg.msg_type.as_str() {
        "data" => {
            handle_transcript(msg.data, ctx).await;
        }
        "events" => {
            if let Some(data) = msg.data {
                let event: SarvamEvent = match serde_json::from_value(data) {
                    Ok(e)  => e,
                    Err(e) => { log::warn!("SarvamStt: event parse: {}", e); return; }
                };
                match event.signal_type.as_deref() {
                    Some("START_SPEECH") => log::debug!("SarvamStt: server VAD start"),
                    Some("END_SPEECH")   => log::debug!("SarvamStt: server VAD end"),
                    other                => log::debug!("SarvamStt: unknown event signal: {:?}", other),
                }
            }
        }
        "error" => {
            log::warn!("SarvamStt: server error: {:?}", msg.data);
        }
        other => {
            log::debug!("SarvamStt: unknown message type: {}", other);
        }
    }
}

async fn handle_transcript(data: Option<serde_json::Value>, ctx: &Arc<RxCtx>) {
    let data = match data {
        Some(d) => d,
        None    => return,
    };

    let t: SarvamTranscript = match serde_json::from_value(data) {
        Ok(t)  => t,
        Err(e) => { log::warn!("SarvamStt: transcript parse: {}", e); return; }
    };

    // Empty / missing text still counts as the answer to our flush: it must
    // claim and release the pending stop so the turn closes promptly instead
    // of waiting for the timeout.
    let text: Option<String> = match t.transcript {
        Some(s) if !s.trim().is_empty() => Some(s),
        _ => None,
    };

    let server_ms = t
        .metrics
        .as_ref()
        .and_then(|m| m.audio_duration)
        .map(|secs| secs * 1000.0);

    let language = t.language_code.or_else(|| ctx.language_fallback.clone());

    let frame_data = text.as_ref().map(|txt| {
        let mut fd = TranscriptionData::new(txt.clone(), "", time_now_iso8601());
        fd.language  = language.clone();
        fd.finalized = true;
        fd
    });

    let outcome = match ctx
        .gate
        .on_transcript(&ctx.processor, frame_data, server_ms)
        .await
    {
        Ok(o) => o,
        Err(e) => {
            log::error!("SarvamStt: gate emission failed: {}", e);
            return;
        }
    };

    log::info!(
        "SarvamStt: transcript='{}' lang={:?} epoch={:?} dur={:.0}ms req_id={:?} released_stop={}",
        text.as_deref().unwrap_or("<empty>"),
        language,
        outcome.father_epoch,
        outcome.billed_ms,
        t.request_id,
        outcome.released_stop,
    );

    // Billing: prefer the server-reported duration (what Sarvam will bill);
    // fall back to our ledger-consumed estimate for legacy models.
    if outcome.billed_ms > 0.0 {
        if let Some(bc) = &ctx.billing {
            bc.record(BillingEvent::SttUsage {
                session_id:        bc.session_id(),
                provider:          "sarvam".to_string(),
                audio_duration_ms: outcome.billed_ms,
                occurred_at:       Utc::now(),
            });
        }
    }

    // Reset noise filter so the next utterance starts from a clean state.
    if let Some(nf) = &ctx.noise_filter {
        nf.lock().await.reset();
    }

    // Reset the enhancer's filter state too (adapted AGC gain is kept —
    // the speaker's level rarely changes between utterances).
    if let Some(enh) = &ctx.enhancer {
        enh.lock().await.reset();
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn time_now_iso8601() -> String {
    let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    format!("{}.{:03}", d.as_secs(), d.subsec_millis())
}

fn urlencoding(s: &str) -> String {
    s.chars()
        .flat_map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => vec![c],
            ':' => vec!['%', '3', 'A'],
            '/' => vec!['%', '2', 'F'],
            _ => format!("%{:02X}", c as u32).chars().collect(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::billing::{BillingCollector, BillingEvent, NoopBillingCollector};

    fn dummy_proc() -> FrameProcessor {
        FrameProcessor::new("test", Box::new(crate::frames::PassthroughHandler), false)
    }

    fn stop_frame() -> Frame {
        Frame::vad_user_stopped_speaking(0.0, 0.0)
    }

    fn start_frame() -> Frame {
        Frame::vad_user_started_speaking(0.0, 0.0)
    }

    // ---- ms_of ---------------------------------------------------------------

    #[test]
    fn ms_of_16khz_one_second() {
        let gate = TurnGate::new(16_000, 500);
        assert!((gate.ms_of(16_000) - 1000.0).abs() < 0.001);
    }

    #[test]
    fn ms_of_8khz_half_second() {
        let gate = TurnGate::new(8_000, 500);
        assert!((gate.ms_of(4_000) - 500.0).abs() < 0.001);
    }

    // ---- pre-roll ring -------------------------------------------------------

    #[test]
    fn pre_roll_ring_is_capped() {
        // 16kHz × 100ms cap = 1600 samples
        let gate = TurnGate::new(16_000, 100);
        let chunk = vec![1i16; 800];
        for _ in 0..5 {
            let sent = gate.admit_audio(&chunk, true);
            assert!(!sent, "quiet audio must be buffered, not sent");
        }
        let s = gate.inner.lock().unwrap();
        assert_eq!(s.pre_roll.len(), 1600, "ring must be capped at pre_roll_cap");
    }

    #[test]
    fn admit_audio_sends_while_speaking_and_accounts_duration() {
        let gate = TurnGate::new(16_000, 100);
        {
            let mut s = gate.inner.lock().unwrap();
            s.speaking = true;
        }
        let chunk = vec![0i16; 16_000]; // 1000ms
        assert!(gate.admit_audio(&chunk, true));
        let s = gate.inner.lock().unwrap();
        assert!((s.current_sent_ms - 1000.0).abs() < 0.001);
    }

    #[test]
    fn admit_audio_ungated_always_sends() {
        let gate = TurnGate::new(16_000, 100);
        let chunk = vec![0i16; 1600];
        assert!(gate.admit_audio(&chunk, false), "ungated mode must always send");
    }

    // ---- stash / claim / drop -------------------------------------------------

    #[tokio::test]
    async fn vad_start_drops_pending_stop_and_bumps_epoch() {
        let gate = TurnGate::new(16_000, 100);
        let gen = gate.on_vad_stop(stop_frame(), 0.0);
        assert!(gate.inner.lock().unwrap().pending_stop.is_some());

        let pre = gate
            .on_vad_start(&dummy_proc(), start_frame(), FrameDirection::Downstream)
            .await
            .unwrap();
        assert!(pre.is_empty());

        let s = gate.inner.lock().unwrap();
        assert!(s.pending_stop.is_none(), "barge-in must drop the pending stop");
        assert_eq!(s.epoch, 1);
        assert_ne!(s.timeout_gen, gen, "drop must disarm the release timer");
    }

    #[tokio::test]
    async fn transcript_claims_pending_stop_exactly_once() {
        let gate = TurnGate::new(16_000, 100);
        gate.on_vad_stop(stop_frame(), 0.0);

        let o1 = gate
            .on_transcript(&dummy_proc(), None, Some(0.0))
            .await
            .unwrap();
        assert!(o1.released_stop, "first claim must win");

        let o2 = gate
            .on_transcript(&dummy_proc(), None, Some(0.0))
            .await
            .unwrap();
        assert!(!o2.released_stop, "second claim must find nothing");
    }

    #[tokio::test]
    async fn stale_timeout_generation_cannot_steal_newer_stash() {
        let gate = TurnGate::new(16_000, 100);
        let old_gen = gate.on_vad_stop(stop_frame(), 0.0);

        // Transcript claims it (and bumps the generation)...
        let o = gate.on_transcript(&dummy_proc(), None, None).await.unwrap();
        assert!(o.released_stop);

        // ...then a NEW turn is stashed.
        gate.on_vad_start(&dummy_proc(), start_frame(), FrameDirection::Downstream)
            .await
            .unwrap();
        gate.on_vad_stop(stop_frame(), 0.0);

        // The old timer fires with its stale generation: must be a no-op.
        gate.clone()
            .release_pending_after(dummy_proc(), old_gen, Duration::from_millis(0))
            .await;
        assert!(
            gate.inner.lock().unwrap().pending_stop.is_some(),
            "stale timer must not release a newer turn's pending stop"
        );
    }

    #[tokio::test]
    async fn empty_transcript_still_releases_pending_stop() {
        let gate = TurnGate::new(16_000, 100);
        gate.on_vad_stop(stop_frame(), 0.0);
        // data: None models an empty/whitespace transcript from Sarvam.
        let o = gate.on_transcript(&dummy_proc(), None, Some(40.0)).await.unwrap();
        assert!(o.released_stop, "empty answer must still close the turn");
        assert!(gate.inner.lock().unwrap().pending_stop.is_none());
    }

    // ---- ledger ----------------------------------------------------------------

    fn gate_with_ledger(entries: &[(u64, f64)]) -> Arc<TurnGate> {
        let gate = TurnGate::new(16_000, 100);
        {
            let mut s = gate.inner.lock().unwrap();
            for &(epoch, ms) in entries {
                s.ledger.push_back(LedgerEntry { epoch, ms });
            }
        }
        gate
    }

    #[test]
    fn ledger_exact_consume_attributes_to_single_epoch() {
        let gate = gate_with_ledger(&[(3, 1840.0)]);
        let mut s = gate.inner.lock().unwrap();
        let (father, billed) = consume_ledger(&mut s, Some(1840.0));
        assert_eq!(father, Some(3));
        assert!((billed - 1840.0).abs() < 0.001);
        assert!(s.ledger.is_empty(), "fully consumed entry must be popped");
    }

    #[test]
    fn ledger_tolerates_server_silence_trim() {
        // Sent 2000ms, server reports 1900ms (trimmed 100ms < tolerance):
        // the 100ms remainder must not linger and pollute the next turn.
        let gate = gate_with_ledger(&[(4, 2000.0)]);
        let mut s = gate.inner.lock().unwrap();
        let (father, _) = consume_ledger(&mut s, Some(1900.0));
        assert_eq!(father, Some(4));
        assert!(s.ledger.is_empty(), "sub-tolerance remainder must be dropped");
    }

    #[test]
    fn ledger_spanning_consume_attributes_to_last_epoch_touched() {
        let gate = gate_with_ledger(&[(5, 500.0), (6, 1500.0)]);
        let mut s = gate.inner.lock().unwrap();
        let (father, _) = consume_ledger(&mut s, Some(1000.0));
        assert_eq!(father, Some(6), "consumption ends in epoch 6");
        assert_eq!(s.ledger.len(), 1);
        assert!((s.ledger[0].ms - 1000.0).abs() < 0.001, "epoch 6 keeps its remainder");
    }

    #[test]
    fn ledger_fallback_without_metrics_consumes_oldest_turn_whole() {
        let gate = gate_with_ledger(&[(7, 800.0), (8, 600.0)]);
        let mut s = gate.inner.lock().unwrap();
        let (father, billed) = consume_ledger(&mut s, None);
        assert_eq!(father, Some(7));
        assert!((billed - 800.0).abs() < 0.001);
        assert_eq!(s.ledger.len(), 1);
    }

    #[test]
    fn ledger_mid_turn_transcript_charges_open_epoch() {
        let gate = TurnGate::new(16_000, 100);
        {
            let mut s = gate.inner.lock().unwrap();
            s.speaking = true;
            s.epoch = 9;
            s.current_sent_ms = 3000.0;
        }
        let mut s = gate.inner.lock().unwrap();
        let (father, _) = consume_ledger(&mut s, Some(1200.0));
        assert_eq!(father, Some(9));
        assert!((s.current_sent_ms - 1800.0).abs() < 0.001);
    }

    // ---- billing integration ----------------------------------------------------

    struct MockCollector {
        session_id: uuid::Uuid,
        events: std::sync::Mutex<Vec<BillingEvent>>,
    }
    impl MockCollector {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                session_id: uuid::Uuid::new_v4(),
                events: std::sync::Mutex::new(vec![]),
            })
        }
        fn events(&self) -> Vec<BillingEvent> { self.events.lock().unwrap().clone() }
    }
    impl BillingCollector for MockCollector {
        fn record(&self, e: BillingEvent) { self.events.lock().unwrap().push(e); }
        fn session_id(&self) -> uuid::Uuid { self.session_id }
    }

    fn rx_ctx(gate: Arc<TurnGate>, billing: Option<Arc<dyn BillingCollector>>) -> Arc<RxCtx> {
        Arc::new(RxCtx {
            processor:         dummy_proc(),
            gate,
            language_fallback: None,
            noise_filter:      None,
            enhancer:          None,
            billing,
        })
    }

    #[tokio::test]
    async fn billing_prefers_server_reported_duration() {
        let gate = gate_with_ledger(&[(1, 2000.0)]);
        let mock = MockCollector::new();
        let ctx  = rx_ctx(gate, Some(mock.clone()));

        // Server says 1.84s even though we sent 2.0s.
        let data = serde_json::json!({
            "transcript": "namaskaram",
            "language_code": "ml-IN",
            "request_id": "req-123",
            "metrics": { "audio_duration": 1.84, "processing_latency": 0.21 }
        });
        handle_transcript(Some(data), &ctx).await;

        let evs = mock.events();
        assert_eq!(evs.len(), 1);
        match &evs[0] {
            BillingEvent::SttUsage { provider, audio_duration_ms, .. } => {
                assert_eq!(provider, "sarvam");
                assert!((audio_duration_ms - 1840.0).abs() < 0.001);
            }
            other => panic!("expected SttUsage, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn billing_falls_back_to_ledger_without_metrics() {
        let gate = gate_with_ledger(&[(2, 750.0)]);
        let mock = MockCollector::new();
        let ctx  = rx_ctx(gate, Some(mock.clone()));

        let data = serde_json::json!({ "transcript": "hello", "language_code": "en-IN" });
        handle_transcript(Some(data), &ctx).await;

        let evs = mock.events();
        assert_eq!(evs.len(), 1);
        match &evs[0] {
            BillingEvent::SttUsage { audio_duration_ms, .. } => {
                assert!((audio_duration_ms - 750.0).abs() < 0.001);
            }
            other => panic!("expected SttUsage, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn no_collector_does_not_panic() {
        let gate = gate_with_ledger(&[(1, 500.0)]);
        let ctx  = rx_ctx(gate, None);
        let data = serde_json::json!({ "transcript": "hello" });
        handle_transcript(Some(data), &ctx).await;
    }

    // ---- builders / config -------------------------------------------------------

    #[test]
    fn with_billing_sets_field() {
        let h = SarvamSttHandler::new(SarvamSttConfig {
            noise_reduction: false,
            ..Default::default()
        }).with_billing(Arc::new(NoopBillingCollector));
        assert!(h.billing.is_some());
    }

    #[test]
    fn ws_url_contains_model_and_sample_rate() {
        let cfg = SarvamSttConfig::default();
        let url = cfg.ws_url();
        assert!(url.contains("saaras"), "model missing: {url}");
        assert!(url.contains("16000"), "sample_rate missing: {url}");
        assert!(url.contains("flush_signal=true"), "flush_signal missing: {url}");
    }

    #[test]
    fn ws_url_translate_model_uses_translate_path() {
        let cfg = SarvamSttConfig {
            model: "saaras:v2.5".into(),
            ..Default::default()
        };
        assert!(cfg.ws_url().contains("speech-to-text-translate"));
    }

    #[test]
    fn urlencoding_handles_special_chars() {
        assert_eq!(urlencoding("saaras:v3"), "saaras%3Av3");
        assert_eq!(urlencoding("en-IN"), "en-IN");
    }
}
