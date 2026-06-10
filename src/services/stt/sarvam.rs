//! Sarvam AI Speech-to-Text service.
//!
//! Connects to Sarvam's WebSocket streaming API and pushes TranscriptionFrames
//! downstream when transcripts arrive.
//!
//! Pipeline position:
//!   transport.input() → SarvamSttHandler → llm → tts → transport.output()
//!
//! Wiring:
//!   let stt = SarvamSttHandler::new(SarvamSttConfig {
//!       api_key: std::env::var("SARVAM_API_KEY").unwrap(),
//!       ..Default::default()
//!   })
//!   .into_processor();
//!
//! Frames consumed:
//!   - StartFrame             → connects WebSocket
//!   - InputAudioRaw          → denoise → base64 encode → send to Sarvam
//!   - VADUserStoppedSpeaking → flush noise filter → send flush signal
//!   - EndFrame / CancelFrame → disconnects WebSocket
//!
//! Frames produced:
//!   - TranscriptionFrame (downstream) on transcript
//!   - ErrorFrame (upstream) on connection / parse errors
//!
//! Auth: api-subscription-key header (lowercase), per SDK source.
//! URL:  wss://api.sarvam.ai/speech-to-text/ws
//! Lang: language-code param (hyphen, not underscore)

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

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
}

impl Default for SarvamSttConfig {
    fn default() -> Self {
        Self {
            api_key:              String::new(),
            model:                "saaras:v3".to_string(),
            language:             Some("unknown".to_string()),
            mode:                 Some("transcribe".to_string()),
            sample_rate:          16_000,
            encoding:             "wav".to_string(),
            high_vad_sensitivity: false,
            vad_signals:          false,
            noise_reduction:      true,
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

/// Transcript payload inside a "data" message.
/// Field is "transcript" for both saarika:v2.5 and saaras:v3.
#[derive(Debug, Deserialize)]
struct SarvamTranscript {
    transcript:    Option<String>,
    language_code: Option<String>,
}

/// VAD event payload inside an "events" message.
#[derive(Debug, Deserialize)]
struct SarvamEvent {
    signal_type: Option<String>,
}

// ---------------------------------------------------------------------------
// Internal state
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

// ---------------------------------------------------------------------------
// SarvamSttHandler
// ---------------------------------------------------------------------------

pub struct SarvamSttHandler {
    config: SarvamSttConfig,
    state:  Arc<Mutex<SarvamSttState>>,
    /// Noise filter shared with the receive task (for reset on transcript).
    noise_filter: Option<Arc<Mutex<RNNoiseFilter>>>,
    /// Optional billing collector — records audio duration per transcript.
    billing: Option<Arc<dyn BillingCollector>>,
    /// Accumulated raw PCM bytes sent to Sarvam since last transcript.
    audio_bytes: Arc<AtomicU64>,
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

        Self {
            config,
            state: Arc::new(Mutex::new(SarvamSttState::new())),
            noise_filter,
            billing: None,
            audio_bytes: Arc::new(AtomicU64::new(0)),
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

        let send_task    = tokio::spawn(run_send_task(sink, ws_rx));
        let lang_fb      = self.config.language.clone();
        let nf_clone     = self.noise_filter.clone();
        let billing      = self.billing.clone();
        let audio_bytes  = self.audio_bytes.clone();
        let sample_rate  = self.config.sample_rate;
        let receive_task = tokio::spawn(run_receive_task(
            stream, processor, lang_fb, nf_clone, billing, audio_bytes, sample_rate,
        ));

        let mut state     = self.state.lock().await;
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
        self.audio_bytes.fetch_add(audio.len() as u64, Ordering::Relaxed);
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
                processor.push_frame(frame, direction).await?;
                self.connect(processor.clone()).await;
            }

            FrameInner::System(SystemFrame::InputAudioRaw(ref audio)) => {
                processor.push_frame(frame.clone(), direction).await?;

                // Denoise then send
                let out_bytes = if let Some(ref nf) = self.noise_filter {
                    let pcm = bytes_to_i16(&audio.audio);
                    let filtered = nf.lock().await.filter(&pcm);
                    if filtered.is_empty() {
                        // Buffering — nothing to send yet
                        return Ok(());
                    }
                    i16_to_bytes(&filtered)
                } else {
                    audio.audio.clone()
                };

                self.send_audio(&out_bytes).await;
            }

            FrameInner::System(SystemFrame::VADUserStoppedSpeaking { .. }) => {
                processor.push_frame(frame, direction).await?;

                // Flush noise filter — send any remaining denoised audio
                if let Some(ref nf) = self.noise_filter {
                    let tail = nf.lock().await.flush();
                    if !tail.is_empty() {
                        self.send_audio(&i16_to_bytes(&tail)).await;
                    }
                }

                self.send_flush().await;
            }

            FrameInner::Control(ControlFrame::End { .. })
            | FrameInner::System(SystemFrame::Cancel { .. }) => {
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

async fn run_receive_task(
    mut stream:          WsStream,
    processor:           FrameProcessor,
    language_fallback:   Option<String>,
    noise_filter:        Option<Arc<Mutex<RNNoiseFilter>>>,
    billing:             Option<Arc<dyn BillingCollector>>,
    audio_bytes:         Arc<AtomicU64>,
    sample_rate:         u32,
) {
    log::debug!("SarvamStt: receive task started");

    while let Some(result) = stream.next().await {
        match result {
            Ok(Message::Text(text)) => {
                handle_message(
                    text.as_str(),
                    &processor,
                    &language_fallback,
                    &noise_filter,
                    &billing,
                    &audio_bytes,
                    sample_rate,
                )
                .await;
            }
            Ok(Message::Close(_)) => {
                log::info!("SarvamStt: server closed WebSocket");
                break;
            }
            Err(e) => {
                let _ = processor
                    .push_error(format!("SarvamStt: receive error: {}", e), false)
                    .await;
                break;
            }
            _ => {}
        }
    }

    log::debug!("SarvamStt: receive task exited");
}

async fn handle_message(
    text:              &str,
    processor:         &FrameProcessor,
    language_fallback: &Option<String>,
    noise_filter:      &Option<Arc<Mutex<RNNoiseFilter>>>,
    billing:           &Option<Arc<dyn BillingCollector>>,
    audio_bytes:       &Arc<AtomicU64>,
    sample_rate:       u32,
) {
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
            handle_transcript(msg.data, processor, language_fallback, noise_filter, billing, audio_bytes, sample_rate).await;
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

async fn handle_transcript(
    data:              Option<serde_json::Value>,
    processor:         &FrameProcessor,
    language_fallback: &Option<String>,
    noise_filter:      &Option<Arc<Mutex<RNNoiseFilter>>>,
    billing:           &Option<Arc<dyn BillingCollector>>,
    audio_bytes:       &Arc<AtomicU64>,
    sample_rate:       u32,
) {
    let data = match data {
        Some(d) => d,
        None    => return,
    };

    let t: SarvamTranscript = match serde_json::from_value(data) {
        Ok(t)  => t,
        Err(e) => { log::warn!("SarvamStt: transcript parse: {}", e); return; }
    };

    let text = match t.transcript {
        Some(s) if !s.trim().is_empty() => s,
        _ => return,
    };

    // Swap the byte counter to zero and emit a billing event for this utterance.
    // PCM i16 LE: 2 bytes per sample, mono → duration_ms = bytes / 2 / sample_rate * 1000
    let bytes = audio_bytes.swap(0, Ordering::Relaxed);
    if bytes > 0 {
        if let Some(bc) = billing {
            let duration_ms = (bytes as f64) / (2.0 * sample_rate as f64) * 1000.0;
            bc.record(BillingEvent::SttUsage {
                session_id:        bc.session_id(),
                provider:          "sarvam".to_string(),
                audio_duration_ms: duration_ms,
                occurred_at:       Utc::now(),
            });
        }
    }

    let language = t.language_code.or_else(|| language_fallback.clone());

    // Reset noise filter — Sarvam's server-side VAD may have finalised
    // this transcript without local VAD firing, so clear any buffered
    // audio to start clean for the next utterance.
    if let Some(ref nf) = noise_filter {
        nf.lock().await.reset();
    }

    let mut frame_data = TranscriptionData::new(text, "", time_now_iso8601());
    frame_data.language  = language;
    frame_data.finalized = true;

    log::info!("SarvamStt: transcript='{}' lang={:?}", frame_data.text, frame_data.language);

    let _ = processor
        .push_frame(Frame::transcription(frame_data), FrameDirection::Downstream)
        .await;
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

    // ---- Duration formula ---------------------------------------------------

    #[test]
    fn audio_duration_formula_16khz_1000ms() {
        // 16000 samples/s × 2 bytes/sample = 32000 bytes → 1000 ms
        let bytes: u64 = 32_000;
        let sample_rate: u32 = 16_000;
        let ms = (bytes as f64) / (2.0 * sample_rate as f64) * 1000.0;
        assert!((ms - 1000.0).abs() < 0.001, "expected 1000ms, got {ms}");
    }

    #[test]
    fn audio_duration_formula_8khz_500ms() {
        // 8000 samples/s × 2 bytes × 0.5s = 8000 bytes → 500 ms
        let bytes: u64 = 8_000;
        let sample_rate: u32 = 8_000;
        let ms = (bytes as f64) / (2.0 * sample_rate as f64) * 1000.0;
        assert!((ms - 500.0).abs() < 0.001, "expected 500ms, got {ms}");
    }

    #[test]
    fn audio_duration_formula_24khz_250ms() {
        // 24000 samples/s × 2 bytes × 0.25s = 12000 bytes → 250ms
        let bytes: u64 = 12_000;
        let sample_rate: u32 = 24_000;
        let ms = (bytes as f64) / (2.0 * sample_rate as f64) * 1000.0;
        assert!((ms - 250.0).abs() < 0.001, "expected 250ms, got {ms}");
    }

    // ---- AtomicU64 counter --------------------------------------------------

    #[test]
    fn audio_bytes_atomic_increments_and_swap_resets() {
        let counter = Arc::new(AtomicU64::new(0));
        counter.fetch_add(1024, Ordering::Relaxed);
        counter.fetch_add(2048, Ordering::Relaxed);
        let total = counter.swap(0, Ordering::Relaxed);
        assert_eq!(total, 3072);
        assert_eq!(counter.load(Ordering::Relaxed), 0, "counter must be zero after swap");
    }

    // ---- with_billing builder -----------------------------------------------

    #[test]
    fn with_billing_sets_field() {
        let h = SarvamSttHandler::new(SarvamSttConfig {
            noise_reduction: false,
            ..Default::default()
        }).with_billing(Arc::new(NoopBillingCollector));
        assert!(h.billing.is_some());
    }

    // ---- handle_transcript billing integration ------------------------------

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

    fn dummy_proc() -> FrameProcessor {
        FrameProcessor::new("test", Box::new(crate::frames::PassthroughHandler), false)
    }

    #[tokio::test]
    async fn billing_transcript_emits_stt_usage_with_correct_duration() {
        // 16kHz, 32000 bytes = 1000ms
        let audio_bytes = Arc::new(AtomicU64::new(32_000));
        let mock    = MockCollector::new();
        let billing: Option<Arc<dyn BillingCollector>> = Some(mock.clone());
        let data    = serde_json::json!({ "transcript": "hello world", "language_code": "en-IN" });

        handle_transcript(Some(data), &dummy_proc(), &None, &None, &billing, &audio_bytes, 16_000).await;

        let evs = mock.events();
        assert_eq!(evs.len(), 1, "expected exactly one SttUsage event");
        match &evs[0] {
            BillingEvent::SttUsage { provider, audio_duration_ms, .. } => {
                assert_eq!(provider, "sarvam");
                assert!((audio_duration_ms - 1000.0).abs() < 0.001, "expected 1000ms, got {audio_duration_ms}");
            }
            other => panic!("expected SttUsage, got {:?}", other),
        }
        // Counter swapped to zero
        assert_eq!(audio_bytes.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn billing_transcript_with_zero_bytes_emits_no_event() {
        let audio_bytes = Arc::new(AtomicU64::new(0));
        let mock    = MockCollector::new();
        let billing: Option<Arc<dyn BillingCollector>> = Some(mock.clone());
        let data    = serde_json::json!({ "transcript": "hello", "language_code": "en-IN" });

        handle_transcript(Some(data), &dummy_proc(), &None, &None, &billing, &audio_bytes, 16_000).await;
        assert_eq!(mock.events().len(), 0, "zero bytes must not emit billing event");
    }

    #[tokio::test]
    async fn billing_no_collector_transcript_does_not_panic() {
        let audio_bytes = Arc::new(AtomicU64::new(16_000));
        let billing: Option<Arc<dyn BillingCollector>> = None;
        let data    = serde_json::json!({ "transcript": "hello", "language_code": "en-IN" });

        handle_transcript(Some(data), &dummy_proc(), &None, &None, &billing, &audio_bytes, 16_000).await;
        // no panic; counter still reset
        assert_eq!(audio_bytes.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn billing_transcript_counter_resets_per_utterance() {
        let audio_bytes = Arc::new(AtomicU64::new(32_000)); // 1st utterance: 1000ms
        let mock    = MockCollector::new();
        let billing: Option<Arc<dyn BillingCollector>> = Some(mock.clone());
        let data1   = serde_json::json!({ "transcript": "first", "language_code": "en-IN" });
        handle_transcript(Some(data1), &dummy_proc(), &None, &None, &billing, &audio_bytes, 16_000).await;

        // Simulate second utterance
        audio_bytes.fetch_add(16_000, Ordering::Relaxed); // 500ms
        let data2 = serde_json::json!({ "transcript": "second", "language_code": "en-IN" });
        handle_transcript(Some(data2), &dummy_proc(), &None, &None, &billing, &audio_bytes, 16_000).await;

        let evs = mock.events();
        assert_eq!(evs.len(), 2);
        let durations: Vec<f64> = evs.iter().filter_map(|e| match e {
            BillingEvent::SttUsage { audio_duration_ms, .. } => Some(*audio_duration_ms),
            _ => None,
        }).collect();
        assert!((durations[0] - 1000.0).abs() < 0.001, "1st utterance: {}", durations[0]);
        assert!((durations[1] -  500.0).abs() < 0.001, "2nd utterance: {}", durations[1]);
    }

    // ---- Config / URL helpers -----------------------------------------------

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
