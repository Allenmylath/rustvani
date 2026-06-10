//! Deepgram Speech-to-Text WebSocket service.
//!
//! Connects to Deepgram's streaming API and pushes TranscriptionFrames
//! downstream when final transcripts arrive.
//!
//! Pipeline position:
//!   transport.input() → DeepgramSttHandler → llm → tts → transport.output()
//!
//! Wiring:
//!   let stt = DeepgramSttHandler::new(DeepgramSttConfig {
//!       api_key: std::env::var("DEEPGRAM_API_KEY").unwrap(),
//!       ..Default::default()
//!   })
//!   .into_processor();
//!
//! Frames consumed:
//!   - StartFrame             → connects WebSocket
//!   - InputAudioRaw          → send raw binary PCM to Deepgram
//!   - VADUserStoppedSpeaking → send Finalize to force final transcript
//!   - EndFrame / CancelFrame → disconnect (CloseStream + task abort)
//!
//! Frames produced:
//!   - TranscriptionFrame (downstream) on final transcript
//!   - ErrorFrame (upstream) on connection / parse errors
//!
//! Auth: Authorization: Token <api_key> header
//! URL:  wss://api.deepgram.com/v1/listen

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use chrono::Utc;
use futures::{SinkExt, StreamExt};
use log;
use serde::Deserialize;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::http::Request;
use tokio_tungstenite::tungstenite::Message;

use crate::billing::{BillingCollector, BillingEvent};
use crate::error::Result;
use crate::frames::{
    ControlFrame, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor,
    SystemFrame, TranscriptionData,
};

const DEEPGRAM_WSS_URL: &str = "wss://api.deepgram.com/v1/listen";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for [`DeepgramSttHandler`].
#[derive(Debug, Clone)]
pub struct DeepgramSttConfig {
    /// Deepgram API key.
    pub api_key: String,

    /// Model name. e.g. "nova-3", "nova-2", "base", "enhanced".
    pub model: String,

    /// BCP-47 language code. e.g. "en-US", "en", "multi".
    pub language: String,

    /// Audio encoding. Use "linear16" for raw PCM i16 LE.
    pub encoding: String,

    /// Audio sample rate in Hz. Must match what the transport sends.
    pub sample_rate: u32,

    /// Number of audio channels.
    pub channels: u32,

    /// Emit interim (non-final) transcription results.
    /// Final results are always emitted.
    pub interim_results: bool,

    /// Add punctuation to transcripts.
    pub punctuate: bool,

    /// Apply smart formatting (numbers, dates, etc.).
    pub smart_format: bool,

    /// Endpointing silence threshold in ms before Deepgram finalises a
    /// transcript automatically. `None` disables server-side endpointing.
    pub endpointing: Option<u32>,

    /// Utterance-end silence in ms. Enables `utterance_end` events.
    pub utterance_end_ms: Option<u32>,

    /// Override the default WebSocket URL (for on-prem / proxy deployments).
    pub base_url: Option<String>,
}

impl Default for DeepgramSttConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: "nova-3".to_string(),
            language: "en-US".to_string(),
            encoding: "linear16".to_string(),
            sample_rate: 16_000,
            channels: 1,
            interim_results: true,
            punctuate: true,
            smart_format: false,
            endpointing: Some(300),
            utterance_end_ms: None,
            base_url: None,
        }
    }
}

impl DeepgramSttConfig {
    fn ws_url(&self) -> String {
        let base = self.base_url.as_deref().unwrap_or(DEEPGRAM_WSS_URL);
        let mut params = vec![
            format!("model={}", self.model),
            format!("language={}", self.language),
            format!("encoding={}", self.encoding),
            format!("sample_rate={}", self.sample_rate),
            format!("channels={}", self.channels),
            format!("interim_results={}", self.interim_results),
            format!("punctuate={}", self.punctuate),
            format!("smart_format={}", self.smart_format),
        ];
        if let Some(ep) = self.endpointing {
            params.push(format!("endpointing={}", ep));
        }
        if let Some(uem) = self.utterance_end_ms {
            params.push(format!("utterance_end_ms={}", uem));
        }
        format!("{}?{}", base, params.join("&"))
    }
}

// ---------------------------------------------------------------------------
// Deepgram WebSocket response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct DgEnvelope {
    #[serde(rename = "type")]
    msg_type: String,
}

#[derive(Debug, Deserialize)]
struct DgResults {
    is_final: bool,
    channel: DgChannel,
    #[serde(default)]
    from_finalize: bool,
}

#[derive(Debug, Deserialize)]
struct DgChannel {
    alternatives: Vec<DgAlternative>,
}

#[derive(Debug, Deserialize)]
struct DgAlternative {
    transcript: String,
}

// ---------------------------------------------------------------------------
// Internal channel message type
// ---------------------------------------------------------------------------

enum OutgoingMsg {
    Binary(Vec<u8>),
    Text(String),
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

struct DeepgramSttState {
    ws_tx:          Option<mpsc::Sender<OutgoingMsg>>,
    send_task:      Option<JoinHandle<()>>,
    receive_task:   Option<JoinHandle<()>>,
    keepalive_task: Option<JoinHandle<()>>,
}

impl DeepgramSttState {
    fn new() -> Self {
        Self {
            ws_tx: None,
            send_task: None,
            receive_task: None,
            keepalive_task: None,
        }
    }
}

// ---------------------------------------------------------------------------
// DeepgramSttHandler
// ---------------------------------------------------------------------------

pub struct DeepgramSttHandler {
    config:      DeepgramSttConfig,
    state:       Arc<Mutex<DeepgramSttState>>,
    billing:     Option<Arc<dyn BillingCollector>>,
    audio_bytes: Arc<AtomicU64>,
}

impl DeepgramSttHandler {
    pub fn new(config: DeepgramSttConfig) -> Self {
        Self {
            config,
            state: Arc::new(Mutex::new(DeepgramSttState::new())),
            billing: None,
            audio_bytes: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn with_billing(mut self, billing: Arc<dyn BillingCollector>) -> Self {
        self.billing = Some(billing);
        self
    }

    pub fn into_processor(self) -> FrameProcessor {
        FrameProcessor::new("DeepgramStt", Box::new(self), false)
    }
}

// ---------------------------------------------------------------------------
// Connection / disconnection
// ---------------------------------------------------------------------------

impl DeepgramSttHandler {
    async fn connect(&self, processor: FrameProcessor) {
        let url = self.config.ws_url();
        log::info!("DeepgramStt: connecting to {}", url);

        let host = url
            .trim_start_matches("wss://")
            .trim_start_matches("ws://")
            .split('/')
            .next()
            .unwrap_or("api.deepgram.com");

        let request = match Request::builder()
            .uri(&url)
            .header("Host", host)
            .header("Authorization", format!("Token {}", self.config.api_key))
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
                    .push_error(format!("DeepgramStt: request build failed: {}", e), false)
                    .await;
                return;
            }
        };

        let ws_stream = match tokio_tungstenite::connect_async(request).await {
            Ok((stream, _)) => stream,
            Err(e) => {
                let _ = processor
                    .push_error(format!("DeepgramStt: connect failed: {}", e), false)
                    .await;
                return;
            }
        };

        let (sink, stream) = ws_stream.split();
        let (ws_tx, ws_rx) = mpsc::channel::<OutgoingMsg>(128);

        let send_task = tokio::spawn(run_send_task(sink, ws_rx));

        let billing     = self.billing.clone();
        let audio_bytes = self.audio_bytes.clone();
        let sample_rate = self.config.sample_rate;
        let receive_task =
            tokio::spawn(run_receive_task(stream, processor, billing, audio_bytes, sample_rate));

        let ka_tx = ws_tx.clone();
        let keepalive_task = tokio::spawn(run_keepalive_task(ka_tx));

        let mut state = self.state.lock().await;
        state.ws_tx          = Some(ws_tx);
        state.send_task      = Some(send_task);
        state.receive_task   = Some(receive_task);
        state.keepalive_task = Some(keepalive_task);

        log::info!("DeepgramStt: connected");
    }

    async fn disconnect(&self) {
        let mut state = self.state.lock().await;

        if let Some(tx) = &state.ws_tx {
            let _ = tx
                .send(OutgoingMsg::Text(r#"{"type":"CloseStream"}"#.to_string()))
                .await;
        }

        if let Some(h) = state.keepalive_task.take() { h.abort(); }
        if let Some(h) = state.receive_task.take()   { h.abort(); }
        if let Some(h) = state.send_task.take()       { h.abort(); }
        state.ws_tx = None;

        log::info!("DeepgramStt: disconnected");
    }

    async fn send_audio(&self, audio: &[u8]) {
        self.audio_bytes.fetch_add(audio.len() as u64, Ordering::Relaxed);
        let tx = { self.state.lock().await.ws_tx.clone() };
        if let Some(tx) = tx {
            let _ = tx.send(OutgoingMsg::Binary(audio.to_vec())).await;
        }
    }

    async fn send_finalize(&self) {
        let tx = { self.state.lock().await.ws_tx.clone() };
        if let Some(tx) = tx {
            let _ = tx
                .send(OutgoingMsg::Text(r#"{"type":"Finalize"}"#.to_string()))
                .await;
        }
    }
}

// ---------------------------------------------------------------------------
// FrameHandler impl
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameHandler for DeepgramSttHandler {
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
                self.send_audio(&audio.audio).await;
            }

            FrameInner::System(SystemFrame::VADUserStoppedSpeaking { .. }) => {
                processor.push_frame(frame, direction).await?;
                self.send_finalize().await;
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

    fn can_generate_metrics(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Background tasks
// ---------------------------------------------------------------------------

type WsSink = futures::stream::SplitSink<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    Message,
>;

type WsStream = futures::stream::SplitStream<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
>;

async fn run_send_task(mut sink: WsSink, mut rx: mpsc::Receiver<OutgoingMsg>) {
    while let Some(msg) = rx.recv().await {
        let ws_msg = match msg {
            OutgoingMsg::Binary(bytes) => Message::Binary(bytes.into()),
            OutgoingMsg::Text(text)    => Message::Text(text.into()),
        };
        if sink.send(ws_msg).await.is_err() {
            log::warn!("DeepgramStt: send failed — closing send task");
            break;
        }
    }
    let _ = sink.close().await;
    log::debug!("DeepgramStt: send task exited");
}

async fn run_receive_task(
    mut stream:  WsStream,
    processor:   FrameProcessor,
    billing:     Option<Arc<dyn BillingCollector>>,
    audio_bytes: Arc<AtomicU64>,
    sample_rate: u32,
) {
    log::debug!("DeepgramStt: receive task started");

    while let Some(result) = stream.next().await {
        match result {
            Ok(Message::Text(text)) => {
                handle_message(text.as_str(), &processor, &billing, &audio_bytes, sample_rate).await;
            }
            Ok(Message::Close(_)) => {
                log::info!("DeepgramStt: server closed WebSocket");
                break;
            }
            Err(e) => {
                let _ = processor
                    .push_error(format!("DeepgramStt: receive error: {}", e), false)
                    .await;
                break;
            }
            _ => {}
        }
    }

    log::debug!("DeepgramStt: receive task exited");
}

// Deepgram closes idle connections after ~10 s; send KeepAlive every 8 s.
async fn run_keepalive_task(tx: mpsc::Sender<OutgoingMsg>) {
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(8)).await;
        if tx
            .send(OutgoingMsg::Text(r#"{"type":"KeepAlive"}"#.to_string()))
            .await
            .is_err()
        {
            break;
        }
        log::trace!("DeepgramStt: sent KeepAlive");
    }
}

async fn handle_message(
    text:        &str,
    processor:   &FrameProcessor,
    billing:     &Option<Arc<dyn BillingCollector>>,
    audio_bytes: &Arc<AtomicU64>,
    sample_rate: u32,
) {
    log::debug!("DeepgramStt: raw message: {}", text);

    let envelope: DgEnvelope = match serde_json::from_str(text) {
        Ok(e)  => e,
        Err(e) => {
            log::warn!("DeepgramStt: parse error: {} — raw: {}", e, text);
            return;
        }
    };

    match envelope.msg_type.as_str() {
        "Results" => {
            let result: DgResults = match serde_json::from_str(text) {
                Ok(r)  => r,
                Err(e) => {
                    log::warn!("DeepgramStt: Results parse: {}", e);
                    return;
                }
            };

            if !result.is_final {
                return;
            }

            let transcript = match result.channel.alternatives.first() {
                Some(a) => a.transcript.trim().to_string(),
                None    => return,
            };

            if transcript.is_empty() {
                return;
            }

            let bytes = audio_bytes.swap(0, Ordering::Relaxed);
            if bytes > 0 {
                if let Some(bc) = billing {
                    // PCM i16 LE: 2 bytes/sample, mono
                    let duration_ms = (bytes as f64) / (2.0 * sample_rate as f64) * 1000.0;
                    bc.record(BillingEvent::SttUsage {
                        session_id:        bc.session_id(),
                        provider:          "deepgram".to_string(),
                        audio_duration_ms: duration_ms,
                        occurred_at:       Utc::now(),
                    });
                }
            }

            let mut frame_data = TranscriptionData::new(transcript.clone(), "", time_now_iso8601());
            frame_data.finalized = true;

            log::info!(
                "DeepgramStt: transcript='{}' from_finalize={}",
                transcript,
                result.from_finalize,
            );

            let _ = processor
                .push_frame(Frame::transcription(frame_data), FrameDirection::Downstream)
                .await;
        }

        "Metadata"      => log::debug!("DeepgramStt: metadata received"),
        "SpeechStarted" => log::debug!("DeepgramStt: speech started"),
        "UtteranceEnd"  => log::debug!("DeepgramStt: utterance end"),

        "Error" => {
            log::warn!("DeepgramStt: server error: {}", text);
            let _ = processor
                .push_error(format!("DeepgramStt: server error: {}", text), false)
                .await;
        }

        other => log::debug!("DeepgramStt: unknown message type: {}", other),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn time_now_iso8601() -> String {
    let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    format!("{}.{:03}", d.as_secs(), d.subsec_millis())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ws_url_contains_required_params() {
        let cfg = DeepgramSttConfig::default();
        let url = cfg.ws_url();
        assert!(url.starts_with("wss://api.deepgram.com/v1/listen"), "base url: {url}");
        assert!(url.contains("model=nova-3"),     "model missing: {url}");
        assert!(url.contains("language=en-US"),   "language missing: {url}");
        assert!(url.contains("sample_rate=16000"), "sample_rate missing: {url}");
        assert!(url.contains("encoding=linear16"), "encoding missing: {url}");
        assert!(url.contains("endpointing=300"),   "endpointing missing: {url}");
    }

    #[test]
    fn ws_url_custom_base_url() {
        let cfg = DeepgramSttConfig {
            base_url: Some("wss://my.proxy.example.com/v1/listen".to_string()),
            ..Default::default()
        };
        assert!(cfg.ws_url().starts_with("wss://my.proxy.example.com"));
    }

    #[test]
    fn ws_url_no_endpointing_when_none() {
        let cfg = DeepgramSttConfig {
            endpointing: None,
            ..Default::default()
        };
        assert!(!cfg.ws_url().contains("endpointing"), "endpointing should be absent: {}", cfg.ws_url());
    }

    #[test]
    fn audio_duration_formula_16khz_1000ms() {
        let bytes: u64 = 32_000;
        let sample_rate: u32 = 16_000;
        let ms = (bytes as f64) / (2.0 * sample_rate as f64) * 1000.0;
        assert!((ms - 1000.0).abs() < 0.001, "expected 1000ms, got {ms}");
    }

    #[test]
    fn audio_bytes_counter_resets_on_swap() {
        let counter = Arc::new(AtomicU64::new(0));
        counter.fetch_add(16_000, Ordering::Relaxed);
        let total = counter.swap(0, Ordering::Relaxed);
        assert_eq!(total, 16_000);
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn with_billing_sets_field() {
        use crate::billing::NoopBillingCollector;
        let h = DeepgramSttHandler::new(DeepgramSttConfig::default())
            .with_billing(Arc::new(NoopBillingCollector));
        assert!(h.billing.is_some());
    }
}
