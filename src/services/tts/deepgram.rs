//! Deepgram Text-to-Speech service.
//!
//! Connects to Deepgram's WebSocket streaming TTS API and pushes OutputAudioRaw
//! frames downstream as audio arrives.
//!
//! Key protocol differences from Sarvam:
//!   - Config is passed via URL query params (no config message needed)
//!   - Audio arrives as binary WS messages (no base64 decode)
//!   - Interruption uses Clear message (no reconnect needed)
//!   - Text is sent via Speak messages; Flush triggers synthesis
//!
//! Timing prints:
//!   [tts] speak           — text sent to Deepgram WS
//!   [tts] flush            — flush signal sent
//!   [tts] first_audio      — first audio bytes received back

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
    AudioRawData, ControlFrame, DataFrame, Frame, FrameDirection, FrameHandler, FrameInner,
    FrameProcessor, SystemFrame,
};

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

// ---------------------------------------------------------------------------
// Supported encodings
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepgramEncoding {
    Linear16,
    Mulaw,
    Alaw,
}

impl DeepgramEncoding {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Linear16 => "linear16",
            Self::Mulaw    => "mulaw",
            Self::Alaw     => "alaw",
        }
    }

    pub fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s.to_lowercase().as_str() {
            "linear16" => Ok(Self::Linear16),
            "mulaw"    => Ok(Self::Mulaw),
            "alaw"     => Ok(Self::Alaw),
            other      => Err(format!(
                "Unsupported encoding '{}'. Must be one of: linear16, mulaw, alaw",
                other
            )),
        }
    }

    /// Bytes per sample for this encoding.
    pub fn bytes_per_sample(&self) -> u32 {
        match self {
            Self::Linear16 => 2,
            Self::Mulaw | Self::Alaw => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DeepgramTtsConfig {
    pub api_key: String,
    /// Deepgram voice model (e.g. "aura-2-helena-en").
    pub voice: String,
    pub encoding: DeepgramEncoding,
    pub sample_rate: u32,
    /// Opt out of Deepgram Model Improvement Program.
    /// See <https://dpgr.am/deepgram-mip> for pricing impacts.
    pub mip_opt_out: Option<bool>,
    pub base_url: String,
}

impl Default for DeepgramTtsConfig {
    fn default() -> Self {
        Self {
            api_key:     String::new(),
            voice:       "aura-2-helena-en".to_string(),
            encoding:    DeepgramEncoding::Linear16,
            sample_rate: 24000,
            mip_opt_out: None,
            base_url:    "wss://api.deepgram.com".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Deepgram WS message types (received)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct DeepgramMessage {
    #[serde(rename = "type")]
    msg_type: String,
    #[serde(default)]
    description: Option<String>,
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

struct TtsState {
    ws_tx:              Option<mpsc::Sender<Message>>,
    bot_speaking:       bool,
    send_task:          Option<JoinHandle<()>>,
    receive_task:       Option<JoinHandle<()>>,
    /// Timestamp when the last flush was sent.
    flush_sent_at:      Option<f64>,
    /// True once first audio has arrived after a flush.
    first_audio_logged: bool,
    /// Accumulated character count since last Flushed confirmation.
    pending_chars:      usize,
}

impl TtsState {
    fn new() -> Self {
        Self {
            ws_tx:              None,
            bot_speaking:       false,
            send_task:          None,
            receive_task:       None,
            flush_sent_at:      None,
            first_audio_logged: true,
            pending_chars:      0,
        }
    }
}

// ---------------------------------------------------------------------------
// DeepgramTtsHandler
// ---------------------------------------------------------------------------

pub struct DeepgramTtsHandler {
    config:      DeepgramTtsConfig,
    sample_rate: u32,
    state:       Arc<Mutex<TtsState>>,
    /// Optional billing collector — records char counts per synthesis flush.
    billing:     Option<Arc<dyn BillingCollector>>,
}

impl DeepgramTtsHandler {
    pub fn new(config: DeepgramTtsConfig) -> std::result::Result<Self, String> {
        if config.api_key.is_empty() {
            return Err("DeepgramTts: api_key is required".to_string());
        }

        let sample_rate = config.sample_rate;

        Ok(Self {
            config,
            sample_rate,
            state: Arc::new(Mutex::new(TtsState::new())),
            billing: None,
        })
    }

    pub fn with_billing(mut self, billing: Arc<dyn BillingCollector>) -> Self {
        self.billing = Some(billing);
        self
    }

    pub fn into_processor(self) -> FrameProcessor {
        FrameProcessor::new("DeepgramTts", Box::new(self), false)
    }

    // ---- Connection management ----

    async fn connect(&self, processor: FrameProcessor) {
        let mut params = vec![
            format!("model={}", urlencoding(&self.config.voice)),
            format!("encoding={}", self.config.encoding.as_str()),
            format!("sample_rate={}", self.sample_rate),
        ];
        if let Some(mip) = self.config.mip_opt_out {
            params.push(format!("mip_opt_out={}", mip));
        }

        let ws_url = format!(
            "{}/v1/speak?{}",
            self.config.base_url,
            params.join("&")
        );
        log::info!("DeepgramTts: connecting to {}", ws_url);

        let request = match Request::builder()
            .uri(&ws_url)
            .header("Host", extract_host(&self.config.base_url))
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
                    .push_error(format!("DeepgramTts: request build failed: {}", e), false)
                    .await;
                return;
            }
        };

        let ws_stream = match tokio_tungstenite::connect_async(request).await {
            Ok((s, _)) => s,
            Err(e) => {
                let _ = processor
                    .push_error(format!("DeepgramTts: connect failed: {}", e), false)
                    .await;
                return;
            }
        };

        let (sink, stream) = ws_stream.split();
        let (ws_tx, ws_rx) = mpsc::channel::<Message>(64);

        let send_task = tokio::spawn(run_send_task(sink, ws_rx));
        let receive_task = tokio::spawn(run_receive_task(
            stream,
            processor.clone(),
            self.sample_rate,
            self.state.clone(),
            self.billing.clone(),
            self.config.voice.clone(),
        ));

        {
            let mut state = self.state.lock().await;
            state.ws_tx        = Some(ws_tx);
            state.send_task    = Some(send_task);
            state.receive_task = Some(receive_task);
        }

        // No config message needed — everything is in the URL query params.
        log::info!("DeepgramTts: connected (voice={}, encoding={}, sample_rate={})",
            self.config.voice, self.config.encoding.as_str(), self.sample_rate);
    }

    async fn disconnect(&self) {
        // Send Close message for graceful shutdown before aborting tasks.
        self.send_ws_json(r#"{"type":"Close"}"#).await;

        let mut state = self.state.lock().await;
        if let Some(h) = state.send_task.take()    { h.abort(); }
        if let Some(h) = state.receive_task.take() { h.abort(); }
        state.ws_tx = None;
        log::info!("DeepgramTts: disconnected");
    }

    /// Bill any text submitted to Deepgram that has not yet been confirmed by a
    /// `Flushed`/`Cleared` message. Called on session end: the WebSocket is
    /// about to close, so those confirmations will never arrive — without this
    /// the chars Deepgram already synthesized would never be billed.
    async fn bill_pending_chars(&self) {
        let chars = {
            let mut s = self.state.lock().await;
            std::mem::replace(&mut s.pending_chars, 0)
        };
        if chars > 0 {
            if let Some(bc) = &self.billing {
                bc.record(BillingEvent::TtsUsage {
                    session_id:  bc.session_id(),
                    provider:    "deepgram".to_string(),
                    voice:       self.config.voice.clone(),
                    char_count:  chars,
                    occurred_at: Utc::now(),
                });
            }
        }
    }

    // ---- WS message helpers ----

    async fn send_ws_json(&self, json: &str) {
        let tx = { self.state.lock().await.ws_tx.clone() };
        if let Some(tx) = tx {
            let _ = tx.send(Message::Text(json.into())).await;
        }
    }

    async fn send_speak(&self, text: &str) {
        let tx = {
            let mut s = self.state.lock().await;
            s.pending_chars += text.chars().count();
            s.ws_tx.clone()
        };
        if let Some(tx) = tx {
            let ts = now();
            println!("[{:.3}] [tts] speak  ({} chars): {:?}", ts, text.len(), text);
            let msg = serde_json::json!({
                "type": "Speak",
                "text": text
            });
            let _ = tx.send(Message::Text(msg.to_string().into())).await;
        }
    }

    async fn send_flush(&self) {
        let tx = { self.state.lock().await.ws_tx.clone() };
        if let Some(tx) = tx {
            let ts = now();
            println!("[{:.3}] [tts] flush  ← synthesis starts now", ts);
            {
                let mut state = self.state.lock().await;
                state.flush_sent_at      = Some(ts);
                state.first_audio_logged = false;
            }
            let msg = r#"{"type":"Flush"}"#;
            let _ = tx.send(Message::Text(msg.into())).await;
        }
    }

    async fn send_clear(&self) {
        let ts = now();
        println!("[{:.3}] [tts] clear  ← interruption", ts);
        self.send_ws_json(r#"{"type":"Clear"}"#).await;
    }
}

// ---------------------------------------------------------------------------
// FrameHandler impl
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameHandler for DeepgramTtsHandler {
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

            FrameInner::Control(ControlFrame::LLMFullResponseStart) => {
                processor.push_frame(frame, direction).await?;
            }

            FrameInner::Data(DataFrame::LLMText(text)) => {
                // Send each text token directly to Deepgram as a Speak message.
                // Deepgram buffers internally and only synthesizes on Flush.
                let text = text.clone();
                if !text.is_empty() {
                    self.send_speak(&text).await;
                }
                processor.push_frame(frame, direction).await?;
            }

            FrameInner::Control(ControlFrame::LLMFullResponseEnd) => {
                // Flush triggers Deepgram to synthesize all buffered text.
                self.send_flush().await;
                processor.push_frame(frame, direction).await?;
            }

            FrameInner::System(SystemFrame::BotStartedSpeaking) => {
                self.state.lock().await.bot_speaking = true;
                processor.push_frame(frame, direction).await?;
            }

            FrameInner::System(SystemFrame::BotStoppedSpeaking) => {
                self.state.lock().await.bot_speaking = false;
                processor.push_frame(frame, direction).await?;
            }

            FrameInner::System(SystemFrame::Interruption) => {
                let bot_speaking = self.state.lock().await.bot_speaking;

                if bot_speaking {
                    // Deepgram supports Clear without reconnecting — just
                    // clears the internal buffer and stops sending audio.
                    log::info!("DeepgramTts: interruption while speaking — sending Clear");
                    self.send_clear().await;
                }

                processor.push_frame(frame, direction).await?;
            }

            FrameInner::Control(ControlFrame::End { .. })
            | FrameInner::System(SystemFrame::Cancel { .. }) => {
                // Bill in-flight text before the socket closes — the Flushed
                // confirmation that would normally trigger billing won't arrive.
                self.bill_pending_chars().await;
                self.disconnect().await;
                processor.push_frame(frame, direction).await?;
            }

            _ => {
                processor.push_frame(frame, direction).await?;
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Background task type aliases
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

// ---------------------------------------------------------------------------
// Background tasks
// ---------------------------------------------------------------------------

/// Forwards messages from the mpsc channel to the WebSocket sink.
async fn run_send_task(mut sink: WsSink, mut rx: mpsc::Receiver<Message>) {
    while let Some(msg) = rx.recv().await {
        if sink.send(msg).await.is_err() {
            log::warn!("DeepgramTts: send task — write failed, closing");
            break;
        }
    }
    let _ = sink.close().await;
    log::debug!("DeepgramTts: send task exited");
}

/// Receives messages from Deepgram and pushes audio frames downstream.
async fn run_receive_task(
    mut stream:  WsStream,
    processor:   FrameProcessor,
    sample_rate: u32,
    state:       Arc<Mutex<TtsState>>,
    billing:     Option<Arc<dyn BillingCollector>>,
    voice:       String,
) {
    log::debug!("DeepgramTts: receive task started");

    while let Some(result) = stream.next().await {
        match result {
            // Binary messages contain raw audio data — no decoding needed.
            Ok(Message::Binary(audio)) => {
                if audio.is_empty() {
                    continue;
                }

                // First-audio timing log
                {
                    let mut s = state.lock().await;
                    if !s.first_audio_logged {
                        s.first_audio_logged = true;
                        let ts = now();
                        if let Some(flush_at) = s.flush_sent_at {
                            println!(
                                "[{:.3}] [tts] first_audio  ← Deepgram latency: {:.3}s",
                                ts,
                                ts - flush_at
                            );
                        } else {
                            println!("[{:.3}] [tts] first_audio", ts);
                        }
                    }
                }

                log::debug!("DeepgramTts: received {} audio bytes", audio.len());

                let frame = Frame::output_audio_raw(AudioRawData::new(
                    audio.to_vec(),
                    sample_rate,
                    1,
                ));
                let _ = processor
                    .push_frame(frame, FrameDirection::Downstream)
                    .await;
            }

            // Text messages contain metadata / control signals.
            Ok(Message::Text(text)) => {
                handle_text_message(text.as_str(), &processor, &state, &billing, &voice).await;
            }

            Ok(Message::Close(_)) => {
                log::info!("DeepgramTts: server closed WebSocket");
                break;
            }

            Err(e) => {
                let _ = processor
                    .push_error(format!("DeepgramTts: receive error: {}", e), false)
                    .await;
                break;
            }

            _ => {}
        }
    }

    log::debug!("DeepgramTts: receive task exited");
}

async fn handle_text_message(
    text:      &str,
    processor: &FrameProcessor,
    state:     &Arc<Mutex<TtsState>>,
    billing:   &Option<Arc<dyn BillingCollector>>,
    voice:     &str,
) {
    log::trace!("DeepgramTts: raw message: {}", text);

    let msg: DeepgramMessage = match serde_json::from_str(text) {
        Ok(m)  => m,
        Err(e) => {
            log::warn!("DeepgramTts: parse error: {} — raw: {}", e, text);
            return;
        }
    };

    match msg.msg_type.as_str() {
        "Metadata" => {
            log::trace!("DeepgramTts: metadata received");
        }

        "Flushed" => {
            log::debug!("DeepgramTts: Flushed — synthesis complete for current buffer");
            // Take the accumulated char count and emit a billing event.
            let chars = {
                let mut s = state.lock().await;
                std::mem::replace(&mut s.pending_chars, 0)
            };
            if chars > 0 {
                if let Some(bc) = billing {
                    bc.record(BillingEvent::TtsUsage {
                        session_id:  bc.session_id(),
                        provider:    "deepgram".to_string(),
                        voice:       voice.to_string(),
                        char_count:  chars,
                        occurred_at: Utc::now(),
                    });
                }
            }
            let _ = processor;
        }

        "Cleared" => {
            log::debug!("DeepgramTts: Cleared — buffer cleared after interruption");
            // The text was already submitted to Deepgram for synthesis (and
            // partially delivered as audio before the Clear took effect), so
            // Deepgram bills for it. Record it as usage rather than discarding,
            // otherwise every barge-in silently under-bills TTS.
            let chars = {
                let mut s = state.lock().await;
                std::mem::replace(&mut s.pending_chars, 0)
            };
            if chars > 0 {
                if let Some(bc) = billing {
                    bc.record(BillingEvent::TtsUsage {
                        session_id:  bc.session_id(),
                        provider:    "deepgram".to_string(),
                        voice:       voice.to_string(),
                        char_count:  chars,
                        occurred_at: Utc::now(),
                    });
                }
            }
        }

        "Warning" => {
            let desc = msg.description.as_deref().unwrap_or("unknown warning");
            log::warn!("DeepgramTts: server warning: {}", desc);
        }

        other => {
            log::debug!("DeepgramTts: unhandled message type: '{}'", other);
        }
    }
}

// ---------------------------------------------------------------------------
// URL / host helpers
// ---------------------------------------------------------------------------

fn urlencoding(s: &str) -> String {
    s.chars()
        .flat_map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9'
            | '-' | '_' | '.' | '~' => vec![c],
            ':' => vec!['%', '3', 'A'],
            '/' => vec!['%', '2', 'F'],
            _ => format!("%{:02X}", c as u32).chars().collect(),
        })
        .collect()
}

/// Extracts the host portion from a URL for the Host header.
fn extract_host(url: &str) -> String {
    url.trim_start_matches("wss://")
        .trim_start_matches("ws://")
        .split('/')
        .next()
        .unwrap_or("api.deepgram.com")
        .to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoding_from_str() {
        assert_eq!(DeepgramEncoding::from_str("linear16").unwrap(), DeepgramEncoding::Linear16);
        assert_eq!(DeepgramEncoding::from_str("mulaw").unwrap(), DeepgramEncoding::Mulaw);
        assert_eq!(DeepgramEncoding::from_str("alaw").unwrap(), DeepgramEncoding::Alaw);
        assert_eq!(DeepgramEncoding::from_str("LINEAR16").unwrap(), DeepgramEncoding::Linear16);
        assert!(DeepgramEncoding::from_str("mp3").is_err());
    }

    #[test]
    fn test_encoding_as_str() {
        assert_eq!(DeepgramEncoding::Linear16.as_str(), "linear16");
        assert_eq!(DeepgramEncoding::Mulaw.as_str(), "mulaw");
        assert_eq!(DeepgramEncoding::Alaw.as_str(), "alaw");
    }

    #[test]
    fn test_bytes_per_sample() {
        assert_eq!(DeepgramEncoding::Linear16.bytes_per_sample(), 2);
        assert_eq!(DeepgramEncoding::Mulaw.bytes_per_sample(), 1);
        assert_eq!(DeepgramEncoding::Alaw.bytes_per_sample(), 1);
    }

    #[test]
    fn test_default_config() {
        let cfg = DeepgramTtsConfig::default();
        assert_eq!(cfg.voice, "aura-2-helena-en");
        assert_eq!(cfg.encoding, DeepgramEncoding::Linear16);
        assert_eq!(cfg.sample_rate, 24000);
        assert_eq!(cfg.base_url, "wss://api.deepgram.com");
        assert!(cfg.mip_opt_out.is_none());
    }

    #[test]
    fn test_handler_requires_api_key() {
        let result = DeepgramTtsHandler::new(DeepgramTtsConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_handler_with_api_key() {
        let h = DeepgramTtsHandler::new(DeepgramTtsConfig {
            api_key: "test-key".to_string(),
            ..DeepgramTtsConfig::default()
        })
        .unwrap();
        assert_eq!(h.sample_rate, 24000);
        assert_eq!(h.config.voice, "aura-2-helena-en");
    }

    #[test]
    fn test_handler_custom_encoding() {
        let h = DeepgramTtsHandler::new(DeepgramTtsConfig {
            api_key:  "test-key".to_string(),
            encoding: DeepgramEncoding::Mulaw,
            ..DeepgramTtsConfig::default()
        })
        .unwrap();
        assert_eq!(h.config.encoding, DeepgramEncoding::Mulaw);
    }

    #[test]
    fn test_handler_custom_sample_rate() {
        let h = DeepgramTtsHandler::new(DeepgramTtsConfig {
            api_key:     "test-key".to_string(),
            sample_rate: 16000,
            ..DeepgramTtsConfig::default()
        })
        .unwrap();
        assert_eq!(h.sample_rate, 16000);
    }

    #[test]
    fn test_extract_host() {
        assert_eq!(extract_host("wss://api.deepgram.com"), "api.deepgram.com");
        assert_eq!(extract_host("wss://custom.host.io/path"), "custom.host.io");
        assert_eq!(extract_host("ws://localhost:8080"), "localhost:8080");
    }

    #[test]
    fn test_urlencoding() {
        assert_eq!(urlencoding("aura-2-helena-en"), "aura-2-helena-en");
        assert_eq!(urlencoding("model:v2"), "model%3Av2");
    }

    // ---- Billing tests -------------------------------------------------------

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
    impl crate::billing::BillingCollector for MockCollector {
        fn record(&self, e: BillingEvent) { self.events.lock().unwrap().push(e); }
        fn session_id(&self) -> uuid::Uuid { self.session_id }
    }

    fn state_with_chars(n: usize) -> Arc<Mutex<TtsState>> {
        let mut s = TtsState::new();
        s.pending_chars = n;
        Arc::new(Mutex::new(s))
    }

    fn dummy_proc() -> FrameProcessor {
        FrameProcessor::new("test", Box::new(crate::frames::PassthroughHandler), false)
    }

    #[tokio::test]
    async fn billing_flushed_emits_tts_usage_event() {
        let state   = state_with_chars(42);
        let mock    = MockCollector::new();
        let billing: Option<Arc<dyn crate::billing::BillingCollector>> = Some(mock.clone());

        handle_text_message(r#"{"type":"Flushed"}"#, &dummy_proc(), &state, &billing, "aura-2-helena-en").await;

        let evs = mock.events();
        assert_eq!(evs.len(), 1, "expected exactly one TtsUsage event");
        match &evs[0] {
            BillingEvent::TtsUsage { provider, char_count, voice, .. } => {
                assert_eq!(provider, "deepgram");
                assert_eq!(*char_count, 42);
                assert_eq!(voice, "aura-2-helena-en");
            }
            other => panic!("expected TtsUsage, got {:?}", other),
        }
        assert_eq!(state.lock().await.pending_chars, 0, "pending_chars must be reset after Flushed");
    }

    #[tokio::test]
    async fn billing_flushed_with_zero_chars_emits_no_event() {
        let state   = state_with_chars(0);
        let mock    = MockCollector::new();
        let billing: Option<Arc<dyn crate::billing::BillingCollector>> = Some(mock.clone());

        handle_text_message(r#"{"type":"Flushed"}"#, &dummy_proc(), &state, &billing, "v").await;
        assert_eq!(mock.events().len(), 0);
    }

    #[tokio::test]
    async fn billing_cleared_bills_submitted_chars_then_resets() {
        let state   = state_with_chars(99);
        let mock    = MockCollector::new();
        let billing: Option<Arc<dyn crate::billing::BillingCollector>> = Some(mock.clone());

        handle_text_message(r#"{"type":"Cleared"}"#, &dummy_proc(), &state, &billing, "v").await;

        // Submitted chars were billed by Deepgram before the Clear took effect,
        // so we record them rather than silently under-billing the barge-in.
        let events = mock.events();
        assert_eq!(events.len(), 1, "Cleared must bill the submitted chars");
        match &events[0] {
            BillingEvent::TtsUsage { char_count, .. } => assert_eq!(*char_count, 99),
            other => panic!("expected TtsUsage, got {other:?}"),
        }
        assert_eq!(state.lock().await.pending_chars, 0, "pending_chars must be zeroed by Cleared");
    }

    #[tokio::test]
    async fn billing_cleared_with_zero_chars_emits_no_event() {
        let state   = state_with_chars(0);
        let mock    = MockCollector::new();
        let billing: Option<Arc<dyn crate::billing::BillingCollector>> = Some(mock.clone());

        handle_text_message(r#"{"type":"Cleared"}"#, &dummy_proc(), &state, &billing, "v").await;

        assert_eq!(mock.events().len(), 0, "no chars submitted → no event");
    }

    #[tokio::test]
    async fn billing_no_collector_flushed_does_not_panic() {
        let state   = state_with_chars(50);
        let billing: Option<Arc<dyn crate::billing::BillingCollector>> = None;

        handle_text_message(r#"{"type":"Flushed"}"#, &dummy_proc(), &state, &billing, "v").await;
        // Still resets pending_chars
        assert_eq!(state.lock().await.pending_chars, 0);
    }

    #[test]
    fn with_billing_sets_field() {
        use crate::billing::NoopBillingCollector;
        let h = DeepgramTtsHandler::new(DeepgramTtsConfig {
            api_key: "key".into(),
            ..DeepgramTtsConfig::default()
        }).unwrap().with_billing(Arc::new(NoopBillingCollector));
        assert!(h.billing.is_some());
    }
}
