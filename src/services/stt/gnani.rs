//! Gnani (Vachana) Speech-to-Text service.
//!
//! Connects to Gnani's WebSocket streaming API (`wss://api.vachana.ai/stt/v3/stream`)
//! and pushes `TranscriptionFrame`s downstream when transcripts arrive.
//!
//! Pipeline position:
//!   transport.input() → GnaniSttHandler → llm → tts → transport.output()
//!
//! Frames consumed:
//!   - StartFrame             → connects WebSocket
//!   - InputAudioRaw          → buffer into 1024-byte chunks → send binary
//!   - VADUserStoppedSpeaking → flush any remaining buffered audio
//!   - EndFrame / CancelFrame → disconnects WebSocket
//!
//! Frames produced:
//!   - TranscriptionFrame (downstream) on transcript
//!   - ErrorFrame (upstream) on connection / parse errors
//!
//! Auth: `x-api-key-id` header.
//! Audio: raw PCM signed 16-bit little-endian, mono, chunked into 1024-byte frames.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use futures::{SinkExt, StreamExt};
use log;
use serde::Deserialize;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::http::Request;
use tokio_tungstenite::tungstenite::Message;

use crate::error::Result;
use crate::frames::{
    ControlFrame, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor,
    SystemFrame, TranscriptionData,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GNANI_BASE_WSS: &str = "wss://api.vachana.ai";
const STT_PATH: &str       = "/stt/v3/stream";
/// Gnani expects 512 samples × 2 bytes = 1024 bytes per binary frame.
const CHUNK_SIZE: usize    = 1024;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for `GnaniSttHandler`.
#[derive(Debug, Clone)]
pub struct GnaniSttConfig {
    /// Vachana API key.
    pub api_key: String,

    /// BCP-47 language code e.g. "hi-IN", "en-IN", "ta-IN".
    /// Default: "en-IN".
    pub language_code: String,

    /// Audio sample rate in Hz. Must match the actual audio source.
    /// Accepted values: 8000, 16000, 44100, 48000.
    /// Default: 16000.
    pub sample_rate: u32,

    /// Output format.
    /// - `"verbatim"` — raw spoken-form output (default).
    /// - `"transcribe"` — enables Inverse Text Normalization (ITN).
    pub format: String,

    /// When `format` is `"transcribe"`, set `true` to render digits in the
    /// native script of the target language.
    /// Default: `false`.
    pub itn_native_numerals: bool,
}

impl Default for GnaniSttConfig {
    fn default() -> Self {
        Self {
            api_key:             String::new(),
            language_code:       "en-IN".to_string(),
            sample_rate:         16_000,
            format:              "verbatim".to_string(),
            itn_native_numerals: false,
        }
    }
}

impl GnaniSttConfig {
    fn ws_url(&self) -> String {
        format!("{}{}", GNANI_BASE_WSS, STT_PATH)
    }
}

// ---------------------------------------------------------------------------
// Server message types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct GnaniMessage {
    #[serde(rename = "type")]
    msg_type: String,
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    timestamp: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    audio_duration_ms: Option<f64>,
    #[serde(default)]
    segment_id: Option<String>,
    #[serde(default)]
    segment_index: Option<u64>,
    #[serde(default)]
    latency: Option<u64>,
    #[serde(default)]
    config: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

struct GnaniSttState {
    ws_tx:        Option<mpsc::Sender<Message>>,
    send_task:    Option<JoinHandle<()>>,
    receive_task: Option<JoinHandle<()>>,
    /// Buffered audio bytes waiting to form a complete 1024-byte chunk.
    audio_buffer: Vec<u8>,
}

impl GnaniSttState {
    fn new() -> Self {
        Self {
            ws_tx:        None,
            send_task:    None,
            receive_task: None,
            audio_buffer: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// GnaniSttHandler
// ---------------------------------------------------------------------------

pub struct GnaniSttHandler {
    config: GnaniSttConfig,
    state:  Arc<Mutex<GnaniSttState>>,
}

impl GnaniSttHandler {
    pub fn new(config: GnaniSttConfig) -> Self {
        Self {
            config,
            state: Arc::new(Mutex::new(GnaniSttState::new())),
        }
    }

    pub fn into_processor(self) -> FrameProcessor {
        FrameProcessor::new("GnaniStt", Box::new(self), false)
    }
}

// ---------------------------------------------------------------------------
// Connection / disconnection
// ---------------------------------------------------------------------------

impl GnaniSttHandler {
    async fn connect(&self, processor: FrameProcessor) {
        let url = self.config.ws_url();
        log::info!("GnaniStt: connecting to {}", url);

        let mut request_builder = Request::builder()
            .uri(&url)
            .header("Host", "api.vachana.ai")
            .header("x-api-key-id", &self.config.api_key)
            .header("lang_code", &self.config.language_code)
            .header("x-sample-rate", self.config.sample_rate.to_string())
            .header("x-format", &self.config.format)
            .header("Connection", "Upgrade")
            .header("Upgrade", "websocket")
            .header("Sec-WebSocket-Version", "13")
            .header(
                "Sec-WebSocket-Key",
                tokio_tungstenite::tungstenite::handshake::client::generate_key(),
            );

        if self.config.itn_native_numerals {
            request_builder = request_builder.header("itn_native_numerals", "true");
        }

        let request = match request_builder.body(()) {
            Ok(r) => r,
            Err(e) => {
                let _ = processor
                    .push_error(format!("GnaniStt: request build failed: {}", e), false)
                    .await;
                return;
            }
        };

        let ws_stream = match tokio_tungstenite::connect_async(request).await {
            Ok((stream, _)) => stream,
            Err(e) => {
                let _ = processor
                    .push_error(format!("GnaniStt: connect failed: {}", e), false)
                    .await;
                return;
            }
        };

        let (sink, stream) = ws_stream.split();
        let (ws_tx, ws_rx) = mpsc::channel::<Message>(64);

        let send_task    = tokio::spawn(run_send_task(sink, ws_rx));
        let lang_clone   = self.config.language_code.clone();
        let receive_task = tokio::spawn(run_receive_task(stream, processor, lang_clone));

        let mut state = self.state.lock().await;
        state.ws_tx        = Some(ws_tx);
        state.send_task    = Some(send_task);
        state.receive_task = Some(receive_task);

        log::info!("GnaniStt: connected");
    }

    async fn disconnect(&self) {
        let mut state = self.state.lock().await;
        if let Some(h) = state.receive_task.take() { h.abort(); }
        if let Some(h) = state.send_task.take()    { h.abort(); }
        state.ws_tx = None;
        state.audio_buffer.clear();
        log::info!("GnaniStt: disconnected");
    }

    /// Buffer audio into 1024-byte chunks and send as binary WS messages.
    async fn send_audio(&self, audio: &[u8]) {
        let mut state = self.state.lock().await;
        state.audio_buffer.extend_from_slice(audio);

        while state.audio_buffer.len() >= CHUNK_SIZE {
            let chunk: Vec<u8> = state.audio_buffer.drain(..CHUNK_SIZE).collect();
            if let Some(tx) = &state.ws_tx {
                let _ = tx.send(Message::Binary(chunk.into())).await;
            }
        }
    }

    /// Send any remaining buffered audio (partial frame allowed on flush).
    async fn flush_audio(&self) {
        let mut state = self.state.lock().await;
        if !state.audio_buffer.is_empty() {
            let chunk: Vec<u8> = state.audio_buffer.drain(..).collect();
            if let Some(tx) = &state.ws_tx {
                let _ = tx.send(Message::Binary(chunk.into())).await;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FrameHandler impl
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameHandler for GnaniSttHandler {
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
                self.flush_audio().await;
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

async fn run_send_task(mut sink: WsSink, mut rx: mpsc::Receiver<Message>) {
    while let Some(msg) = rx.recv().await {
        if sink.send(msg).await.is_err() {
            log::warn!("GnaniStt: send failed — closing send task");
            break;
        }
    }
    let _ = sink.close().await;
    log::debug!("GnaniStt: send task exited");
}

async fn run_receive_task(
    mut stream:          WsStream,
    processor:           FrameProcessor,
    language_fallback:   String,
) {
    log::debug!("GnaniStt: receive task started");

    while let Some(result) = stream.next().await {
        match result {
            Ok(Message::Text(text)) => {
                handle_message(text.as_str(), &processor, &language_fallback).await;
            }
            Ok(Message::Close(_)) => {
                log::info!("GnaniStt: server closed WebSocket");
                break;
            }
            Err(e) => {
                let _ = processor
                    .push_error(format!("GnaniStt: receive error: {}", e), false)
                    .await;
                break;
            }
            _ => {}
        }
    }

    log::debug!("GnaniStt: receive task exited");
}

async fn handle_message(
    text:              &str,
    processor:         &FrameProcessor,
    language_fallback: &str,
) {
    log::debug!("GnaniStt: raw message: {}", text);

    let msg: GnaniMessage = match serde_json::from_str(text) {
        Ok(m) => m,
        Err(e) => {
            log::warn!("GnaniStt: parse error: {} — raw: {}", e, text);
            return;
        }
    };

    match msg.msg_type.as_str() {
        "connected" => {
            log::info!(
                "GnaniStt: server connected — msg={:?} config={:?}",
                msg.message,
                msg.config
            );
        }
        "processing" => {
            log::debug!("GnaniStt: processing segment");
        }
        "transcript" => {
            if let Some(text) = msg.text {
                if !text.trim().is_empty() {
                    let mut frame_data = TranscriptionData::new(text, "", time_now_iso8601());
                    frame_data.language  = Some(language_fallback.to_string());
                    frame_data.finalized = true;

                    log::info!(
                        "GnaniStt: transcript='{}' segment_index={:?} latency={:?}ms",
                        frame_data.text,
                        msg.segment_index,
                        msg.latency
                    );

                    let _ = processor
                        .push_frame(Frame::transcription(frame_data), FrameDirection::Downstream)
                        .await;
                }
            }
        }
        "error" => {
            let err_msg = msg.message.unwrap_or_else(|| "unknown server error".to_string());
            log::warn!("GnaniStt: server error: {}", err_msg);
            let _ = processor
                .push_error(format!("GnaniStt: server error: {}", err_msg), false)
                .await;
        }
        other => {
            log::debug!("GnaniStt: unknown message type: {}", other);
        }
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
    fn test_default_config() {
        let cfg = GnaniSttConfig::default();
        assert!(cfg.api_key.is_empty());
        assert_eq!(cfg.language_code, "en-IN");
        assert_eq!(cfg.sample_rate, 16_000);
        assert_eq!(cfg.format, "verbatim");
        assert!(!cfg.itn_native_numerals);
    }

    #[test]
    fn test_ws_url() {
        let cfg = GnaniSttConfig::default();
        assert_eq!(cfg.ws_url(), "wss://api.vachana.ai/stt/v3/stream");
    }

    #[test]
    fn test_config_with_itn() {
        let cfg = GnaniSttConfig {
            api_key: "test-key".to_string(),
            language_code: "hi-IN".to_string(),
            format: "transcribe".to_string(),
            itn_native_numerals: true,
            ..Default::default()
        };
        assert_eq!(cfg.language_code, "hi-IN");
        assert_eq!(cfg.format, "transcribe");
        assert!(cfg.itn_native_numerals);
    }

    #[test]
    fn test_handler_creation() {
        let handler = GnaniSttHandler::new(GnaniSttConfig {
            api_key: "test-key".to_string(),
            ..Default::default()
        });
        assert_eq!(handler.config.sample_rate, 16_000);
    }

    #[test]
    fn test_time_now_iso8601() {
        let ts = time_now_iso8601();
        assert!(!ts.is_empty());
        assert!(ts.contains('.'));
    }
}
