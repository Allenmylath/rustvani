//! Sarvam AI Text-to-Speech service.
//!
//! Connects to Sarvam's WebSocket streaming TTS API and pushes OutputAudioRaw
//! frames downstream as audio arrives.
//!
//! Timing prints added to expose exact moments:
//!   [tts] send_text_chunk — text sent to Sarvam WS
//!   [tts] send_flush      — flush signal sent, synthesis starts
//!   [tts] first_audio     — first audio bytes received back

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

use crate::billing::{BillingCollector, BillingEvent};
use crate::error::Result;
use crate::frames::{
    AudioRawData, ControlFrame, DataFrame, Frame, FrameDirection, FrameHandler, FrameInner,
    FrameProcessor, SystemFrame,
};
use crate::utils::sentence_splitter::{extract_sentences, find_sentence_end};
use crate::utils::text_preprocessor::preprocess_for_tts;

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
// Model configs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TtsModelConfig {
    pub supports_pitch: bool,
    pub supports_loudness: bool,
    pub supports_temperature: bool,
    pub default_sample_rate: u32,
    pub default_speaker: &'static str,
    pub pace_min: f32,
    pub pace_max: f32,
    pub preprocessing_always_enabled: bool,
    pub speakers: &'static [&'static str],
}

const V2_SPEAKERS: &[&str] = &[
    "anushka", "abhilash", "manisha", "vidya", "arya", "karun", "hitesh",
];

const V3_SPEAKERS: &[&str] = &[
    "aditya", "ritu", "priya", "neha", "rahul", "pooja", "rohan", "simran",
    "kavya", "amit", "dev", "ishita", "shreya", "ratan", "varun", "manan",
    "sumit", "roopa", "kabir", "aayan", "shubh", "ashutosh", "advait",
    "amelia", "sophia",
];

pub const MODEL_BULBUL_V2: TtsModelConfig = TtsModelConfig {
    supports_pitch:                true,
    supports_loudness:             true,
    supports_temperature:          false,
    default_sample_rate:           22050,
    default_speaker:               "anushka",
    pace_min:                      0.3,
    pace_max:                      3.0,
    preprocessing_always_enabled:  false,
    speakers:                      V2_SPEAKERS,
};

pub const MODEL_BULBUL_V3_BETA: TtsModelConfig = TtsModelConfig {
    supports_pitch:                false,
    supports_loudness:             false,
    supports_temperature:          true,
    default_sample_rate:           24000,
    default_speaker:               "shubh",
    pace_min:                      0.5,
    pace_max:                      2.0,
    preprocessing_always_enabled:  true,
    speakers:                      V3_SPEAKERS,
};

pub const MODEL_BULBUL_V3: TtsModelConfig = TtsModelConfig {
    supports_pitch:                false,
    supports_loudness:             false,
    supports_temperature:          true,
    default_sample_rate:           24000,
    default_speaker:               "shubh",
    pace_min:                      0.5,
    pace_max:                      2.0,
    preprocessing_always_enabled:  true,
    speakers:                      V3_SPEAKERS,
};

pub fn get_model_config(model: &str) -> Option<&'static TtsModelConfig> {
    match model {
        "bulbul:v2"      => Some(&MODEL_BULBUL_V2),
        "bulbul:v3-beta" => Some(&MODEL_BULBUL_V3_BETA),
        "bulbul:v3"      => Some(&MODEL_BULBUL_V3),
        _                => None,
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SarvamTtsConfig {
    pub api_key: String,
    pub model: String,
    pub voice: String,
    pub language: String,
    pub sample_rate: Option<u32>,
    pub pace: f32,
    pub pitch: Option<f32>,
    pub loudness: Option<f32>,
    pub temperature: Option<f32>,
    pub enable_preprocessing: bool,
    pub min_buffer_size: usize,
    pub max_chunk_length: usize,
    pub url: String,
}

impl Default for SarvamTtsConfig {
    fn default() -> Self {
        Self {
            api_key:              String::new(),
            model:                "bulbul:v2".to_string(),
            voice:                String::new(),
            language:             "en-IN".to_string(),
            sample_rate:          None,
            pace:                 1.0,
            pitch:                None,
            loudness:             None,
            temperature:          None,
            enable_preprocessing: false,
            min_buffer_size:      50,
            max_chunk_length:     150,
            url:                  "wss://api.sarvam.ai/text-to-speech/ws".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Sarvam WS message types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct SarvamTtsMessage {
    #[serde(rename = "type")]
    msg_type: String,
    data: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

struct TtsState {
    ws_tx:          Option<mpsc::Sender<String>>,
    text_buffer:    String,
    bot_speaking:   bool,
    send_task:      Option<JoinHandle<()>>,
    receive_task:   Option<JoinHandle<()>>,
    keepalive_task: Option<JoinHandle<()>>,
    /// Timestamp when the last flush was sent — for measuring TTS API latency.
    flush_sent_at:  Option<f64>,
    /// True once first audio has arrived after a flush — reset each flush.
    first_audio_logged: bool,
    /// Accumulated character count since last "done" confirmation.
    pending_chars: usize,
}

impl TtsState {
    fn new() -> Self {
        Self {
            ws_tx:              None,
            text_buffer:        String::new(),
            bot_speaking:       false,
            send_task:          None,
            receive_task:       None,
            keepalive_task:     None,
            flush_sent_at:      None,
            first_audio_logged: true,
            pending_chars:      0,
        }
    }
}

// ---------------------------------------------------------------------------
// SarvamTtsHandler
// ---------------------------------------------------------------------------

pub struct SarvamTtsHandler {
    config:       SarvamTtsConfig,
    model_config: &'static TtsModelConfig,
    sample_rate:  u32,
    state:        Arc<Mutex<TtsState>>,
    /// Optional billing collector — records char counts per synthesis cycle.
    billing:      Option<Arc<dyn BillingCollector>>,
}

impl SarvamTtsHandler {
    pub fn new(mut config: SarvamTtsConfig) -> std::result::Result<Self, String> {
        let model_config = get_model_config(&config.model)
            .ok_or_else(|| format!("Unsupported TTS model: '{}'", config.model))?;

        let sample_rate = config.sample_rate
            .unwrap_or(model_config.default_sample_rate);

        if config.voice.is_empty() {
            config.voice = model_config.default_speaker.to_string();
        }

        if config.pace < model_config.pace_min || config.pace > model_config.pace_max {
            log::warn!(
                "SarvamTts: pace {:.2} outside range ({:.1}–{:.1}) for {} — clamping",
                config.pace, model_config.pace_min, model_config.pace_max, config.model
            );
            config.pace = config.pace.clamp(model_config.pace_min, model_config.pace_max);
        }

        if model_config.preprocessing_always_enabled {
            config.enable_preprocessing = true;
        }

        if !model_config.supports_pitch && config.pitch.is_some() {
            log::warn!("SarvamTts: pitch not supported for {} — ignoring", config.model);
            config.pitch = None;
        }
        if !model_config.supports_loudness && config.loudness.is_some() {
            log::warn!("SarvamTts: loudness not supported for {} — ignoring", config.model);
            config.loudness = None;
        }
        if !model_config.supports_temperature && config.temperature.is_some() {
            log::warn!("SarvamTts: temperature not supported for {} — ignoring", config.model);
            config.temperature = None;
        }

        Ok(Self {
            config,
            model_config,
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
        FrameProcessor::new("SarvamTts", Box::new(self), false)
    }

    // ---- Connection management ----

    async fn connect(&self, processor: FrameProcessor) {
        let ws_url = format!(
            "{}?model={}",
            self.config.url,
            urlencoding(&self.config.model)
        );
        log::info!("SarvamTts: connecting to {}", ws_url);

        let request = match Request::builder()
            .uri(&ws_url)
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
                    .push_error(format!("SarvamTts: request build failed: {}", e), false)
                    .await;
                return;
            }
        };

        let ws_stream = match tokio_tungstenite::connect_async(request).await {
            Ok((s, _)) => s,
            Err(e) => {
                let _ = processor
                    .push_error(format!("SarvamTts: connect failed: {}", e), false)
                    .await;
                return;
            }
        };

        let (sink, stream) = ws_stream.split();
        let (ws_tx, ws_rx) = mpsc::channel::<String>(64);

        let send_task      = tokio::spawn(run_tts_send_task(sink, ws_rx));
        let receive_task   = tokio::spawn(run_tts_receive_task(
            stream, processor.clone(), self.sample_rate, self.state.clone(),
            self.billing.clone(), self.config.voice.clone(),
        ));
        let keepalive_task = tokio::spawn(run_tts_keepalive_task(ws_tx.clone()));

        {
            let mut state      = self.state.lock().await;
            state.ws_tx        = Some(ws_tx.clone());
            state.send_task    = Some(send_task);
            state.receive_task = Some(receive_task);
            state.keepalive_task = Some(keepalive_task);
        }

        self.send_config(&ws_tx).await;
        log::info!("SarvamTts: connected and configured");
    }

    async fn disconnect(&self) {
        let mut state = self.state.lock().await;
        if let Some(h) = state.send_task.take()      { h.abort(); }
        if let Some(h) = state.receive_task.take()   { h.abort(); }
        if let Some(h) = state.keepalive_task.take() { h.abort(); }
        state.ws_tx = None;
        log::info!("SarvamTts: disconnected");
    }

    // ---- WS message builders ----

    async fn send_config(&self, ws_tx: &mpsc::Sender<String>) {
        let mut config_data = serde_json::json!({
            "target_language_code": self.config.language,
            "speaker":              self.config.voice,
            "speech_sample_rate":   self.sample_rate.to_string(),
            "enable_preprocessing": self.config.enable_preprocessing,
            "min_buffer_size":      self.config.min_buffer_size,
            "max_chunk_length":     self.config.max_chunk_length,
            "output_audio_codec":   "linear16",
            "output_audio_bitrate": "128k",
            "pace":                 self.config.pace,
            "model":                self.config.model,
        });

        if let Some(v) = self.config.pitch       { config_data["pitch"]       = v.into(); }
        if let Some(v) = self.config.loudness    { config_data["loudness"]    = v.into(); }
        if let Some(v) = self.config.temperature { config_data["temperature"] = v.into(); }

        let msg = serde_json::json!({ "type": "config", "data": config_data });
        let _ = ws_tx.send(msg.to_string()).await;
        log::debug!("SarvamTts: config sent");
    }

    async fn send_text_chunk(&self, text: &str) {
        if !text.chars().any(|c| c.is_alphanumeric()) {
            log::debug!("SarvamTts: skipping punctuation-only chunk: {:?}", text);
            return;
        }
        let tx = {
            let mut s = self.state.lock().await;
            s.pending_chars += text.chars().count();
            s.ws_tx.clone()
        };
        if let Some(tx) = tx {
            let ts = now();
            println!("[{:.3}] [tts] send_text_chunk  ({} chars): {:?}", ts, text.len(), text);
            let msg = serde_json::json!({
                "type": "text",
                "data": { "text": text }
            });
            let _ = tx.send(msg.to_string()).await;
        }
    }

    async fn send_flush(&self) {
        let tx = { self.state.lock().await.ws_tx.clone() };
        if let Some(tx) = tx {
            let ts = now();
            println!("[{:.3}] [tts] send_flush  ← synthesis starts now", ts);
            // Record flush time and reset first-audio flag
            {
                let mut state = self.state.lock().await;
                state.flush_sent_at      = Some(ts);
                state.first_audio_logged = false;
            }
            let msg = serde_json::json!({ "type": "flush" });
            let _ = tx.send(msg.to_string()).await;
        }
    }
}

// ---------------------------------------------------------------------------
// FrameHandler impl
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameHandler for SarvamTtsHandler {
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
                let text = text.clone();
                let min_buffer_size  = self.config.min_buffer_size;
                let max_chunk_length = self.config.max_chunk_length;

                let chunks = {
                    let mut state = self.state.lock().await;
                    state.text_buffer.push_str(&text);

                    if state.text_buffer.len() < min_buffer_size
                        && find_sentence_end(&state.text_buffer).is_none()
                    {
                        vec![]
                    } else {
                        extract_sentences(&mut state.text_buffer, max_chunk_length)
                    }
                };

                for chunk in chunks {
                    self.send_text_chunk(&preprocess_for_tts(&chunk)).await;
                }

                processor.push_frame(frame, direction).await?;
            }

            FrameInner::Control(ControlFrame::LLMFullResponseEnd) => {
                let remaining = {
                    let mut state = self.state.lock().await;
                    let tail = state.text_buffer.trim().to_string();
                    state.text_buffer.clear();
                    tail
                };

                if !remaining.is_empty() {
                    self.send_text_chunk(&preprocess_for_tts(&remaining)).await;
                }

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
                let bot_speaking = {
                    let mut state = self.state.lock().await;
                    state.text_buffer.clear();
                    state.bot_speaking
                };

                if bot_speaking {
                    log::info!("SarvamTts: interruption while speaking — reconnecting");
                    self.disconnect().await;
                    self.connect(processor.clone()).await;
                }

                processor.push_frame(frame, direction).await?;
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

async fn run_tts_send_task(mut sink: WsSink, mut rx: mpsc::Receiver<String>) {
    while let Some(text) = rx.recv().await {
        if sink.send(Message::Text(text.into())).await.is_err() {
            log::warn!("SarvamTts: send task — write failed, closing");
            break;
        }
    }
    let _ = sink.close().await;
    log::debug!("SarvamTts: send task exited");
}

async fn run_tts_receive_task(
    mut stream:  WsStream,
    processor:   FrameProcessor,
    sample_rate: u32,
    state:       Arc<Mutex<TtsState>>,
    billing:     Option<Arc<dyn BillingCollector>>,
    voice:       String,
) {
    log::debug!("SarvamTts: receive task started");

    while let Some(result) = stream.next().await {
        match result {
            Ok(Message::Text(text)) => {
                handle_tts_message(
                    text.as_str(), &processor, sample_rate, &state, &billing, &voice,
                ).await;
            }
            Ok(Message::Close(_)) => {
                log::info!("SarvamTts: server closed WebSocket");
                break;
            }
            Err(e) => {
                let _ = processor
                    .push_error(format!("SarvamTts: receive error: {}", e), false)
                    .await;
                break;
            }
            _ => {}
        }
    }

    log::debug!("SarvamTts: receive task exited");
}

async fn run_tts_keepalive_task(tx: mpsc::Sender<String>) {
    const KEEPALIVE_SECS: u64 = 20;
    loop {
        tokio::time::sleep(Duration::from_secs(KEEPALIVE_SECS)).await;
        let msg = serde_json::json!({ "type": "ping" }).to_string();
        if tx.send(msg).await.is_err() {
            break;
        }
        log::debug!("SarvamTts: keepalive ping sent");
    }
}

async fn handle_tts_message(
    text:        &str,
    processor:   &FrameProcessor,
    sample_rate: u32,
    state:       &Arc<Mutex<TtsState>>,
    billing:     &Option<Arc<dyn BillingCollector>>,
    voice:       &str,
) {
    log::trace!("SarvamTts: raw message: {}", text);

    let msg: SarvamTtsMessage = match serde_json::from_str(text) {
        Ok(m)  => m,
        Err(e) => {
            log::warn!("SarvamTts: parse error: {} — raw: {}", e, text);
            return;
        }
    };

    match msg.msg_type.as_str() {
        "audio" => {
            let data = match msg.data {
                Some(d) => d,
                None    => return,
            };

            let b64 = match data["audio"].as_str() {
                Some(s) if !s.is_empty() => s,
                _ => return,
            };

            let mut audio = match BASE64.decode(b64) {
                Ok(b)  => b,
                Err(e) => {
                    log::warn!("SarvamTts: base64 decode error: {}", e);
                    return;
                }
            };

            if audio.len() > 44 && audio.starts_with(b"RIFF") {
                audio = audio[44..].to_vec();
            }

            if audio.is_empty() {
                return;
            }

            // ---- First audio timing log ----
            {
                let mut s = state.lock().await;
                if !s.first_audio_logged {
                    s.first_audio_logged = true;
                    let ts = now();
                    if let Some(flush_at) = s.flush_sent_at {
                        println!(
                            "[{:.3}] [tts] first_audio  ← Sarvam synthesis latency: {:.3}s",
                            ts,
                            ts - flush_at
                        );
                    } else {
                        println!("[{:.3}] [tts] first_audio", ts);
                    }
                }
            }

            log::debug!("SarvamTts: received {} audio bytes", audio.len());

            let frame = Frame::output_audio_raw(AudioRawData::new(audio, sample_rate, 1));
            let _ = processor
                .push_frame(frame, FrameDirection::Downstream)
                .await;
        }

        "error" => {
            let err = msg.data
                .as_ref()
                .and_then(|d| d["message"].as_str())
                .unwrap_or("unknown error")
                .to_string();
            log::error!("SarvamTts: server error: {}", err);
            let _ = processor
                .push_error(format!("SarvamTts: {}", err), false)
                .await;
        }

        "done" => {
            log::debug!("SarvamTts: synthesis done signal received");
            let chars = {
                let mut s = state.lock().await;
                std::mem::replace(&mut s.pending_chars, 0)
            };
            if chars > 0 {
                if let Some(bc) = billing {
                    bc.record(BillingEvent::TtsUsage {
                        session_id:  bc.session_id(),
                        provider:    "sarvam".to_string(),
                        voice:       voice.to_string(),
                        char_count:  chars,
                        occurred_at: Utc::now(),
                    });
                }
            }
        }

        other => {
            log::debug!("SarvamTts: unhandled message type: '{}'", other);
        }
    }
}

// ---------------------------------------------------------------------------
// URL encoding helper
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_v2() {
        let c = get_model_config("bulbul:v2").unwrap();
        assert!(c.supports_pitch && c.supports_loudness && !c.supports_temperature);
        assert_eq!(c.default_sample_rate, 22050);
        assert_eq!(c.default_speaker, "anushka");
        assert!(!c.preprocessing_always_enabled);
    }

    #[test]
    fn test_model_config_v3_beta() {
        let c = get_model_config("bulbul:v3-beta").unwrap();
        assert!(!c.supports_pitch && !c.supports_loudness && c.supports_temperature);
        assert_eq!(c.default_sample_rate, 24000);
        assert!(c.preprocessing_always_enabled);
    }

    #[test]
    fn test_model_config_v3() {
        let c = get_model_config("bulbul:v3").unwrap();
        assert!(c.supports_temperature && c.preprocessing_always_enabled);
    }

    #[test]
    fn test_unknown_model() {
        assert!(get_model_config("bulbul:v99").is_none());
    }

    #[test]
    fn test_handler_default_voice_resolved() {
        let h = SarvamTtsHandler::new(SarvamTtsConfig {
            api_key: "test".to_string(),
            ..SarvamTtsConfig::default()
        })
        .unwrap();
        assert_eq!(h.config.voice, "anushka");
        assert_eq!(h.sample_rate, 22050);
    }

    #[test]
    fn test_handler_v3_forces_preprocessing() {
        let h = SarvamTtsHandler::new(SarvamTtsConfig {
            api_key:              "test".to_string(),
            model:                "bulbul:v3".to_string(),
            enable_preprocessing: false,
            ..SarvamTtsConfig::default()
        })
        .unwrap();
        assert!(h.config.enable_preprocessing);
    }

    #[test]
    fn test_handler_v3_clears_pitch() {
        let h = SarvamTtsHandler::new(SarvamTtsConfig {
            api_key: "test".to_string(),
            model:   "bulbul:v3".to_string(),
            pitch:   Some(0.5),
            ..SarvamTtsConfig::default()
        })
        .unwrap();
        assert!(h.config.pitch.is_none());
    }

    #[test]
    fn test_handler_unsupported_model_errors() {
        let result = SarvamTtsHandler::new(SarvamTtsConfig {
            api_key: "test".to_string(),
            model:   "unknown:model".to_string(),
            ..SarvamTtsConfig::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_handler_pace_clamped_high() {
        let h = SarvamTtsHandler::new(SarvamTtsConfig {
            api_key: "test".to_string(),
            pace:    99.0,
            ..SarvamTtsConfig::default()
        })
        .unwrap();
        assert_eq!(h.config.pace, 3.0);
    }

    #[test]
    fn test_handler_pace_clamped_low() {
        let h = SarvamTtsHandler::new(SarvamTtsConfig {
            api_key: "test".to_string(),
            pace:    0.01,
            ..SarvamTtsConfig::default()
        })
        .unwrap();
        assert_eq!(h.config.pace, 0.3);
    }

    #[test]
    fn test_handler_v3_sample_rate() {
        let h = SarvamTtsHandler::new(SarvamTtsConfig {
            api_key: "test".to_string(),
            model:   "bulbul:v3".to_string(),
            ..SarvamTtsConfig::default()
        })
        .unwrap();
        assert_eq!(h.sample_rate, 24000);
    }

    #[test]
    fn test_handler_explicit_sample_rate_respected() {
        let h = SarvamTtsHandler::new(SarvamTtsConfig {
            api_key:     "test".to_string(),
            sample_rate: Some(16000),
            ..SarvamTtsConfig::default()
        })
        .unwrap();
        assert_eq!(h.sample_rate, 16000);
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
    async fn billing_done_emits_tts_usage_event() {
        let state   = state_with_chars(60);
        let mock    = MockCollector::new();
        let billing: Option<Arc<dyn crate::billing::BillingCollector>> = Some(mock.clone());

        handle_tts_message(r#"{"type":"done"}"#, &dummy_proc(), 22050, &state, &billing, "anushka").await;

        let evs = mock.events();
        assert_eq!(evs.len(), 1, "expected exactly one TtsUsage event");
        match &evs[0] {
            BillingEvent::TtsUsage { provider, char_count, voice, .. } => {
                assert_eq!(provider, "sarvam");
                assert_eq!(*char_count, 60);
                assert_eq!(voice, "anushka");
            }
            other => panic!("expected TtsUsage, got {:?}", other),
        }
        assert_eq!(state.lock().await.pending_chars, 0, "pending_chars must be reset after done");
    }

    #[tokio::test]
    async fn billing_done_with_zero_chars_emits_no_event() {
        let state   = state_with_chars(0);
        let mock    = MockCollector::new();
        let billing: Option<Arc<dyn crate::billing::BillingCollector>> = Some(mock.clone());

        handle_tts_message(r#"{"type":"done"}"#, &dummy_proc(), 22050, &state, &billing, "anushka").await;
        assert_eq!(mock.events().len(), 0);
    }

    #[tokio::test]
    async fn billing_no_collector_done_does_not_panic() {
        let state   = state_with_chars(30);
        let billing: Option<Arc<dyn crate::billing::BillingCollector>> = None;

        handle_tts_message(r#"{"type":"done"}"#, &dummy_proc(), 22050, &state, &billing, "anushka").await;
        assert_eq!(state.lock().await.pending_chars, 0);
    }

    #[tokio::test]
    async fn billing_multiple_done_signals_accumulate_chars_correctly() {
        // Each synthesis cycle: set pending_chars, receive "done", repeat
        let state   = state_with_chars(100);
        let mock    = MockCollector::new();
        let billing: Option<Arc<dyn crate::billing::BillingCollector>> = Some(mock.clone());

        handle_tts_message(r#"{"type":"done"}"#, &dummy_proc(), 22050, &state, &billing, "anushka").await;
        // Simulate second utterance
        state.lock().await.pending_chars = 200;
        handle_tts_message(r#"{"type":"done"}"#, &dummy_proc(), 22050, &state, &billing, "anushka").await;

        let evs = mock.events();
        assert_eq!(evs.len(), 2);
        let total: usize = evs.iter().filter_map(|e| match e {
            BillingEvent::TtsUsage { char_count, .. } => Some(*char_count),
            _ => None,
        }).sum();
        assert_eq!(total, 300);
    }

    #[test]
    fn with_billing_sets_field() {
        use crate::billing::NoopBillingCollector;
        let h = SarvamTtsHandler::new(SarvamTtsConfig {
            api_key: "key".into(),
            ..SarvamTtsConfig::default()
        }).unwrap().with_billing(Arc::new(NoopBillingCollector));
        assert!(h.billing.is_some());
    }
}