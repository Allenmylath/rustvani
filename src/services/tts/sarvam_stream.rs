//! Sarvam Bulbul over its streaming WebSocket, one utterance at a time with the audio arriving
//! as it is made: the first chunk a few hundred milliseconds after the text, the rest behind it.
//!
//! One connection per (language, voice) is kept open and reused (its handshake would otherwise
//! cost more than the streaming saves); utterances on it go one after another: `config` (the
//! pace, when it changes), `text`, `flush`, then `audio` messages until the server's final event.
//! A connection the server closed (it idles out after a minute) is reopened on the next
//! utterance. Streaming is limited to 24 kHz.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::{mpsc, Mutex, OwnedMutexGuard};
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::http::Request;
use tokio_tungstenite::tungstenite::Message;

use super::sarvam::{get_model_config, TtsModelConfig, MODEL_BULBUL_V3};
use super::sarvam_rest::SarvamRestTts;
use super::synth::{Pcm, PcmChunks, TtsSynthesizer};
use crate::error::{PipecatError, Result};

/// The most streaming synthesis delivers.
pub const MAX_STREAM_RATE: u32 = 24_000;
/// An utterance without its final event this long after the flush counts as lost.
const UTTERANCE_TIMEOUT: Duration = Duration::from_secs(20);

#[derive(Debug, Clone)]
pub struct SarvamStreamTtsConfig {
    pub api_key: String,
    /// `bulbul:v3` (default).
    pub model: String,
    pub url: String,
    /// `bulbul:v3`: 0.01–1.0, lower is steadier.
    pub temperature: Option<f32>,
}

impl Default for SarvamStreamTtsConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: "bulbul:v3".to_string(),
            url: "wss://api.sarvam.ai/text-to-speech/ws".to_string(),
            temperature: None,
        }
    }
}

/// What a connection is doing: the utterance being received, if any.
struct Current {
    chunks: mpsc::Sender<Result<Bytes>>,
    /// Held while an utterance is in flight: the next waits for it.
    _turn: OwnedMutexGuard<()>,
}

struct Connection {
    tx: mpsc::Sender<String>,
    current: Arc<std::sync::Mutex<Option<Current>>>,
    /// Utterances go one after another.
    turn: Arc<Mutex<()>>,
    pace: std::sync::Mutex<Option<f32>>,
    alive: Arc<std::sync::atomic::AtomicBool>,
    tasks: Vec<JoinHandle<()>>,
}

impl Drop for Connection {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
    }
}

pub struct SarvamStreamTts {
    config: SarvamStreamTtsConfig,
    model: &'static TtsModelConfig,
    connections: Mutex<HashMap<(String, String), Arc<Connection>>>,
}

#[derive(Deserialize)]
struct ServerMessage {
    #[serde(rename = "type")]
    kind: String,
    data: Option<serde_json::Value>,
}

impl SarvamStreamTts {
    pub fn new(config: SarvamStreamTtsConfig) -> Self {
        let model = get_model_config(&config.model).unwrap_or(&MODEL_BULBUL_V3);
        Self { config, model, connections: Mutex::new(HashMap::new()) }
    }

    pub fn config(&self) -> &SarvamStreamTtsConfig {
        &self.config
    }

    async fn connection(&self, code: &str, voice: &str) -> Result<Arc<Connection>> {
        let key = (code.to_string(), voice.to_string());
        let mut connections = self.connections.lock().await;
        if let Some(c) = connections.get(&key) {
            if c.alive.load(std::sync::atomic::Ordering::Relaxed) {
                return Ok(Arc::clone(c));
            }
            connections.remove(&key);
        }
        let c = Arc::new(self.connect().await?);
        connections.insert(key, Arc::clone(&c));
        Ok(c)
    }

    async fn connect(&self) -> Result<Connection> {
        let url = format!("{}?model={}&send_completion_event=true", self.config.url, urlencode(&self.config.model));
        let host = url.split('/').nth(2).unwrap_or("api.sarvam.ai").to_string();
        let request = Request::builder()
            .uri(&url)
            .header("Host", host)
            .header("api-subscription-key", &self.config.api_key)
            .header("Connection", "Upgrade")
            .header("Upgrade", "websocket")
            .header("Sec-WebSocket-Version", "13")
            .header("Sec-WebSocket-Key", tokio_tungstenite::tungstenite::handshake::client::generate_key())
            .body(())
            .map_err(|e| PipecatError::pipeline(format!("SarvamStreamTts: request: {e}")))?;
        let (ws, _) = tokio_tungstenite::connect_async(request)
            .await
            .map_err(|e| PipecatError::pipeline(format!("SarvamStreamTts: connect: {e}")))?;
        let (mut sink, mut stream) = ws.split();
        let (tx, mut rx) = mpsc::channel::<String>(64);
        let current: Arc<std::sync::Mutex<Option<Current>>> = Arc::default();
        let alive = Arc::new(std::sync::atomic::AtomicBool::new(true));

        let send_task = tokio::spawn(async move {
            while let Some(text) = rx.recv().await {
                if sink.send(Message::Text(text.into())).await.is_err() {
                    break;
                }
            }
        });
        let receive_task = tokio::spawn({
            let current = Arc::clone(&current);
            let alive = Arc::clone(&alive);
            async move {
                while let Some(message) = stream.next().await {
                    match message {
                        Ok(Message::Text(text)) => {
                            // After an error the server answers nothing more on this connection:
                            // retire it, so the next utterance opens a new one.
                            if Self::on_message(&text, &current) {
                                alive.store(false, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                        Ok(Message::Close(_)) | Err(_) => break,
                        _ => {}
                    }
                }
                alive.store(false, std::sync::atomic::Ordering::Relaxed);
                if let Some(c) = current.lock().unwrap().take() {
                    let _ = c.chunks.try_send(Err(PipecatError::pipeline("SarvamStreamTts: connection closed")));
                }
                log::info!("SarvamStreamTts: connection closed");
            }
        });
        let keepalive_task = tokio::spawn({
            let tx = tx.clone();
            async move {
                loop {
                    tokio::time::sleep(Duration::from_secs(20)).await;
                    if tx.send(json!({ "type": "ping" }).to_string()).await.is_err() {
                        break;
                    }
                }
            }
        });
        Ok(Connection {
            tx,
            current,
            turn: Arc::new(Mutex::new(())),
            pace: std::sync::Mutex::new(None),
            alive,
            tasks: vec![send_task, receive_task, keepalive_task],
        })
    }

    /// One server message: audio to the utterance in flight; its final event ends it. Whether it
    /// was an error, after which the connection is no use.
    fn on_message(text: &str, current: &std::sync::Mutex<Option<Current>>) -> bool {
        let Ok(message) = serde_json::from_str::<ServerMessage>(text) else { return false };
        match message.kind.as_str() {
            "audio" => {
                let Some(b64) = message.data.as_ref().and_then(|d| d["audio"].as_str()).filter(|s| !s.is_empty()) else { return false };
                let Ok(mut audio) = BASE64.decode(b64) else { return false };
                if audio.len() > 44 && audio.starts_with(b"RIFF") {
                    audio.drain(..44);
                }
                if audio.is_empty() {
                    return false;
                }
                if let Some(c) = current.lock().unwrap().as_ref() {
                    if let Err(mpsc::error::TrySendError::Full(_)) = c.chunks.try_send(Ok(Bytes::from(audio))) {
                        log::warn!("SarvamStreamTts: audio arriving faster than it is taken; a chunk was dropped");
                    }
                }
            }
            "event" | "done" => {
                let is_final = message.kind == "done"
                    || message.data.as_ref().and_then(|d| d["event_type"].as_str()) == Some("final");
                if is_final {
                    current.lock().unwrap().take(); // dropping the sender ends the chunks
                }
            }
            "error" => {
                let why = message.data.as_ref().and_then(|d| d["message"].as_str()).unwrap_or("error").to_string();
                if let Some(c) = current.lock().unwrap().take() {
                    let _ = c.chunks.try_send(Err(PipecatError::pipeline(format!("SarvamStreamTts: {why}"))));
                }
                return true;
            }
            other => log::debug!("SarvamStreamTts: unhandled message {other:?}"),
        }
        false
    }
}

fn urlencode(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
            other => other.to_string().bytes().map(|b| format!("%{b:02X}")).collect(),
        })
        .collect()
}

#[async_trait]
impl TtsSynthesizer for SarvamStreamTts {
    fn name(&self) -> &'static str {
        "sarvam"
    }

    fn model(&self) -> String {
        format!("{} (streaming)", self.config.model)
    }

    fn supports(&self, lang: &str) -> bool {
        SarvamRestTts::code_for(lang).is_some()
    }

    fn pace_range(&self) -> (f32, f32) {
        (self.model.pace_min, self.model.pace_max)
    }

    fn streams(&self) -> bool {
        true
    }

    async fn synthesize(&self, text: &str, lang: &str, voice: &str, pace: f32, sample_rate: u32) -> Result<Pcm> {
        let mut chunks = self.synthesize_streaming(text, lang, voice, pace, sample_rate).await?;
        let mut samples = Vec::new();
        while let Some(chunk) = chunks.chunks.recv().await {
            samples.extend_from_slice(&chunk?);
        }
        Ok(Pcm { samples: Bytes::from(samples), rate: chunks.rate, channels: chunks.channels })
    }

    async fn synthesize_streaming(&self, text: &str, lang: &str, voice: &str, pace: f32, sample_rate: u32) -> Result<PcmChunks> {
        let code = SarvamRestTts::code_for(lang)
            .ok_or_else(|| PipecatError::pipeline(format!("SarvamStreamTts: Bulbul does not speak {lang:?}")))?;
        let voice = if voice.is_empty() { self.model.default_speaker } else { voice };
        let pace = pace.clamp(self.model.pace_min, self.model.pace_max);
        let rate = sample_rate.min(MAX_STREAM_RATE);
        let text = text.trim();
        let (tx, rx) = mpsc::channel::<Result<Bytes>>(64);
        if text.is_empty() {
            drop(tx);
            return Ok(PcmChunks { rate, channels: 1, chunks: rx });
        }

        // One utterance at a time on a connection; a dead one is replaced.
        let mut attempt = 0;
        let connection = loop {
            attempt += 1;
            let c = self.connection(code, voice).await?;
            let turn = Arc::clone(&c.turn).lock_owned().await;
            if c.alive.load(std::sync::atomic::Ordering::Relaxed) {
                *c.current.lock().unwrap() = Some(Current { chunks: tx.clone(), _turn: turn });
                break c;
            }
            if attempt >= 2 {
                return Err(PipecatError::pipeline("SarvamStreamTts: no connection"));
            }
        };
        let configured = *connection.pace.lock().unwrap();
        if configured != Some(pace) {
            let mut config = json!({
                "target_language_code": code,
                "speaker": voice,
                "speech_sample_rate": rate.to_string(),
                "output_audio_codec": "linear16",
                "pace": pace,
                "model": self.config.model,
                "min_buffer_size": 30,
                "max_chunk_length": 500,
            });
            if let Some(t) = self.config.temperature {
                config["temperature"] = json!(t);
            }
            let sent = connection.tx.send(json!({ "type": "config", "data": config }).to_string()).await;
            if sent.is_err() {
                connection.current.lock().unwrap().take();
                return Err(PipecatError::pipeline("SarvamStreamTts: connection gone"));
            }
            *connection.pace.lock().unwrap() = Some(pace);
        }
        let sent = connection.tx.send(json!({ "type": "text", "data": { "text": text } }).to_string()).await;
        let flushed = connection.tx.send(json!({ "type": "flush" }).to_string()).await;
        if sent.is_err() || flushed.is_err() {
            connection.current.lock().unwrap().take();
            return Err(PipecatError::pipeline("SarvamStreamTts: connection gone"));
        }
        // An utterance the server never finishes would hold the connection: time it out. (A weak
        // sender, so this does not itself keep the chunks open.)
        tokio::spawn({
            let current = Arc::clone(&connection.current);
            let alive = Arc::clone(&connection.alive);
            let weak = tx.downgrade();
            async move {
                tokio::time::sleep(UTTERANCE_TIMEOUT).await;
                let Some(tx) = weak.upgrade() else { return };
                let stale = current.lock().unwrap().as_ref().is_some_and(|c| c.chunks.same_channel(&tx));
                if stale {
                    // A connection that dropped an utterance would drop the next too.
                    alive.store(false, std::sync::atomic::Ordering::Relaxed);
                    current.lock().unwrap().take();
                    let _ = tx.try_send(Err(PipecatError::pipeline("SarvamStreamTts: no final event")));
                }
            }
        });
        drop(tx);
        Ok(PcmChunks { rate, channels: 1, chunks: rx })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_feed_the_utterance_in_flight() {
        let (tx, mut rx) = mpsc::channel::<Result<Bytes>>(8);
        let turn = Arc::new(Mutex::new(()));
        let guard = Arc::clone(&turn).try_lock_owned().unwrap();
        let current = std::sync::Mutex::new(Some(Current { chunks: tx, _turn: guard }));
        let wav = BASE64.encode([b"RIFF".as_slice(), &[0u8; 40], &[1, 0, 2, 0]].concat());
        SarvamStreamTts::on_message(&json!({ "type": "audio", "data": { "audio": wav } }).to_string(), &current);
        SarvamStreamTts::on_message(&json!({ "type": "audio", "data": { "audio": BASE64.encode([3u8, 0]) } }).to_string(), &current);
        SarvamStreamTts::on_message(&json!({ "type": "event", "data": { "event_type": "final" } }).to_string(), &current);
        assert_eq!(rx.try_recv().unwrap().unwrap(), Bytes::from_static(&[1, 0, 2, 0]), "the WAV header is stripped");
        assert_eq!(rx.try_recv().unwrap().unwrap(), Bytes::from_static(&[3, 0]));
        assert!(rx.try_recv().is_err(), "the final event ended the utterance");
        assert!(current.lock().unwrap().is_none());
        assert!(turn.try_lock().is_ok(), "and freed the connection for the next");
        assert_eq!(urlencode("bulbul:v3"), "bulbul%3Av3");
    }

    /// Against the real API: `SARVAM_API_KEY=… cargo test --features tts-sarvam sarvam_stream_tts_live -- --ignored --nocapture`.
    #[tokio::test]
    #[ignore]
    async fn sarvam_stream_tts_live() {
        let Ok(api_key) = std::env::var("SARVAM_API_KEY") else {
            eprintln!("SARVAM_API_KEY not set");
            return;
        };
        let tts = SarvamStreamTts::new(SarvamStreamTtsConfig { api_key, ..Default::default() });
        for (lang, text) in [("hi", "नमस्ते, आज मौसम बहुत अच्छा है।"), ("ta", "இன்று காலை நல்ல வானிலை."), ("hi", "यह दूसरा वाक्य है, उसी कनेक्शन पर।")] {
            let t0 = std::time::Instant::now();
            let mut chunks = tts.synthesize_streaming(text, lang, "", 1.0, 48_000).await.unwrap();
            let mut first = None;
            let mut total = 0usize;
            let mut n = 0;
            while let Some(chunk) = chunks.chunks.recv().await {
                let chunk = chunk.unwrap();
                first.get_or_insert_with(|| t0.elapsed());
                total += chunk.len();
                n += 1;
            }
            let secs = total as f64 / 2.0 / chunks.rate as f64;
            println!("{lang}: first audio after {:.2} s, {n} chunks, {secs:.2} s of audio at {} Hz, all in {:.2} s", first.unwrap().as_secs_f64(), chunks.rate, t0.elapsed().as_secs_f64());
            assert_eq!(chunks.rate, 24_000);
            assert!(secs > 0.5);
        }
    }
}
