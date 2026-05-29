//! 60db Speech-to-Text WebSocket service.
//!
//! Connects to 60db's real-time STT WebSocket API and pushes
//! TranscriptionFrames downstream when transcripts arrive.
//!
//! Pipeline position:
//!   transport.input() → SixtyDbSttHandler → llm → tts → transport.output()
//!
//! Wiring:
//!   let stt = SixtyDbSttHandler::new(SixtyDbSttConfig {
//!       api_key: std::env::var("SIXTYDB_API_KEY").unwrap(),
//!       ..Default::default()
//!   })
//!   .into_processor();
//!
//! Frames consumed:
//!   - StartFrame             → connects WebSocket
//!   - InputAudioRaw          → denoise → resample → encode → send to 60db
//!   - EndFrame / CancelFrame → disconnects WebSocket
//!
//! Frames produced:
//!   - TranscriptionFrame (downstream) on transcript
//!   - UserStartedSpeaking (downstream) on speech_started (barge-in)
//!   - UserStoppedSpeaking (downstream) on canonical final
//!   - ErrorFrame (upstream) on connection / parse errors
//!
//! Auth: apiKey query parameter.
//! URL:  wss://api.60db.ai/ws/stt?apiKey=...

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use futures::{SinkExt, StreamExt};
use log;
use serde::Serialize;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::http::Request;
use tokio_tungstenite::tungstenite::Message;

use crate::audio_process::noisefilter::RNNoiseFilter;
use crate::audio_process::resamplers::{ResamplerQuality, StreamResampler};
use crate::error::Result;
use crate::frames::{
    ControlFrame, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor,
    SystemFrame, TranscriptionData,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SIXTYDB_BASE_WSS: &str = "wss://api.60db.ai/ws/stt";
const SIXTYDB_BASE_WS: &str = "ws://api.60db.ai/ws/stt";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Audio encoding for the 60db STT session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SixtyDbEncoding {
    /// 16-bit little-endian PCM (browser mode). Sent as JSON base64.
    Linear,
    /// G.711 μ-law (telephony mode). Sent as binary WS frames.
    Mulaw,
}

impl Default for SixtyDbEncoding {
    fn default() -> Self {
        Self::Linear
    }
}

impl SixtyDbEncoding {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Mulaw => "mulaw",
        }
    }
}

/// Real-time audio enhancement level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SixtyDbAudioEnhancement {
    Off,
    Light,
    Adaptive,
}

impl Default for SixtyDbAudioEnhancement {
    fn default() -> Self {
        Self::Off
    }
}

impl SixtyDbAudioEnhancement {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Light => "light",
            Self::Adaptive => "adaptive",
        }
    }
}

/// Context hint for LLM refinement.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SixtyDbContext {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub general: Vec<SixtyDbContextItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub terms: Vec<String>,
}

/// A single key/value context hint.
#[derive(Debug, Clone, Serialize)]
pub struct SixtyDbContextItem {
    pub key: String,
    pub value: String,
}

/// Configuration for SixtyDbSttHandler.
#[derive(Debug, Clone)]
pub struct SixtyDbSttConfig {
    /// 60db API key.
    pub api_key: String,

    /// Language codes e.g. `["en"]`, `["en", "hi"]`.
    /// Empty vec or omitted = auto-detect.
    pub languages: Vec<String>,

    /// Optional context for LLM refinement.
    pub context: Option<SixtyDbContext>,

    /// Audio encoding.
    pub encoding: SixtyDbEncoding,

    /// Audio sample rate sent to the server.
    /// If the pipeline input rate differs, the handler resamples automatically.
    pub sample_rate: u32,

    /// Silence duration (ms) after last speech chunk before finalizing.
    /// Values below 300 ms are silently clamped to 300 ms by the server.
    pub utterance_end_ms: u32,

    /// Keep session alive between utterances.
    pub continuous_mode: bool,

    /// How often (ms) to emit interim partial results.
    /// Values below 300 ms are clamped to 300 ms.
    /// `None` = disabled.
    pub interim_results_frequency: Option<u32>,

    /// Audio enhancement level.
    pub audio_enhancement: SixtyDbAudioEnhancement,

    /// Enable speaker diarization.
    pub diarize: bool,

    /// Lower bound on speaker count (only when diarize=true).
    pub min_speakers: Option<i32>,

    /// Upper bound on speaker count (only when diarize=true).
    pub max_speakers: Option<i32>,

    /// Use `ws://` instead of `wss://`.
    pub insecure: bool,

    /// Enable RNNoise noise suppression before sending audio to 60db.
    pub noise_reduction: bool,

    /// Quality preset for the internal sample-rate converter.
    /// Only used when the incoming `InputAudioRaw` rate differs from `sample_rate`.
    pub resampler_quality: ResamplerQuality,
}

impl Default for SixtyDbSttConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            languages: vec!["en".to_string()],
            context: None,
            encoding: SixtyDbEncoding::Linear,
            sample_rate: 16_000,
            utterance_end_ms: 500,
            continuous_mode: true,
            interim_results_frequency: Some(300),
            audio_enhancement: SixtyDbAudioEnhancement::Off,
            diarize: false,
            min_speakers: None,
            max_speakers: None,
            insecure: false,
            noise_reduction: true,
            resampler_quality: ResamplerQuality::Quick,
        }
    }
}

impl SixtyDbSttConfig {
    fn ws_url(&self) -> String {
        let base = if self.insecure {
            SIXTYDB_BASE_WS
        } else {
            SIXTYDB_BASE_WSS
        };
        format!("{}?apiKey={}", base, urlencoding(&self.api_key))
    }

    fn start_message(&self) -> String {
        #[derive(Serialize)]
        struct StartMsg {
            #[serde(rename = "type")]
            msg_type: &'static str,
            #[serde(skip_serializing_if = "Option::is_none")]
            languages: Option<Vec<String>>,
            #[serde(skip_serializing_if = "Option::is_none")]
            context: Option<SixtyDbContext>,
            config: Config,
        }

        #[derive(Serialize)]
        struct Config {
            encoding: &'static str,
            sample_rate: u32,
            utterance_end_ms: u32,
            continuous_mode: bool,
            #[serde(skip_serializing_if = "Option::is_none")]
            interim_results_frequency: Option<u32>,
            audio_enhancement: &'static str,
            diarize: bool,
            #[serde(skip_serializing_if = "Option::is_none")]
            min_speakers: Option<i32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            max_speakers: Option<i32>,
        }

        let languages = if self.languages.is_empty() {
            None
        } else {
            Some(self.languages.clone())
        };

        let msg = StartMsg {
            msg_type: "start",
            languages,
            context: self.context.clone(),
            config: Config {
                encoding: self.encoding.as_str(),
                sample_rate: self.sample_rate,
                utterance_end_ms: self.utterance_end_ms,
                continuous_mode: self.continuous_mode,
                interim_results_frequency: self.interim_results_frequency,
                audio_enhancement: self.audio_enhancement.as_str(),
                diarize: self.diarize,
                min_speakers: self.min_speakers,
                max_speakers: self.max_speakers,
            },
        };

        serde_json::to_string(&msg).unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Connection state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WsState {
    Disconnected,
    WsConnected,
    ConnectionEstablished,
    SessionStarted,
    Stopping,
    Stopped,
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

struct SixtyDbSttState {
    ws_tx: Option<mpsc::Sender<Message>>,
    send_task: Option<JoinHandle<()>>,
    receive_task: Option<JoinHandle<()>>,
    ws_state: WsState,
    /// Audio buffered before `session_started` arrives.
    audio_buffer: Vec<Vec<u8>>,
    /// Sample rate observed from the first `InputAudioRaw` frame.
    input_sample_rate: Option<u32>,
}

impl SixtyDbSttState {
    fn new() -> Self {
        Self {
            ws_tx: None,
            send_task: None,
            receive_task: None,
            ws_state: WsState::Disconnected,
            audio_buffer: Vec::new(),
            input_sample_rate: None,
        }
    }
}

// ---------------------------------------------------------------------------
// SixtyDbSttHandler
// ---------------------------------------------------------------------------

pub struct SixtyDbSttHandler {
    config: SixtyDbSttConfig,
    state: Arc<Mutex<SixtyDbSttState>>,
    /// Lazily initialised on first `InputAudioRaw` with the actual input rate.
    noise_filter: Arc<Mutex<Option<RNNoiseFilter>>>,
    /// Lazily initialised when incoming rate differs from `config.sample_rate`.
    resampler: Arc<Mutex<Option<StreamResampler>>>,
}

impl SixtyDbSttHandler {
    pub fn new(config: SixtyDbSttConfig) -> Self {
        Self {
            config,
            state: Arc::new(Mutex::new(SixtyDbSttState::new())),
            noise_filter: Arc::new(Mutex::new(None)),
            resampler: Arc::new(Mutex::new(None)),
        }
    }

    pub fn into_processor(self) -> FrameProcessor {
        FrameProcessor::new("SixtyDbStt", Box::new(self), false)
    }
}

// ---------------------------------------------------------------------------
// Connection / disconnection
// ---------------------------------------------------------------------------

impl SixtyDbSttHandler {
    async fn connect(&self, processor: FrameProcessor) {
        let url = self.config.ws_url();
        log::info!("SixtyDbStt: connecting to {}", url);

        let request = match Request::builder()
            .uri(&url)
            .header("Host", "api.60db.ai")
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
                    .push_error(format!("SixtyDbStt: request build failed: {}", e), false)
                    .await;
                return;
            }
        };

        let ws_stream = match tokio_tungstenite::connect_async(request).await {
            Ok((stream, _)) => stream,
            Err(e) => {
                let _ = processor
                    .push_error(format!("SixtyDbStt: connect failed: {}", e), false)
                    .await;
                return;
            }
        };

        let (sink, stream) = ws_stream.split();
        let (ws_tx, ws_rx) = mpsc::channel::<Message>(64);

        let send_task = tokio::spawn(run_send_task(sink, ws_rx));
        let config_clone = self.config.clone();
        let state_clone = self.state.clone();
        let receive_task = tokio::spawn(run_receive_task(
            stream,
            processor,
            config_clone,
            state_clone,
        ));

        let mut state = self.state.lock().await;
        state.ws_tx = Some(ws_tx);
        state.send_task = Some(send_task);
        state.receive_task = Some(receive_task);
        state.ws_state = WsState::WsConnected;
        state.audio_buffer.clear();
        state.input_sample_rate = None;

        log::info!("SixtyDbStt: WebSocket connected, awaiting handshake");
    }

    async fn disconnect(&self) {
        let mut state = self.state.lock().await;
        state.ws_state = WsState::Stopping;
        if let Some(h) = state.receive_task.take() {
            h.abort();
        }
        if let Some(h) = state.send_task.take() {
            h.abort();
        }
        state.ws_tx = None;
        state.audio_buffer.clear();
        state.input_sample_rate = None;
        drop(state);

        *self.noise_filter.lock().await = None;
        *self.resampler.lock().await = None;

        log::info!("SixtyDbStt: disconnected");
    }

    async fn send_ws_message(&self, msg: Message) {
        let tx = { self.state.lock().await.ws_tx.clone() };
        if let Some(tx) = tx {
            let _ = tx.send(msg).await;
        }
    }

    async fn send_json(&self, json: String) {
        self.send_ws_message(Message::Text(json.into())).await;
    }

    /// Send audio to 60db.
    /// Binary mode → raw μ-law bytes as WS binary frame.
    /// JSON mode → base64-encoded Int16 PCM wrapped in JSON.
    async fn send_audio(&self, audio: &[u8]) {
        let mut state = self.state.lock().await;

        match state.ws_state {
            WsState::SessionStarted => {
                // Flush any buffered audio first
                let buffered: Vec<Vec<u8>> = state.audio_buffer.drain(..).collect();
                drop(state);

                for chunk in buffered {
                    self.deliver_audio(&chunk).await;
                }
                self.deliver_audio(audio).await;
            }
            WsState::WsConnected | WsState::ConnectionEstablished => {
                state.audio_buffer.push(audio.to_vec());
            }
            _ => {}
        }
    }

    async fn deliver_audio(&self, audio: &[u8]) {
        match self.config.encoding {
            SixtyDbEncoding::Mulaw => {
                self.send_ws_message(Message::Binary(audio.to_vec().into())).await;
            }
            SixtyDbEncoding::Linear => {
                let msg = serde_json::json!({
                    "type": "audio",
                    "audio": BASE64.encode(audio),
                    "encoding": "linear",
                    "sample_rate": self.config.sample_rate,
                    "timestamp": unix_ms(),
                });
                self.send_json(msg.to_string()).await;
            }
        }
    }

    async fn send_stop(&self) {
        self.send_json(r#"{"type":"stop"}"#.to_string()).await;
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

fn i16_to_f32(samples: &[i16]) -> Vec<f32> {
    samples.iter().map(|&s| s as f32).collect()
}

fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&s| s.clamp(i16::MIN as f32, i16::MAX as f32) as i16)
        .collect()
}

// ---------------------------------------------------------------------------
// Audio pipeline helpers
// ---------------------------------------------------------------------------

impl SixtyDbSttHandler {
    /// Denoise (if enabled) → resample (if needed) → send to WebSocket.
    async fn prepare_and_send(&self, pcm: &[i16], sample_rate: u32, denoise: bool) {
        // 1. Optional denoising — lazily initialised with the actual input rate.
        let denoised = if denoise && self.config.noise_reduction {
            let mut nf_guard = self.noise_filter.lock().await;
            if nf_guard.is_none() {
                log::info!(
                    "SixtyDbStt: noise reduction enabled (input_rate={})",
                    sample_rate
                );
                *nf_guard = Some(RNNoiseFilter::new(sample_rate));
            }
            nf_guard.as_mut().unwrap().filter(pcm)
        } else {
            pcm.to_vec()
        };

        if denoised.is_empty() {
            return;
        }

        // Track input rate for downstream tail-flush logic.
        {
            let mut state = self.state.lock().await;
            state.input_sample_rate = Some(sample_rate);
        }

        // 2. Optional resampling — lazily initialised when rates diverge.
        let resampled = if sample_rate != self.config.sample_rate {
            let mut r_guard = self.resampler.lock().await;
            if r_guard.is_none() {
                log::info!(
                    "SixtyDbStt: resampling {} → {} Hz",
                    sample_rate,
                    self.config.sample_rate
                );
                *r_guard = Some(StreamResampler::new(
                    sample_rate,
                    self.config.sample_rate,
                    self.config.resampler_quality,
                ));
            }
            let f32_samples = i16_to_f32(&denoised);
            let resampled_f32 = r_guard.as_mut().unwrap().process(&f32_samples);
            f32_to_i16(&resampled_f32)
        } else {
            denoised
        };

        if !resampled.is_empty() {
            self.send_audio(&i16_to_bytes(&resampled)).await;
        }
    }
}

// ---------------------------------------------------------------------------
// FrameHandler impl
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameHandler for SixtyDbSttHandler {
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

                let pcm = bytes_to_i16(&audio.audio);
                self.prepare_and_send(&pcm, audio.sample_rate, true).await;
            }

            FrameInner::System(SystemFrame::VADUserStoppedSpeaking { .. }) => {
                processor.push_frame(frame, direction).await?;

                // Flush noise filter tail
                if self.config.noise_reduction {
                    let tail = {
                        let mut nf_guard = self.noise_filter.lock().await;
                        nf_guard.as_mut().map(|nf| nf.flush()).unwrap_or_default()
                    };
                    if !tail.is_empty() {
                        let sample_rate = {
                            let state = self.state.lock().await;
                            state.input_sample_rate.unwrap_or(self.config.sample_rate)
                        };
                        self.prepare_and_send(&tail, sample_rate, false).await;
                    }
                }
            }

            FrameInner::Control(ControlFrame::End { .. })
            | FrameInner::System(SystemFrame::Cancel { .. }) => {
                let sample_rate = {
                    let state = self.state.lock().await;
                    state.input_sample_rate.unwrap_or(self.config.sample_rate)
                };

                // Flush noise filter tail
                if self.config.noise_reduction {
                    let tail = {
                        let mut nf_guard = self.noise_filter.lock().await;
                        nf_guard.as_mut().map(|nf| nf.flush()).unwrap_or_default()
                    };
                    if !tail.is_empty() {
                        self.prepare_and_send(&tail, sample_rate, false).await;
                    }
                }

                // Flush resampler tail
                let resampler_tail = {
                    let mut r_guard = self.resampler.lock().await;
                    r_guard.as_mut().map(|r| r.flush()).unwrap_or_default()
                };
                if !resampler_tail.is_empty() {
                    let i16_tail = f32_to_i16(&resampler_tail);
                    self.send_audio(&i16_to_bytes(&i16_tail)).await;
                }

                self.send_stop().await;
                // Give the server a moment to process remaining buffer,
                // then disconnect.
                tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
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

async fn run_send_task(mut sink: WsSink, mut rx: mpsc::Receiver<Message>) {
    while let Some(msg) = rx.recv().await {
        if sink.send(msg).await.is_err() {
            log::warn!("SixtyDbStt: send failed — closing send task");
            break;
        }
    }
    let _ = sink.close().await;
    log::debug!("SixtyDbStt: send task exited");
}

async fn run_receive_task(
    mut stream: WsStream,
    processor: FrameProcessor,
    config: SixtyDbSttConfig,
    shared_state: Arc<Mutex<SixtyDbSttState>>,
) {
    log::debug!("SixtyDbStt: receive task started");

    while let Some(result) = stream.next().await {
        match result {
            Ok(Message::Text(text)) => {
                handle_text_message(text.as_str(), &processor, &config, &shared_state).await;
            }
            Ok(Message::Close(_)) => {
                log::info!("SixtyDbStt: server closed WebSocket");
                break;
            }
            Err(e) => {
                let _ = processor
                    .push_error(format!("SixtyDbStt: receive error: {}", e), false)
                    .await;
                break;
            }
            _ => {}
        }
    }

    log::debug!("SixtyDbStt: receive task exited");
}

// ---------------------------------------------------------------------------
// Message handling
// ---------------------------------------------------------------------------

async fn handle_text_message(
    text: &str,
    processor: &FrameProcessor,
    config: &SixtyDbSttConfig,
    shared_state: &Arc<Mutex<SixtyDbSttState>>,
) {
    log::trace!("SixtyDbStt: raw message: {}", text);

    let val: serde_json::Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(e) => {
            log::warn!("SixtyDbStt: JSON parse error: {} — raw: {}", e, text);
            return;
        }
    };

    let obj = match val.as_object() {
        Some(o) => o,
        None => return,
    };

    // `connecting` and `connection_established` have no `type` field.
    if obj.contains_key("connecting") {
        let msg = obj.get("message").and_then(|v| v.as_str()).unwrap_or("");
        log::info!("SixtyDbStt: connecting — {}", msg);
        return;
    }

    if obj.contains_key("connection_established") {
        log::info!("SixtyDbStt: connection_established — sending start");
        let start_json = config.start_message();
        {
            let mut state = shared_state.lock().await;
            if let Some(ref tx) = state.ws_tx {
                let _ = tx.send(Message::Text(start_json.into())).await;
            }
            state.ws_state = WsState::ConnectionEstablished;
        }
        return;
    }

    let msg_type = match obj.get("type").and_then(|v| v.as_str()) {
        Some(t) => t,
        None => {
            log::debug!("SixtyDbStt: unknown message shape: {}", text);
            return;
        }
    };

    match msg_type {
        "connected" => {
            log::info!("SixtyDbStt: proxy connected to upstream STT");
        }

        "session_started" => {
            log::info!("SixtyDbStt: session_started — audio streaming enabled");
            let mut state = shared_state.lock().await;
            state.ws_state = WsState::SessionStarted;
            let buffered: Vec<Vec<u8>> = state.audio_buffer.drain(..).collect();
            let tx = state.ws_tx.clone();
            drop(state);

            // Flush buffered audio through ws_tx
            if let Some(tx) = tx {
                for chunk in buffered {
                    let msg = match config.encoding {
                        SixtyDbEncoding::Mulaw => Message::Binary(chunk.into()),
                        SixtyDbEncoding::Linear => {
                            let json = serde_json::json!({
                                "type": "audio",
                                "audio": BASE64.encode(&chunk),
                                "encoding": "linear",
                                "sample_rate": config.sample_rate,
                                "timestamp": unix_ms(),
                            });
                            Message::Text(json.to_string().into())
                        }
                    };
                    let _ = tx.send(msg).await;
                }
            }
        }

        "speech_started" => {
            log::debug!("SixtyDbStt: speech_started — barge-in");
            let _ = processor
                .push_frame(Frame::user_started_speaking(), FrameDirection::Downstream)
                .await;
        }

        "transcription" => {
            handle_transcription(obj, processor).await;
        }

        "language_changed" => {
            let lang = obj.get("language").and_then(|v| v.as_str()).unwrap_or("unknown");
            log::info!("SixtyDbStt: language_changed — {}", lang);
        }

        "mode_changed" => {
            let mode = obj.get("mode_name").and_then(|v| v.as_str()).unwrap_or("unknown");
            log::info!("SixtyDbStt: mode_changed — {}", mode);
        }

        "session_stopped" => {
            log::info!("SixtyDbStt: session_stopped");
            if let Some(summary) = obj.get("billing_summary") {
                log::info!("SixtyDbStt: billing_summary: {}", summary);
            }
            let mut state = shared_state.lock().await;
            state.ws_state = WsState::Stopped;
        }

        "error" => {
            let error = obj.get("error").and_then(|v| v.as_str()).unwrap_or("unknown error");
            let error_code = obj.get("error_code").and_then(|v| v.as_str());
            log::warn!(
                "SixtyDbStt: server error: {} (code: {:?})",
                error,
                error_code
            );
            let _ = processor
                .push_error(format!("SixtyDbStt: {} (code: {:?})", error, error_code), false)
                .await;
        }

        "test_response" => {
            log::trace!("SixtyDbStt: test_response");
        }

        other => {
            log::debug!("SixtyDbStt: unhandled message type: {}", other);
        }
    }
}

async fn handle_transcription(
    obj: &serde_json::Map<String, serde_json::Value>,
    processor: &FrameProcessor,
) {
    let text = match obj.get("text").and_then(|v| v.as_str()) {
        Some(t) => t.to_string(),
        None => return,
    };

    let is_final = obj.get("is_final").and_then(|v| v.as_bool()).unwrap_or(false);
    let speech_final = obj.get("speech_final").and_then(|v| v.as_bool()).unwrap_or(false);
    let is_partial = obj.get("is_partial").and_then(|v| v.as_bool()).unwrap_or(false);

    // Skip interim partials — they are only for barge-in word-count checks.
    // Never send interim text to the LLM.
    if is_partial && !is_final {
        log::trace!("SixtyDbStt: interim — '{}'", text);
        return;
    }

    // When LLM refinement is active, we get a first emit (is_final=true, speech_final=false).
    // We emit it as a non-finalized transcription for fast UI paint / barge-in.
    // The canonical (speech_final=true) follows shortly and is emitted as finalized.
    let finalized = is_final && speech_final;

    // Empty speech_final signal — reset state, do not treat as error.
    if text.is_empty() && finalized {
        let processing_mode = obj
            .get("processing_mode")
            .and_then(|v| v.as_str())
            .unwrap_or("speech_end_no_result");
        log::debug!(
            "SixtyDbStt: empty final (mode={}) — resetting state",
            processing_mode
        );
        return;
    }

    let language = obj
        .get("language")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let confidence = obj.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.0);

    let mut frame_data = TranscriptionData::new(text.clone(), "", time_now());
    frame_data.language = language;
    frame_data.finalized = finalized;

    log::info!(
        "SixtyDbStt: transcript='{}' is_final={} speech_final={} finalized={} confidence={:.2} lang={:?}",
        frame_data.text,
        is_final,
        speech_final,
        finalized,
        confidence,
        frame_data.language
    );

    let _ = processor
        .push_frame(Frame::transcription(frame_data), FrameDirection::Downstream)
        .await;

    // Emit UserStoppedSpeaking on canonical final to signal end of utterance.
    if finalized {
        let _ = processor
            .push_frame(Frame::user_stopped_speaking(), FrameDirection::Downstream)
            .await;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn time_now() -> String {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}.{:03}", d.as_secs(), d.subsec_millis())
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn urlencoding(s: &str) -> String {
    s.chars()
        .flat_map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => vec![c],
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
    use crate::frames::{AudioRawData, DataFrame, ErrorFrameData, FrameInner, StartFrameData};
    use crate::FrameProcessor;
    use crate::PassthroughHandler;

    // ========================================================================
    // Helpers
    // ========================================================================

    fn default_config() -> SixtyDbSttConfig {
        SixtyDbSttConfig {
            api_key: "test_key".to_string(),
            ..Default::default()
        }
    }

    async fn started_processor() -> FrameProcessor {
        let proc = FrameProcessor::new("test", Box::new(PassthroughHandler), false);
        proc.process_frame(Frame::start(StartFrameData::default()), FrameDirection::Downstream)
            .await
            .unwrap();
        proc
    }

    fn capture_pushes(proc: &FrameProcessor) -> Arc<std::sync::Mutex<Vec<Frame>>> {
        let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
        let cap = captured.clone();
        proc.on_after_push_frame(move |f| {
            cap.lock().unwrap().push(f.clone());
        });
        captured
    }

    fn capture_errors(proc: &FrameProcessor) -> Arc<std::sync::Mutex<Vec<ErrorFrameData>>> {
        let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
        let cap = captured.clone();
        proc.on_error(move |e| {
            cap.lock().unwrap().push(e.clone());
        });
        captured
    }

    async fn handler_with_state(
        config: SixtyDbSttConfig,
        ws_state: WsState,
    ) -> (SixtyDbSttHandler, mpsc::Receiver<Message>) {
        let handler = SixtyDbSttHandler::new(config);
        let (tx, rx) = mpsc::channel::<Message>(64);
        {
            let mut state = handler.state.lock().await;
            state.ws_tx = Some(tx);
            state.ws_state = ws_state;
        }
        (handler, rx)
    }

    // ========================================================================
    // Config & serialization
    // ========================================================================

    #[test]
    fn test_default_config() {
        let cfg = SixtyDbSttConfig::default();
        assert_eq!(cfg.api_key, "");
        assert_eq!(cfg.languages, vec!["en"]);
        assert_eq!(cfg.encoding, SixtyDbEncoding::Linear);
        assert_eq!(cfg.sample_rate, 16_000);
        assert_eq!(cfg.utterance_end_ms, 500);
        assert!(cfg.continuous_mode);
        assert_eq!(cfg.interim_results_frequency, Some(300));
        assert_eq!(cfg.audio_enhancement, SixtyDbAudioEnhancement::Off);
        assert!(!cfg.diarize);
        assert!(!cfg.insecure);
        assert!(cfg.noise_reduction);
        assert!(matches!(cfg.resampler_quality, ResamplerQuality::Quick));
    }

    #[test]
    fn test_ws_url_secure() {
        let cfg = SixtyDbSttConfig {
            api_key: "sk_test_123".to_string(),
            ..Default::default()
        };
        assert_eq!(cfg.ws_url(), "wss://api.60db.ai/ws/stt?apiKey=sk_test_123");
    }

    #[test]
    fn test_ws_url_insecure() {
        let cfg = SixtyDbSttConfig {
            api_key: "sk_test_123".to_string(),
            insecure: true,
            ..Default::default()
        };
        assert_eq!(cfg.ws_url(), "ws://api.60db.ai/ws/stt?apiKey=sk_test_123");
    }

    #[test]
    fn test_start_message_json() {
        let cfg = SixtyDbSttConfig {
            api_key: "key".to_string(),
            languages: vec!["en".to_string(), "hi".to_string()],
            encoding: SixtyDbEncoding::Linear,
            sample_rate: 48_000,
            utterance_end_ms: 500,
            continuous_mode: true,
            interim_results_frequency: Some(300),
            audio_enhancement: SixtyDbAudioEnhancement::Adaptive,
            diarize: true,
            min_speakers: Some(2),
            max_speakers: Some(4),
            ..Default::default()
        };

        let json = cfg.start_message();
        let val: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(val["type"], "start");
        assert_eq!(val["languages"], serde_json::json!(["en", "hi"]));
        assert_eq!(val["config"]["encoding"], "linear");
        assert_eq!(val["config"]["sample_rate"], 48_000);
        assert_eq!(val["config"]["utterance_end_ms"], 500);
        assert_eq!(val["config"]["continuous_mode"], true);
        assert_eq!(val["config"]["interim_results_frequency"], 300);
        assert_eq!(val["config"]["audio_enhancement"], "adaptive");
        assert_eq!(val["config"]["diarize"], true);
        assert_eq!(val["config"]["min_speakers"], 2);
        assert_eq!(val["config"]["max_speakers"], 4);
    }

    #[test]
    fn test_start_message_auto_languages() {
        let cfg = SixtyDbSttConfig {
            api_key: "key".to_string(),
            languages: vec![],
            ..Default::default()
        };
        let json = cfg.start_message();
        let val: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(val["languages"].is_null());
    }

    #[test]
    fn test_start_message_with_context() {
        let cfg = SixtyDbSttConfig {
            api_key: "key".to_string(),
            context: Some(SixtyDbContext {
                general: vec![
                    SixtyDbContextItem {
                        key: "domain".to_string(),
                        value: "Healthcare".to_string(),
                    },
                ],
                text: Some("Routine check-up.".to_string()),
                terms: vec!["Metformin".to_string()],
            }),
            ..Default::default()
        };
        let json = cfg.start_message();
        let val: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(val["context"]["general"][0]["key"], "domain");
        assert_eq!(val["context"]["general"][0]["value"], "Healthcare");
        assert_eq!(val["context"]["text"], "Routine check-up.");
        assert_eq!(val["context"]["terms"][0], "Metformin");
    }

    #[test]
    fn test_urlencoding() {
        assert_eq!(urlencoding("hello world"), "hello%20world");
        assert_eq!(urlencoding("a+b"), "a%2Bb");
    }

    // ========================================================================
    // Message parsing — connection handshake
    // ========================================================================

    #[tokio::test]
    async fn test_handle_connecting_logs_only() {
        let config = default_config();
        let proc = started_processor().await;
        let state = Arc::new(Mutex::new(SixtyDbSttState::new()));
        let text = r#"{"connecting":true,"message":"Authenticating...","timestamp":1234}"#;
        handle_text_message(text, &proc, &config, &state).await;
        assert!(matches!(state.lock().await.ws_state, WsState::Disconnected));
    }

    #[tokio::test]
    async fn test_handle_connection_established_sends_start() {
        let config = default_config();
        let proc = started_processor().await;
        let (tx, mut rx) = mpsc::channel::<Message>(64);
        let mut s = SixtyDbSttState::new();
        s.ws_tx = Some(tx);
        let state = Arc::new(Mutex::new(s));

        let text = r#"{"connection_established":{"service":"stt","user_id":1,"credit_balance":10.0,"workspace":"default"}}"#;
        handle_text_message(text, &proc, &config, &state).await;

        let msg = rx.recv().await.expect("start message should be sent");
        if let Message::Text(t) = msg {
            let val: serde_json::Value = serde_json::from_str(&t).unwrap();
            assert_eq!(val["type"], "start");
            assert_eq!(val["config"]["sample_rate"], 16_000);
        } else {
            panic!("expected text message, got {:?}", msg);
        }
        assert!(matches!(state.lock().await.ws_state, WsState::ConnectionEstablished));
    }

    #[tokio::test]
    async fn test_handle_connected_logs_only() {
        let config = default_config();
        let proc = started_processor().await;
        let state = Arc::new(Mutex::new(SixtyDbSttState::new()));
        let text = r#"{"type":"connected","server_info":{"server_type":"60db STT","ready":true}}"#;
        handle_text_message(text, &proc, &config, &state).await;
        assert!(state.lock().await.ws_tx.is_none());
    }

    #[tokio::test]
    async fn test_handle_session_started_flushes_buffered_audio() {
        let config = default_config();
        let proc = started_processor().await;
        let (tx, mut rx) = mpsc::channel::<Message>(64);
        let mut s = SixtyDbSttState::new();
        s.ws_tx = Some(tx);
        s.ws_state = WsState::ConnectionEstablished;
        s.audio_buffer.push(vec![0xAB, 0xCD, 0xEF, 0x01]);
        let state = Arc::new(Mutex::new(s));

        let text = r#"{"type":"session_started","session_id":"sess_123","language":"EN"}"#;
        handle_text_message(text, &proc, &config, &state).await;

        assert!(matches!(state.lock().await.ws_state, WsState::SessionStarted));
        assert!(state.lock().await.audio_buffer.is_empty());

        let msg = rx.recv().await.expect("buffered audio should be flushed");
        match msg {
            Message::Text(t) => {
                let val: serde_json::Value = serde_json::from_str(&t).unwrap();
                assert_eq!(val["type"], "audio");
            }
            other => panic!("expected text message, got {:?}", other),
        }
    }

    // ========================================================================
    // Message parsing — transcription & speech events
    // ========================================================================

    #[tokio::test]
    async fn test_handle_speech_started_pushes_user_started_speaking() {
        let config = default_config();
        let proc = started_processor().await;
        let captured = capture_pushes(&proc);
        let state = Arc::new(Mutex::new(SixtyDbSttState::new()));

        let text = r#"{"type":"speech_started","timestamp":1700000000.123}"#;
        handle_text_message(text, &proc, &config, &state).await;

        let frames = captured.lock().unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].name(), "UserStartedSpeakingFrame");
    }

    #[tokio::test]
    async fn test_handle_transcription_interim_ignored() {
        let config = default_config();
        let proc = started_processor().await;
        let captured = capture_pushes(&proc);
        let state = Arc::new(Mutex::new(SixtyDbSttState::new()));

        let text = r#"{"type":"transcription","text":"hello how","confidence":0.72,"language":"en","is_final":false,"speech_final":false,"is_partial":true}"#;
        handle_text_message(text, &proc, &config, &state).await;

        assert!(captured.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_handle_transcription_first_emit_non_finalized() {
        let config = default_config();
        let proc = started_processor().await;
        let captured = capture_pushes(&proc);
        let state = Arc::new(Mutex::new(SixtyDbSttState::new()));

        let text = r#"{"type":"transcription","text":"Hello how are you","confidence":0.85,"language":"en","is_final":true,"speech_final":false,"is_partial":false,"sentence_id":1}"#;
        handle_text_message(text, &proc, &config, &state).await;

        let frames = captured.lock().unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].name(), "TranscriptionFrame");
        if let FrameInner::Data(DataFrame::Transcription(ref data)) = frames[0].inner {
            assert_eq!(data.text, "Hello how are you");
            assert!(!data.finalized);
            assert_eq!(data.language, Some("en".to_string()));
        } else {
            panic!("expected TranscriptionFrame");
        }
    }

    #[tokio::test]
    async fn test_handle_transcription_canonical_finalized() {
        let config = default_config();
        let proc = started_processor().await;
        let captured = capture_pushes(&proc);
        let state = Arc::new(Mutex::new(SixtyDbSttState::new()));

        let text = r#"{"type":"transcription","text":"Hello, how are you?","confidence":0.87,"language":"en","is_final":true,"speech_final":true,"is_partial":false,"sentence_id":1,"duration":1.82}"#;
        handle_text_message(text, &proc, &config, &state).await;

        let frames = captured.lock().unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].name(), "TranscriptionFrame");
        assert_eq!(frames[1].name(), "UserStoppedSpeakingFrame");

        if let FrameInner::Data(DataFrame::Transcription(ref data)) = frames[0].inner {
            assert_eq!(data.text, "Hello, how are you?");
            assert!(data.finalized);
        } else {
            panic!("expected TranscriptionFrame");
        }
    }

    #[tokio::test]
    async fn test_handle_transcription_empty_final_ignored() {
        let config = default_config();
        let proc = started_processor().await;
        let captured = capture_pushes(&proc);
        let state = Arc::new(Mutex::new(SixtyDbSttState::new()));

        let text = r#"{"type":"transcription","text":"","confidence":0.0,"is_final":true,"speech_final":true,"processing_mode":"speech_end_no_result","timestamp":1700000000.789}"#;
        handle_text_message(text, &proc, &config, &state).await;

        assert!(captured.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_handle_language_changed() {
        let config = default_config();
        let proc = started_processor().await;
        let state = Arc::new(Mutex::new(SixtyDbSttState::new()));
        let text = r#"{"type":"language_changed","language":"Multi-language: HI","language_code":["hi"]}"#;
        handle_text_message(text, &proc, &config, &state).await;
        assert!(state.lock().await.ws_tx.is_none());
    }

    #[tokio::test]
    async fn test_handle_mode_changed() {
        let config = default_config();
        let proc = started_processor().await;
        let state = Arc::new(Mutex::new(SixtyDbSttState::new()));
        let text = r#"{"type":"mode_changed","continuous_mode":true,"mode_name":"continuous","silence_threshold":0.5}"#;
        handle_text_message(text, &proc, &config, &state).await;
        assert!(state.lock().await.ws_tx.is_none());
    }

    // ========================================================================
    // Message parsing — errors & lifecycle
    // ========================================================================

    #[tokio::test]
    async fn test_handle_error_pushes_upstream() {
        let config = default_config();
        let proc = started_processor().await;
        let errors = capture_errors(&proc);
        let state = Arc::new(Mutex::new(SixtyDbSttState::new()));

        let text = r#"{"type":"error","error":"Audio processing error","timestamp":1700000000.0}"#;
        handle_text_message(text, &proc, &config, &state).await;

        let errs = errors.lock().unwrap();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].error.contains("Audio processing error"));
    }

    #[tokio::test]
    async fn test_handle_concurrency_error() {
        let config = default_config();
        let proc = started_processor().await;
        let errors = capture_errors(&proc);
        let state = Arc::new(Mutex::new(SixtyDbSttState::new()));

        let text = r#"{"type":"error","error":"Too many concurrent STT sessions","error_code":"STT_CONCURRENCY_LIMIT","details":{"limit":8}}"#;
        handle_text_message(text, &proc, &config, &state).await;

        let errs = errors.lock().unwrap();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].error.contains("STT_CONCURRENCY_LIMIT"));
    }

    #[tokio::test]
    async fn test_handle_session_stopped_sets_state() {
        let config = default_config();
        let proc = started_processor().await;
        let state = Arc::new(Mutex::new(SixtyDbSttState::new()));

        let text = r#"{"type":"session_stopped","billing_summary":{"total_duration_seconds":12.40,"total_cost":0.000620}}"#;
        handle_text_message(text, &proc, &config, &state).await;

        assert!(matches!(state.lock().await.ws_state, WsState::Stopped));
    }

    #[tokio::test]
    async fn test_handle_unknown_type_ignored() {
        let config = default_config();
        let proc = started_processor().await;
        let captured = capture_pushes(&proc);
        let state = Arc::new(Mutex::new(SixtyDbSttState::new()));

        let text = r#"{"type":"flibble","data":123}"#;
        handle_text_message(text, &proc, &config, &state).await;

        assert!(captured.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_handle_malformed_json_ignored() {
        let config = default_config();
        let proc = started_processor().await;
        let captured = capture_pushes(&proc);
        let state = Arc::new(Mutex::new(SixtyDbSttState::new()));

        handle_text_message("not json at all", &proc, &config, &state).await;
        assert!(captured.lock().unwrap().is_empty());
    }

    // ========================================================================
    // Audio pipeline — prepare_and_send
    // ========================================================================

    #[tokio::test]
    async fn test_prepare_and_send_no_resample_passes_through() {
        let config = SixtyDbSttConfig {
            api_key: "test".to_string(),
            sample_rate: 16_000,
            encoding: SixtyDbEncoding::Linear,
            noise_reduction: false,
            ..Default::default()
        };
        let (handler, mut rx) = handler_with_state(config, WsState::SessionStarted).await;

        let pcm = vec![1000i16; 960];
        handler.prepare_and_send(&pcm, 16_000, false).await;

        let msg = rx.recv().await.expect("audio message should be sent");
        if let Message::Text(t) = msg {
            let val: serde_json::Value = serde_json::from_str(&t).unwrap();
            assert_eq!(val["type"], "audio");
            assert_eq!(val["encoding"], "linear");
            assert_eq!(val["sample_rate"], 16_000);
            let audio_bytes = BASE64.decode(val["audio"].as_str().unwrap()).unwrap();
            let samples = bytes_to_i16(&audio_bytes);
            assert_eq!(samples.len(), 960);
        } else {
            panic!("expected text message");
        }
    }

    #[tokio::test]
    async fn test_prepare_and_send_resamples_16k_to_48k() {
        let config = SixtyDbSttConfig {
            api_key: "test".to_string(),
            sample_rate: 48_000,
            encoding: SixtyDbEncoding::Linear,
            noise_reduction: false,
            resampler_quality: ResamplerQuality::Quick,
            ..Default::default()
        };
        let (handler, mut rx) = handler_with_state(config, WsState::SessionStarted).await;

        let pcm = vec![1000i16; 960];
        handler.prepare_and_send(&pcm, 16_000, false).await;

        let msg = rx.recv().await.expect("audio message should be sent");
        if let Message::Text(t) = msg {
            let val: serde_json::Value = serde_json::from_str(&t).unwrap();
            assert_eq!(val["type"], "audio");
            assert_eq!(val["sample_rate"], 48_000);
            let audio_bytes = BASE64.decode(val["audio"].as_str().unwrap()).unwrap();
            let samples = bytes_to_i16(&audio_bytes);
            assert!(
                samples.len() >= 2800 && samples.len() <= 3000,
                "expected ~2880 samples after upsampling, got {}",
                samples.len()
            );
        } else {
            panic!("expected text message");
        }

        assert!(handler.resampler.lock().await.is_some());
    }

    #[tokio::test]
    async fn test_prepare_and_send_with_denoise() {
        let config = SixtyDbSttConfig {
            api_key: "test".to_string(),
            sample_rate: 16_000,
            encoding: SixtyDbEncoding::Linear,
            noise_reduction: true,
            ..Default::default()
        };
        let (handler, mut rx) = handler_with_state(config, WsState::SessionStarted).await;

        let pcm = vec![1000i16; 960];
        handler.prepare_and_send(&pcm, 16_000, true).await;

        assert!(handler.noise_filter.lock().await.is_some());

        while let Ok(Some(_)) = tokio::time::timeout(tokio::time::Duration::from_millis(50), rx.recv()).await {}
    }

    #[tokio::test]
    async fn test_prepare_and_send_with_denoise_and_resample() {
        let config = SixtyDbSttConfig {
            api_key: "test".to_string(),
            sample_rate: 48_000,
            encoding: SixtyDbEncoding::Linear,
            noise_reduction: true,
            resampler_quality: ResamplerQuality::Quick,
            ..Default::default()
        };
        let (handler, mut rx) = handler_with_state(config, WsState::SessionStarted).await;

        let pcm = vec![1000i16; 960];
        handler.prepare_and_send(&pcm, 16_000, true).await;

        assert!(handler.noise_filter.lock().await.is_some());
        assert!(handler.resampler.lock().await.is_some());

        while let Ok(Some(_)) = tokio::time::timeout(tokio::time::Duration::from_millis(50), rx.recv()).await {}
    }

    #[tokio::test]
    async fn test_prepare_and_send_buffers_when_not_session_started() {
        let config = SixtyDbSttConfig {
            api_key: "test".to_string(),
            sample_rate: 16_000,
            encoding: SixtyDbEncoding::Linear,
            noise_reduction: false,
            ..Default::default()
        };
        let (handler, mut rx) = handler_with_state(config, WsState::ConnectionEstablished).await;

        let pcm = vec![1000i16; 100];
        handler.prepare_and_send(&pcm, 16_000, false).await;

        assert!(rx.try_recv().is_err());
        let state = handler.state.lock().await;
        assert_eq!(state.audio_buffer.len(), 1);
        assert_eq!(bytes_to_i16(&state.audio_buffer[0]), pcm);
    }

    // ========================================================================
    // Audio pipeline — send_audio / deliver_audio encoding modes
    // ========================================================================

    #[tokio::test]
    async fn test_deliver_audio_linear_sends_json() {
        let config = SixtyDbSttConfig {
            api_key: "test".to_string(),
            sample_rate: 16_000,
            encoding: SixtyDbEncoding::Linear,
            ..Default::default()
        };
        let (handler, mut rx) = handler_with_state(config, WsState::SessionStarted).await;

        let bytes = i16_to_bytes(&vec![1000i16, 2000i16, 3000i16]);
        handler.deliver_audio(&bytes).await;

        let msg = rx.recv().await.unwrap();
        if let Message::Text(t) = msg {
            let val: serde_json::Value = serde_json::from_str(&t).unwrap();
            assert_eq!(val["type"], "audio");
            assert_eq!(val["encoding"], "linear");
            let decoded = BASE64.decode(val["audio"].as_str().unwrap()).unwrap();
            assert_eq!(decoded, bytes);
        } else {
            panic!("expected text message");
        }
    }

    #[tokio::test]
    async fn test_deliver_audio_mulaw_sends_binary() {
        let config = SixtyDbSttConfig {
            api_key: "test".to_string(),
            sample_rate: 8000,
            encoding: SixtyDbEncoding::Mulaw,
            ..Default::default()
        };
        let (handler, mut rx) = handler_with_state(config, WsState::SessionStarted).await;

        let bytes = vec![0xFFu8, 0xAA, 0x55];
        handler.deliver_audio(&bytes).await;

        let msg = rx.recv().await.unwrap();
        if let Message::Binary(b) = msg {
            assert_eq!(b.to_vec(), bytes);
        } else {
            panic!("expected binary message");
        }
    }

    // ========================================================================
    // Frame routing — on_process_frame
    // ========================================================================

    #[tokio::test]
    async fn test_on_process_input_audio_passes_frame_downstream() {
        let config = SixtyDbSttConfig {
            api_key: "test".to_string(),
            sample_rate: 16_000,
            encoding: SixtyDbEncoding::Linear,
            noise_reduction: false,
            ..Default::default()
        };
        let handler = SixtyDbSttHandler::new(config);
        let proc = started_processor().await;
        let captured = capture_pushes(&proc);

        let (tx, mut rx) = mpsc::channel::<Message>(64);
        {
            let mut state = handler.state.lock().await;
            state.ws_tx = Some(tx);
            state.ws_state = WsState::SessionStarted;
        }

        let audio_data = AudioRawData::new(i16_to_bytes(&vec![1000i16; 100]), 16_000, 1);
        let frame = Frame::input_audio_raw(audio_data);
        handler.on_process_frame(&proc, frame, FrameDirection::Downstream).await.unwrap();

        let frames = captured.lock().unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].name(), "InputAudioRawFrame");

        let msg = rx.recv().await.expect("audio should be sent");
        assert!(matches!(msg, Message::Text(_)));
    }

    #[tokio::test]
    async fn test_on_process_vad_stop_flushes_noise_filter() {
        let config = SixtyDbSttConfig {
            api_key: "test".to_string(),
            sample_rate: 16_000,
            encoding: SixtyDbEncoding::Linear,
            noise_reduction: true,
            ..Default::default()
        };
        let handler = SixtyDbSttHandler::new(config);
        let proc = started_processor().await;

        let (tx, mut rx) = mpsc::channel::<Message>(64);
        {
            let mut state = handler.state.lock().await;
            state.ws_tx = Some(tx);
            state.ws_state = WsState::SessionStarted;
        }

        let audio_data = AudioRawData::new(i16_to_bytes(&vec![1000i16; 960]), 16_000, 1);
        let frame = Frame::input_audio_raw(audio_data);
        handler.on_process_frame(&proc, frame.clone(), FrameDirection::Downstream).await.unwrap();

        let vad_frame = Frame::vad_user_stopped_speaking(0.0, 0.0);
        handler.on_process_frame(&proc, vad_frame, FrameDirection::Downstream).await.unwrap();

        assert!(handler.noise_filter.lock().await.is_some());
    }

    #[tokio::test]
    async fn test_on_process_end_frame_sends_stop() {
        let config = SixtyDbSttConfig {
            api_key: "test".to_string(),
            sample_rate: 16_000,
            encoding: SixtyDbEncoding::Linear,
            noise_reduction: false,
            ..Default::default()
        };
        let handler = SixtyDbSttHandler::new(config);
        let proc = started_processor().await;

        let (tx, mut rx) = mpsc::channel::<Message>(64);
        {
            let mut state = handler.state.lock().await;
            state.ws_tx = Some(tx);
            state.ws_state = WsState::SessionStarted;
        }

        let frame = Frame::end();
        handler.on_process_frame(&proc, frame, FrameDirection::Downstream).await.unwrap();

        let msg = tokio::time::timeout(tokio::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("timeout waiting for stop")
            .expect("stop message should be sent");
        if let Message::Text(t) = msg {
            assert!(t.contains("stop"));
        } else {
            panic!("expected text message");
        }

        assert!(matches!(
            handler.state.lock().await.ws_state,
            WsState::Stopping | WsState::Disconnected
        ));
    }

    #[tokio::test]
    async fn test_on_process_cancel_frame_sends_stop() {
        let config = SixtyDbSttConfig {
            api_key: "test".to_string(),
            sample_rate: 16_000,
            encoding: SixtyDbEncoding::Linear,
            noise_reduction: false,
            ..Default::default()
        };
        let handler = SixtyDbSttHandler::new(config);
        let proc = started_processor().await;

        let (tx, mut rx) = mpsc::channel::<Message>(64);
        {
            let mut state = handler.state.lock().await;
            state.ws_tx = Some(tx);
            state.ws_state = WsState::SessionStarted;
        }

        let frame = Frame::cancel();
        handler.on_process_frame(&proc, frame, FrameDirection::Downstream).await.unwrap();

        let msg = tokio::time::timeout(tokio::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("timeout waiting for stop")
            .expect("stop message should be sent");
        if let Message::Text(t) = msg {
            assert!(t.contains("stop"));
        } else {
            panic!("expected text message");
        }
    }

    // ========================================================================
    // State machine & lifecycle
    // ========================================================================

    #[tokio::test]
    async fn test_disconnect_cleans_up_state() {
        let config = default_config();
        let handler = SixtyDbSttHandler::new(config);

        {
            let mut nf = handler.noise_filter.lock().await;
            *nf = Some(RNNoiseFilter::new(16_000));
        }
        {
            let mut r = handler.resampler.lock().await;
            *r = Some(StreamResampler::new(16_000, 48_000, ResamplerQuality::Quick));
        }
        {
            let mut state = handler.state.lock().await;
            state.input_sample_rate = Some(16_000);
            state.audio_buffer.push(vec![1, 2, 3]);
        }

        handler.disconnect().await;

        let state = handler.state.lock().await;
        assert!(matches!(state.ws_state, WsState::Stopping));
        assert!(state.ws_tx.is_none());
        assert!(state.audio_buffer.is_empty());
        assert!(state.input_sample_rate.is_none());
        assert!(handler.noise_filter.lock().await.is_none());
        assert!(handler.resampler.lock().await.is_none());
    }

    #[tokio::test]
    async fn test_send_stop_produces_valid_json() {
        let config = default_config();
        let (handler, mut rx) = handler_with_state(config, WsState::SessionStarted).await;

        handler.send_stop().await;

        let msg = rx.recv().await.expect("stop message");
        if let Message::Text(t) = msg {
            let val: serde_json::Value = serde_json::from_str(&t).unwrap();
            assert_eq!(val["type"], "stop");
        } else {
            panic!("expected text message");
        }
    }

    #[tokio::test]
    async fn test_audio_buffering_and_flush() {
        let config = SixtyDbSttConfig {
            api_key: "test".to_string(),
            sample_rate: 16_000,
            encoding: SixtyDbEncoding::Linear,
            ..Default::default()
        };
        let (handler, mut rx) = handler_with_state(config, WsState::ConnectionEstablished).await;

        let chunk1 = vec![1u8, 2, 3, 4];
        let chunk2 = vec![5u8, 6, 7, 8];
        handler.send_audio(&chunk1).await;
        handler.send_audio(&chunk2).await;

        {
            let state = handler.state.lock().await;
            assert_eq!(state.audio_buffer.len(), 2);
            assert_eq!(state.audio_buffer[0], chunk1);
            assert_eq!(state.audio_buffer[1], chunk2);
        }

        {
            let mut state = handler.state.lock().await;
            state.ws_state = WsState::SessionStarted;
        }

        let chunk3 = vec![9u8, 10, 11, 12];
        handler.send_audio(&chunk3).await;

        let msg1 = rx.recv().await.expect("chunk1");
        let msg2 = rx.recv().await.expect("chunk2");
        let msg3 = rx.recv().await.expect("chunk3");

        for msg in [msg1, msg2, msg3] {
            assert!(matches!(msg, Message::Text(_)));
        }

        assert!(handler.state.lock().await.audio_buffer.is_empty());
    }

    // ========================================================================
    // Edge cases
    // ========================================================================

    #[tokio::test]
    async fn test_prepare_and_send_empty_pcm_does_nothing() {
        let config = SixtyDbSttConfig {
            api_key: "test".to_string(),
            sample_rate: 16_000,
            encoding: SixtyDbEncoding::Linear,
            noise_reduction: false,
            ..Default::default()
        };
        let (handler, mut rx) = handler_with_state(config, WsState::SessionStarted).await;

        handler.prepare_and_send(&[], 16_000, false).await;

        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn test_transcription_with_words_field() {
        let config = default_config();
        let proc = started_processor().await;
        let captured = capture_pushes(&proc);
        let state = Arc::new(Mutex::new(SixtyDbSttState::new()));

        let text = r#"{"type":"transcription","text":"Hello","confidence":0.94,"language":"en","is_final":true,"speech_final":true,"words":[{"word":"Hello","start":0.0,"end":0.32,"confidence":0.94}],"sentence_id":5}"#;
        handle_text_message(text, &proc, &config, &state).await;

        let frames = captured.lock().unwrap();
        assert_eq!(frames.len(), 2);
        if let FrameInner::Data(DataFrame::Transcription(ref data)) = frames[0].inner {
            assert_eq!(data.text, "Hello");
            assert!(data.finalized);
        } else {
            panic!("expected TranscriptionFrame");
        }
    }

    #[tokio::test]
    async fn test_transcription_boosted_word() {
        let config = default_config();
        let proc = started_processor().await;
        let captured = capture_pushes(&proc);
        let state = Arc::new(Mutex::new(SixtyDbSttState::new()));

        let text = r#"{"type":"transcription","text":"Acme","confidence":0.85,"language":"en","is_final":true,"speech_final":true,"words":[{"word":"Acme","start":1.5,"end":2.0,"confidence":0.85,"boosted":true,"original":"akmie"}]}"#;
        handle_text_message(text, &proc, &config, &state).await;

        let frames = captured.lock().unwrap();
        assert_eq!(frames[0].name(), "TranscriptionFrame");
        if let FrameInner::Data(DataFrame::Transcription(ref data)) = frames[0].inner {
            assert_eq!(data.text, "Acme");
        } else {
            panic!("expected TranscriptionFrame");
        }
    }
}
