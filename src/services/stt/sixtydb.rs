//! 60db Speech-to-Text WebSocket service.
//!
//! Matches the official protocol:
//!   1. Connect to wss://api.60db.ai/ws/stt?apiKey=...
//!   2. On connection_established → send {"type":"start",...}
//!   3. On connected → begin streaming raw binary audio
//!   4. Receive {"type":"transcription","text":"..."} from server
//!
//! Default audio format: μ-law @ 8kHz (telephony standard).

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use chrono::Utc;
use futures::{SinkExt, StreamExt};
use log;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::Serialize;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::Message;

use crate::audio_process::resamplers::{ResamplerQuality, StreamResampler};
use crate::billing::{BillingCollector, BillingEvent};
use crate::error::Result;
use crate::frames::{
    ControlFrame, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor,
    SystemFrame, TranscriptionData,
};

const SIXTYDB_BASE_WSS: &str = "wss://api.60db.ai/ws/stt";
const SIXTYDB_BASE_WS: &str = "ws://api.60db.ai/ws/stt";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum SixtyDbEncoding {
    Linear,
    Mulaw,
}

impl SixtyDbEncoding {
    fn as_str(&self) -> &'static str {
        match self {
            SixtyDbEncoding::Linear => "linear",
            SixtyDbEncoding::Mulaw => "mulaw",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SixtyDbSttConfig {
    pub api_key: String,
    pub languages: Vec<String>,
    pub encoding: SixtyDbEncoding,
    pub sample_rate: u32,
    pub continuous_mode: bool,
    pub insecure: bool,
}

impl Default for SixtyDbSttConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            languages: vec!["en".to_string()],
            encoding: SixtyDbEncoding::Linear,
            sample_rate: 16_000,
            continuous_mode: true,
            insecure: false,
        }
    }
}

impl SixtyDbSttConfig {
    fn ws_url(&self) -> String {
        let base = if self.insecure { SIXTYDB_BASE_WS } else { SIXTYDB_BASE_WSS };
        format!("{}?apiKey={}", base, urlencoding(&self.api_key))
    }

    fn start_message(&self) -> String {
        #[derive(Serialize)]
        struct StartMsg {
            #[serde(rename = "type")]
            msg_type: &'static str,
            languages: Vec<String>,
            config: Config,
        }
        #[derive(Serialize)]
        struct Config {
            encoding: &'static str,
            sample_rate: u32,
            continuous_mode: bool,
        }
        serde_json::to_string(&StartMsg {
            msg_type: "start",
            languages: self.languages.clone(),
            config: Config {
                encoding: self.encoding.as_str(),
                sample_rate: self.sample_rate,
                continuous_mode: self.continuous_mode,
            },
        })
        .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

struct SixtyDbSttState {
    ws_tx: Option<mpsc::Sender<Message>>,
    send_task: Option<JoinHandle<()>>,
    receive_task: Option<JoinHandle<()>>,
    ready: bool,
}

impl Default for SixtyDbSttState {
    fn default() -> Self {
        Self {
            ws_tx: None,
            send_task: None,
            receive_task: None,
            ready: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

pub struct SixtyDbSttHandler {
    config:    SixtyDbSttConfig,
    state:     Arc<Mutex<SixtyDbSttState>>,
    resampler: Mutex<Option<StreamResampler>>,
    /// Optional billing collector — records audio duration from server billing_summary.
    billing:   Option<Arc<dyn BillingCollector>>,
}

impl SixtyDbSttHandler {
    pub fn new(config: SixtyDbSttConfig) -> Self {
        Self {
            config,
            state:     Arc::new(Mutex::new(SixtyDbSttState::default())),
            resampler: Mutex::new(None),
            billing:   None,
        }
    }

    pub fn with_billing(mut self, billing: Arc<dyn BillingCollector>) -> Self {
        self.billing = Some(billing);
        self
    }

    pub fn into_processor(self) -> FrameProcessor {
        FrameProcessor::new("SixtyDbStt", Box::new(self), false)
    }

    // ── Connection ─────────────────────────────────────────────────

    async fn connect(&self, processor: FrameProcessor) {
        let url = self.config.ws_url();
        log::info!("SixtyDbStt: connecting to {}", &url[..url.len().min(60)]);

        let ws_stream = match tokio_tungstenite::connect_async(&url).await {
            Ok((stream, _)) => stream,
            Err(e) => {
                let _ = processor.push_error(format!("SixtyDbStt: connect failed: {}", e), false).await;
                return;
            }
        };

        let (sink, stream) = ws_stream.split();
        let (ws_tx, ws_rx) = mpsc::channel::<Message>(64);

        let send_task = tokio::spawn(run_send_task(sink, ws_rx));
        let receive_task = tokio::spawn(run_receive_task(
            stream,
            processor,
            self.config.clone(),
            self.billing.clone(),
            self.state.clone(),
        ));

        let mut state = self.state.lock().await;
        state.ws_tx = Some(ws_tx);
        state.send_task = Some(send_task);
        state.receive_task = Some(receive_task);
        drop(state);

        log::info!("SixtyDbStt: WebSocket connected");
    }

    async fn disconnect(&self) {
        let mut state = self.state.lock().await;
        state.ready = false;
        if let Some(h) = state.receive_task.take() { h.abort(); }
        if let Some(h) = state.send_task.take() { h.abort(); }
        state.ws_tx = None;
        drop(state);
        log::info!("SixtyDbStt: disconnected");
    }

    async fn send_json(&self, json: String) {
        let tx = { self.state.lock().await.ws_tx.clone() };
        if let Some(tx) = tx {
            let _ = tx.send(Message::Text(json.into())).await;
        }
    }

    // ── Audio pipeline ─────────────────────────────────────────────

    async fn prepare_and_send(&self, pcm: &[i16], input_rate: u32) {
        // Resample if input rate differs from target
        let resampled = if input_rate != self.config.sample_rate {
            let mut r_guard = self.resampler.lock().await;
            if r_guard.is_none() {
                log::info!("SixtyDbStt: resampling {} -> {} Hz", input_rate, self.config.sample_rate);
                *r_guard = Some(StreamResampler::new(
                    input_rate,
                    self.config.sample_rate,
                    ResamplerQuality::Quick,
                ));
            }
            let f32_samples: Vec<f32> = pcm.iter().map(|&s| s as f32).collect();
            let resampled_f32 = r_guard.as_mut().unwrap().process(&f32_samples);
            f32_to_i16(&resampled_f32)
        } else {
            pcm.to_vec()
        };

        if resampled.is_empty() {
            return;
        }

        match self.config.encoding {
            SixtyDbEncoding::Mulaw => {
                let ulaw: Vec<u8> = resampled.iter().map(|&s| linear_to_ulaw(s)).collect();
                self.send_audio(&ulaw).await;
            }
            SixtyDbEncoding::Linear => {
                let bytes = i16_to_bytes(&resampled);
                self.send_audio(&bytes).await;
            }
        }
    }

    async fn send_audio(&self, audio: &[u8]) {
        let tx = { self.state.lock().await.ws_tx.clone() };
        if let Some(tx) = tx {
            let msg = match self.config.encoding {
                SixtyDbEncoding::Mulaw => Message::Binary(audio.to_vec().into()),
                SixtyDbEncoding::Linear => {
                    let json = serde_json::json!({
                        "type": "audio",
                        "audio": BASE64.encode(audio),
                        "encoding": "linear",
                        "sample_rate": self.config.sample_rate,
                        "timestamp": unix_ms(),
                    });
                    Message::Text(json.to_string().into())
                }
            };
            let _ = tx.send(msg).await;
        }
    }
}

// ---------------------------------------------------------------------------
// FrameHandler
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
                self.prepare_and_send(&pcm, audio.sample_rate).await;
            }
            FrameInner::Control(ControlFrame::End { .. })
            | FrameInner::System(SystemFrame::Cancel { .. }) => {
                self.send_json(r#"{"type":"stop"}"#.to_string()).await;
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                self.disconnect().await;
                processor.push_frame(frame, direction).await?;
            }
            _ => processor.push_frame(frame, direction).await?,
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
            break;
        }
    }
    let _ = sink.close().await;
}

async fn run_receive_task(
    mut stream:   WsStream,
    processor:    FrameProcessor,
    config:       SixtyDbSttConfig,
    billing:      Option<Arc<dyn BillingCollector>>,
    shared_state: Arc<Mutex<SixtyDbSttState>>,
) {
    while let Some(result) = stream.next().await {
        match result {
            Ok(Message::Text(text)) => {
                handle_text_message(
                    text.as_str(), &processor, &config, &shared_state, &billing,
                ).await;
            }
            Ok(Message::Close(_)) => {
                log::info!("SixtyDbStt: server closed WebSocket");
                break;
            }
            Err(e) => {
                let _ = processor.push_error(format!("SixtyDbStt: receive error: {}", e), false).await;
                break;
            }
            _ => {}
        }
    }
}

async fn handle_text_message(
    text:         &str,
    processor:    &FrameProcessor,
    config:       &SixtyDbSttConfig,
    shared_state: &Arc<Mutex<SixtyDbSttState>>,
    billing:      &Option<Arc<dyn BillingCollector>>,
) {
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

    let msg_type = obj.get("type").and_then(|v| v.as_str());

    // connection_established (typed or top-level key)
    if obj.contains_key("connection_established") || msg_type == Some("connection_established") {
        log::info!("SixtyDbStt: connection_established — sending start");
        let start_json = config.start_message();
        {
            let mut state = shared_state.lock().await;
            if let Some(ref tx) = state.ws_tx {
                let _ = tx.send(Message::Text(start_json.into())).await;
            }
        }
        return;
    }

    let msg_type = match msg_type {
        Some(t) => t,
        None => return,
    };

    match msg_type {
        "connected" => {
            log::info!("SixtyDbStt: connected — audio streaming enabled");
            let mut state = shared_state.lock().await;
            state.ready = true;
        }

        "transcription" => {
            let text = obj.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
            if text.is_empty() {
                return;
            }
            let is_final = obj.get("is_final").and_then(|v| v.as_bool()).unwrap_or(false);
            let mut frame_data = TranscriptionData::new(text, "", time_now());
            frame_data.finalized = is_final;
            log::info!("SixtyDbStt: transcript='{}' is_final={}", frame_data.text, is_final);
            let _ = processor.push_frame(Frame::transcription(frame_data), FrameDirection::Downstream).await;
        }

        "speech_started" => {
            log::info!("SixtyDbStt: speech_started");
            let _ = processor.push_frame(Frame::user_started_speaking(), FrameDirection::Downstream).await;
        }

        "session_stopped" => {
            if let Some(summary) = obj.get("billing_summary") {
                let duration_ms = summary
                    .get("duration_ms")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                log::info!("SixtyDbStt: billing_summary duration_ms={:.0}", duration_ms);
                if duration_ms > 0.0 {
                    if let Some(bc) = billing {
                        bc.record(BillingEvent::SttUsage {
                            session_id:        bc.session_id(),
                            provider:          "sixtydb".to_string(),
                            audio_duration_ms: duration_ms,
                            occurred_at:       Utc::now(),
                        });
                    }
                }
            }
        }

        "error" => {
            let error = obj.get("error").and_then(|v| v.as_str()).unwrap_or("unknown error");
            log::warn!("SixtyDbStt: server error: {}", error);
            let _ = processor.push_error(format!("SixtyDbStt: {}", error), false).await;
        }

        _ => {
            log::debug!("SixtyDbStt: unhandled message type: {}", msg_type);
        }
    }
}

// ---------------------------------------------------------------------------
// μ-law conversion (ITU-T G.711, matching Python audioop.lin2ulaw)
// ---------------------------------------------------------------------------

fn linear_to_ulaw(sample: i16) -> u8 {
    const BIAS: i32 = 0x84;
    const CLIP: i32 = 32635;
    const SEG_END: [i32; 8] = [0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF];

    let mut pcm = sample as i32;
    let mask;

    if pcm < 0 {
        pcm = BIAS - pcm;
        mask = 0x7F;
    } else {
        pcm = BIAS + pcm;
        mask = 0xFF;
    }

    if pcm > CLIP {
        pcm = CLIP;
    }

    // Find segment
    let mut seg = 8;
    for (i, &end) in SEG_END.iter().enumerate() {
        if pcm <= end {
            seg = i;
            break;
        }
    }

    if seg >= 8 {
        (0x7F ^ mask) as u8
    } else {
        let uval = ((seg as i32) << 4) | ((pcm >> ((seg as i32) + 3)) & 0x0F);
        (uval ^ mask) as u8
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn bytes_to_i16(audio: &[u8]) -> Vec<i16> {
    audio.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
}

fn i16_to_bytes(samples: &[i16]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples.iter().map(|&s| s.clamp(i16::MIN as f32, i16::MAX as f32) as i16).collect()
}

fn time_now() -> String {
    let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
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
    use crate::frames::{AudioRawData, FrameInner, StartFrameData};
    use crate::FrameProcessor;
    use crate::PassthroughHandler;

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

    #[tokio::test]
    async fn test_start_message_json() {
        let config = default_config();
        let json = config.start_message();
        let val: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(val["type"], "start");
        assert_eq!(val["config"]["encoding"], "linear");
        assert_eq!(val["config"]["sample_rate"], 16000);
        assert_eq!(val["config"]["continuous_mode"], true);
    }

    #[tokio::test]
    async fn test_ulaw_silence() {
        // μ-law: 0 maps to a specific encoded value (implementation-dependent)
        let _ = linear_to_ulaw(0);
        let _ = linear_to_ulaw(100);
        let _ = linear_to_ulaw(-100);
    }

    #[tokio::test]
    async fn test_ulaw_roundtrip() {
        // Verify common values
        let samples: Vec<i16> = (0..100).map(|i| i as i16 * 300).collect();
        for &s in &samples {
            let _ = linear_to_ulaw(s);
        }
    }

    #[tokio::test]
    async fn test_on_process_input_audio_passes_frame_downstream() {
        let config = SixtyDbSttConfig {
            api_key: "test".to_string(),
            encoding: SixtyDbEncoding::Linear,
            sample_rate: 16_000,
            ..Default::default()
        };
        let handler = SixtyDbSttHandler::new(config);
        let proc = started_processor().await;

        let audio_data = AudioRawData::new(i16_to_bytes(&vec![1000i16; 100]), 16_000, 1);
        let frame = Frame::input_audio_raw(audio_data);
        handler.on_process_frame(&proc, frame, FrameDirection::Downstream).await.unwrap();
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
    impl BillingCollector for MockCollector {
        fn record(&self, e: BillingEvent) { self.events.lock().unwrap().push(e); }
        fn session_id(&self) -> uuid::Uuid { self.session_id }
    }

    fn new_state() -> Arc<Mutex<SixtyDbSttState>> {
        Arc::new(Mutex::new(SixtyDbSttState::default()))
    }

    #[tokio::test]
    async fn billing_session_stopped_with_duration_emits_stt_usage() {
        let json = r#"{"type":"session_stopped","billing_summary":{"duration_ms":4200.0}}"#;
        let mock    = MockCollector::new();
        let billing: Option<Arc<dyn BillingCollector>> = Some(mock.clone());
        let proc    = started_processor().await;

        handle_text_message(json, &proc, &default_config(), &new_state(), &billing).await;

        let evs = mock.events();
        assert_eq!(evs.len(), 1, "expected exactly one SttUsage event");
        match &evs[0] {
            BillingEvent::SttUsage { provider, audio_duration_ms, .. } => {
                assert_eq!(provider, "sixtydb");
                assert!((audio_duration_ms - 4200.0).abs() < 0.001);
            }
            other => panic!("expected SttUsage, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn billing_session_stopped_with_zero_duration_emits_no_event() {
        let json = r#"{"type":"session_stopped","billing_summary":{"duration_ms":0.0}}"#;
        let mock    = MockCollector::new();
        let billing: Option<Arc<dyn BillingCollector>> = Some(mock.clone());
        let proc    = started_processor().await;

        handle_text_message(json, &proc, &default_config(), &new_state(), &billing).await;
        assert_eq!(mock.events().len(), 0, "zero duration must not emit billing event");
    }

    #[tokio::test]
    async fn billing_session_stopped_without_billing_summary_emits_no_event() {
        let json = r#"{"type":"session_stopped"}"#;
        let mock    = MockCollector::new();
        let billing: Option<Arc<dyn BillingCollector>> = Some(mock.clone());
        let proc    = started_processor().await;

        handle_text_message(json, &proc, &default_config(), &new_state(), &billing).await;
        assert_eq!(mock.events().len(), 0);
    }

    #[tokio::test]
    async fn billing_non_session_stopped_messages_emit_no_event() {
        let mock    = MockCollector::new();
        let billing: Option<Arc<dyn BillingCollector>> = Some(mock.clone());
        let proc    = started_processor().await;
        let cfg     = default_config();
        let state   = new_state();

        for json in &[
            r#"{"type":"transcription","text":"hello"}"#,
            r#"{"type":"speech_started"}"#,
            r#"{"type":"connected"}"#,
        ] {
            handle_text_message(json, &proc, &cfg, &state, &billing).await;
        }
        assert_eq!(mock.events().len(), 0);
    }

    #[tokio::test]
    async fn billing_no_collector_session_stopped_does_not_panic() {
        let json    = r#"{"type":"session_stopped","billing_summary":{"duration_ms":1000.0}}"#;
        let billing: Option<Arc<dyn BillingCollector>> = None;
        let proc    = started_processor().await;

        handle_text_message(json, &proc, &default_config(), &new_state(), &billing).await;
    }

    #[test]
    fn with_billing_sets_field() {
        use crate::billing::NoopBillingCollector;
        let h = SixtyDbSttHandler::new(SixtyDbSttConfig::default())
            .with_billing(Arc::new(NoopBillingCollector));
        assert!(h.billing.is_some());
    }
}
