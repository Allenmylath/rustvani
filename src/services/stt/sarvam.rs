//! Sarvam AI Speech-to-Text — a [`SttProvider`] over the streaming WebSocket API.
//!
//! All of the turn machinery this service used to own — the TurnGate, the
//! audio front-end, billing, the WebSocket tasks — now lives in
//! [`stt::core`](super::core) and is shared by every provider. What remains
//! here is Sarvam's wire protocol and its configuration.
//!
//! Pipeline position:
//!   transport.input() → SarvamSttHandler → LLMUserAggregator → llm → tts → out
//!
//! Wiring:
//!   let stt = SarvamSttHandler::new(SarvamSttConfig {
//!       api_key: std::env::var("SARVAM_API_KEY").unwrap(),
//!       ..Default::default()
//!   })
//!   .into_processor();
//!
//! Frames consumed / produced: see [`SttService`].
//!
//! Auth: api-subscription-key header (lowercase), per SDK source.
//! URL:  wss://api.sarvam.ai/speech-to-text/ws
//! Lang: language-code param (hyphen, not underscore)

use serde::Deserialize;

use crate::frames::FrameProcessor;

use super::core::util::percent_encode;
use super::core::{
    AudioSpec, Handshake, InterimPolicy, Outgoing, SttCoreConfig, SttEvent, SttProvider,
    SttService, WsMessage,
};

// `NoiseBackend` moved to `core::frontend` — it is not Sarvam-specific. Kept
// re-exported here so `services::stt::sarvam::NoiseBackend` still resolves.
pub use super::core::NoiseBackend;

// ---------------------------------------------------------------------------
// Constants — verified against SDK source and AsyncAPI spec
// ---------------------------------------------------------------------------

const SARVAM_BASE_WSS: &str = "wss://api.sarvam.ai";
const STT_PATH: &str = "/speech-to-text/ws";
const STT_TRANSLATE_PATH: &str = "/speech-to-text-translate/ws";

// saaras:v2.5 uses the translate endpoint
const TRANSLATE_MODELS: &[&str] = &["saaras:v2.5"];
// saaras:v3 supports the mode param and the fine-grained VAD tuning set
const MODE_MODELS: &[&str] = &["saaras:v3"];

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Fine-grained server-side VAD tuning. **`saaras:v3` only** — ignored by
/// other models, and only emitted for fields that are `Some`.
///
/// These tune Sarvam's *server* VAD. They do not affect the local Silero VAD
/// that owns turn boundaries in this pipeline; with
/// [`SttCoreConfig::audio_gating`] on (the default) the server sees only
/// local-VAD-attested audio anyway, so these mostly matter when gating is off.
#[derive(Debug, Clone, Default)]
pub struct SarvamVadTuning {
    /// Speech probability above which a frame counts as speech (0.0–1.0).
    pub positive_speech_threshold: Option<f32>,
    /// Probability below which a frame counts as silence (0.0–1.0).
    pub negative_speech_threshold: Option<f32>,
    /// Consecutive speech frames required to declare speech.
    pub min_speech_frames: Option<u32>,
    /// As above, for the first turn of the session.
    pub first_turn_min_speech_frames: Option<u32>,
    /// Silence frames required to end a segment.
    pub negative_frames_count: Option<u32>,
    /// Sliding window over which `negative_frames_count` is measured.
    pub negative_frames_window: Option<u32>,
    /// Volume (dB) floor for speech onset.
    pub start_speech_volume_threshold: Option<f32>,
    /// Speech frames required to register a barge-in.
    pub interrupt_min_speech_frames: Option<u32>,
    /// Frames of audio to prepend before detected onset.
    pub pre_speech_pad_frames: Option<u32>,
    /// Leading frames to skip at session start.
    pub num_initial_ignored_frames: Option<u32>,
}

impl SarvamVadTuning {
    fn push_params(&self, params: &mut Vec<String>) {
        fn f(params: &mut Vec<String>, key: &str, v: Option<f32>) {
            if let Some(v) = v {
                params.push(format!("{}={}", key, v));
            }
        }
        fn u(params: &mut Vec<String>, key: &str, v: Option<u32>) {
            if let Some(v) = v {
                params.push(format!("{}={}", key, v));
            }
        }

        f(params, "positive_speech_threshold", self.positive_speech_threshold);
        f(params, "negative_speech_threshold", self.negative_speech_threshold);
        u(params, "min_speech_frames", self.min_speech_frames);
        u(params, "first_turn_min_speech_frames", self.first_turn_min_speech_frames);
        u(params, "negative_frames_count", self.negative_frames_count);
        u(params, "negative_frames_window", self.negative_frames_window);
        f(params, "start_speech_volume_threshold", self.start_speech_volume_threshold);
        u(params, "interrupt_min_speech_frames", self.interrupt_min_speech_frames);
        u(params, "pre_speech_pad_frames", self.pre_speech_pad_frames);
        u(params, "num_initial_ignored_frames", self.num_initial_ignored_frames);
    }
}

/// Configuration for [`SarvamSttHandler`].
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

    /// Optional biasing prompt (domain terms, names) to steer recognition.
    pub prompt: Option<String>,

    /// Audio sample rate. Audio arriving at any other rate is resampled to
    /// this by the core's front-end.
    pub sample_rate: u32,

    /// Audio encoding. "wav" or "pcm_s16le" / "pcm_l16" / "pcm_raw".
    pub encoding: String,

    /// Enable high VAD sensitivity (shorter silence before flush).
    pub high_vad_sensitivity: bool,

    /// Receive VAD signals from server (speech_start / speech_end events).
    pub vad_signals: bool,

    /// Fine-grained server VAD tuning — saaras:v3 only.
    pub vad_tuning: SarvamVadTuning,

    /// Enable noise suppression before sending audio to Sarvam.
    /// Default: true.
    pub noise_reduction: bool,

    /// Which noise-suppression backend to use when `noise_reduction` is on.
    /// Default: [`NoiseBackend::Rnnoise`].
    pub noise_backend: NoiseBackend,

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
            prompt:                  None,
            sample_rate:             16_000,
            encoding:                "wav".to_string(),
            high_vad_sensitivity:    false,
            vad_signals:             false,
            vad_tuning:              SarvamVadTuning::default(),
            noise_reduction:         true,
            noise_backend:           NoiseBackend::Rnnoise,
            agc:                     true,
            audio_gating:            true,
            pre_roll_ms:             500,
            stop_release_timeout_ms: 1_200,
        }
    }
}

impl SarvamSttConfig {
    /// Split into the provider-independent core settings and the Sarvam
    /// protocol adapter.
    fn split(self) -> (SttCoreConfig, SarvamProvider) {
        let core = SttCoreConfig {
            sample_rate:             self.sample_rate,
            noise_reduction:         self.noise_reduction,
            noise_backend:           self.noise_backend,
            agc:                     self.agc,
            agc_config:              Default::default(),
            audio_gating:            self.audio_gating,
            pre_roll_ms:             self.pre_roll_ms,
            stop_release_timeout_ms: self.stop_release_timeout_ms,
            // Sarvam's streaming API emits finals only; nothing to drop.
            interim_policy:          InterimPolicy::Drop,
        };
        let audio = AudioSpec::new(self.sample_rate, self.encoding.clone());
        (core, SarvamProvider { config: self, audio })
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
    #[allow(dead_code)]
    request_id:    Option<String>,
    metrics:       Option<SarvamMetrics>,
}

/// VAD event payload inside an "events" message.
#[derive(Debug, Deserialize)]
struct SarvamEvent {
    signal_type: Option<String>,
}

// ---------------------------------------------------------------------------
// The provider
// ---------------------------------------------------------------------------

/// Sarvam's wire protocol. Holds configuration only — no session state.
pub struct SarvamProvider {
    config: SarvamSttConfig,
    audio: AudioSpec,
}

impl SarvamProvider {
    fn ws_path(&self) -> &'static str {
        if TRANSLATE_MODELS.contains(&self.config.model.as_str()) {
            STT_TRANSLATE_PATH
        } else {
            STT_PATH
        }
    }

    fn ws_url(&self) -> String {
        let c = &self.config;

        let mut params = vec![
            format!("model={}", percent_encode(&c.model)),
            format!("sample_rate={}", c.sample_rate),
            format!("input_audio_codec={}", percent_encode(&c.encoding)),
            "flush_signal=true".to_string(),
        ];

        // NOTE: language param uses hyphen: language-code (not language_code)
        if let Some(lang) = &c.language {
            if !TRANSLATE_MODELS.contains(&c.model.as_str()) {
                params.push(format!("language-code={}", percent_encode(lang)));
            }
        }

        if let Some(mode) = &c.mode {
            if MODE_MODELS.contains(&c.model.as_str()) {
                params.push(format!("mode={}", percent_encode(mode)));
            }
        }

        if let Some(prompt) = &c.prompt {
            params.push(format!("prompt={}", percent_encode(prompt)));
        }

        if c.high_vad_sensitivity {
            params.push("high_vad_sensitivity=true".to_string());
        }

        if c.vad_signals {
            params.push("vad_signals=true".to_string());
        }

        // Fine-grained VAD tuning is a saaras:v3 feature.
        if MODE_MODELS.contains(&c.model.as_str()) {
            c.vad_tuning.push_params(&mut params);
        }

        format!("{}{}?{}", SARVAM_BASE_WSS, self.ws_path(), params.join("&"))
    }
}

impl SttProvider for SarvamProvider {
    fn name(&self) -> &'static str {
        "SarvamStt"
    }

    fn audio(&self) -> &AudioSpec {
        &self.audio
    }

    fn handshake(&self) -> Handshake {
        // Auth: api-subscription-key header (lowercase), per SDK source.
        Handshake::new(self.ws_url()).header("api-subscription-key", &self.config.api_key)
    }

    /// Per AsyncAPI spec:
    /// `{"audio": {"data": <base64>, "sample_rate": "<rate>", "encoding": "audio/wav"}}`
    fn encode_audio(&self, pcm_le: &[u8]) -> Outgoing {
        use base64::engine::general_purpose::STANDARD as BASE64;
        use base64::Engine as _;

        let msg = serde_json::json!({
            "audio": {
                "data":        BASE64.encode(pcm_le),
                "sample_rate": self.config.sample_rate.to_string(),
                "encoding":    format!("audio/{}", self.config.encoding),
            }
        });
        Outgoing::Text(serde_json::to_string(&msg).unwrap_or_default())
    }

    /// Flush signal per AsyncAPI spec: `{"type": "flush"}`
    fn finalize_msg(&self) -> Option<Outgoing> {
        Some(Outgoing::Text(r#"{"type":"flush"}"#.to_string()))
    }

    fn parse(&self, msg: WsMessage<'_>) -> SttEvent {
        let text = match msg {
            WsMessage::Text(t) => t,
            WsMessage::Binary(_) => {
                log::debug!("SarvamStt: unexpected binary message");
                return SttEvent::Ignore;
            }
        };

        let envelope: SarvamMessage = match serde_json::from_str(text) {
            Ok(m) => m,
            Err(e) => {
                log::warn!("SarvamStt: parse error: {} — raw: {}", e, text);
                return SttEvent::Ignore;
            }
        };

        match envelope.msg_type.as_str() {
            "data" => {
                let Some(data) = envelope.data else {
                    return SttEvent::Ignore;
                };
                let t: SarvamTranscript = match serde_json::from_value(data) {
                    Ok(t) => t,
                    Err(e) => {
                        log::warn!("SarvamStt: transcript parse: {}", e);
                        return SttEvent::Ignore;
                    }
                };

                let audio_ms = t
                    .metrics
                    .as_ref()
                    .and_then(|m| m.audio_duration)
                    .map(|secs| secs * 1000.0);

                // Sarvam's streaming API only ever returns finals. An empty
                // or whitespace-only transcript is still the answer to our
                // flush and must close the turn, so it maps to EmptyFinal
                // rather than being ignored.
                let language = t.language_code.or_else(|| self.config.language.clone());

                match t.transcript {
                    Some(s) if !s.trim().is_empty() => {
                        SttEvent::Final { text: s, language, audio_ms }
                    }
                    _ => SttEvent::EmptyFinal { audio_ms },
                }
            }

            "events" => {
                let Some(data) = envelope.data else {
                    return SttEvent::Ignore;
                };
                let event: SarvamEvent = match serde_json::from_value(data) {
                    Ok(e) => e,
                    Err(e) => {
                        log::warn!("SarvamStt: event parse: {}", e);
                        return SttEvent::Ignore;
                    }
                };
                match event.signal_type.as_deref() {
                    Some("START_SPEECH") => SttEvent::SpeechStarted,
                    Some("END_SPEECH") => SttEvent::SpeechEnded,
                    other => {
                        log::debug!("SarvamStt: unknown event signal: {:?}", other);
                        SttEvent::Ignore
                    }
                }
            }

            "error" => SttEvent::Error(format!("{:?}", envelope.data)),

            other => {
                log::debug!("SarvamStt: unknown message type: {}", other);
                SttEvent::Ignore
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SarvamSttHandler — thin wrapper preserving the historical API
// ---------------------------------------------------------------------------

/// Sarvam STT as a pipeline component.
pub struct SarvamSttHandler {
    service: SttService<SarvamProvider>,
}

impl SarvamSttHandler {
    pub fn new(config: SarvamSttConfig) -> Self {
        let (core, provider) = config.split();
        Self { service: SttService::new(provider, core) }
    }

    pub fn with_billing(
        mut self,
        billing: std::sync::Arc<dyn crate::billing::BillingCollector>,
    ) -> Self {
        self.service = self.service.with_billing(billing);
        self
    }

    pub fn into_processor(self) -> FrameProcessor {
        self.service.into_processor()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(config: SarvamSttConfig) -> SarvamProvider {
        config.split().1
    }

    fn default_provider() -> SarvamProvider {
        provider(SarvamSttConfig::default())
    }

    // ---- URL construction ----------------------------------------------------

    #[test]
    fn ws_url_contains_model_and_sample_rate() {
        let url = default_provider().ws_url();
        assert!(url.contains("saaras"), "model missing: {url}");
        assert!(url.contains("16000"), "sample_rate missing: {url}");
        assert!(url.contains("flush_signal=true"), "flush_signal missing: {url}");
    }

    #[test]
    fn ws_url_translate_model_uses_translate_path() {
        let p = provider(SarvamSttConfig { model: "saaras:v2.5".into(), ..Default::default() });
        let url = p.ws_url();
        assert!(url.contains("speech-to-text-translate"));
        assert!(!url.contains("language-code"), "translate model takes no language: {url}");
    }

    #[test]
    fn ws_url_omits_mode_for_non_mode_models() {
        let p = provider(SarvamSttConfig {
            model: "saarika:v2.5".into(),
            mode: Some("transcribe".into()),
            ..Default::default()
        });
        assert!(!p.ws_url().contains("mode="), "mode is saaras:v3 only");
    }

    #[test]
    fn ws_url_includes_prompt_when_set() {
        let p = provider(SarvamSttConfig {
            prompt: Some("pizza toppings".into()),
            ..Default::default()
        });
        assert!(p.ws_url().contains("prompt=pizza%20toppings"), "{}", p.ws_url());
    }

    #[test]
    fn ws_url_includes_only_the_vad_params_that_are_set() {
        let p = provider(SarvamSttConfig {
            vad_tuning: SarvamVadTuning {
                min_speech_frames: Some(4),
                positive_speech_threshold: Some(0.6),
                ..Default::default()
            },
            ..Default::default()
        });
        let url = p.ws_url();
        assert!(url.contains("min_speech_frames=4"), "{url}");
        assert!(url.contains("positive_speech_threshold=0.6"), "{url}");
        assert!(!url.contains("pre_speech_pad_frames"), "unset params must be omitted: {url}");
    }

    #[test]
    fn ws_url_omits_vad_tuning_for_non_v3_models() {
        let p = provider(SarvamSttConfig {
            model: "saarika:v2.5".into(),
            vad_tuning: SarvamVadTuning { min_speech_frames: Some(4), ..Default::default() },
            ..Default::default()
        });
        assert!(!p.ws_url().contains("min_speech_frames"), "v3-only param leaked");
    }

    // ---- handshake -----------------------------------------------------------

    #[test]
    fn handshake_carries_the_subscription_key() {
        let p = provider(SarvamSttConfig { api_key: "sk-test".into(), ..Default::default() });
        let hs = p.handshake();
        assert!(hs.url.starts_with("wss://api.sarvam.ai/speech-to-text/ws?"));
        assert_eq!(
            hs.headers,
            vec![("api-subscription-key".to_string(), "sk-test".to_string())]
        );
    }

    // ---- audio framing -------------------------------------------------------

    #[test]
    fn encode_audio_matches_the_asyncapi_shape() {
        let p = default_provider();
        let Outgoing::Text(json) = p.encode_audio(&[0x01, 0x00, 0x02, 0x00]) else {
            panic!("Sarvam frames audio as text");
        };
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["audio"]["sample_rate"], "16000");
        assert_eq!(v["audio"]["encoding"], "audio/wav");
        assert_eq!(v["audio"]["data"], "AQACAA==");
    }

    #[test]
    fn finalize_is_the_flush_signal() {
        assert_eq!(
            default_provider().finalize_msg(),
            Some(Outgoing::Text(r#"{"type":"flush"}"#.into()))
        );
    }

    // ---- parsing -------------------------------------------------------------

    #[test]
    fn parse_data_message_yields_final_with_duration() {
        let p = default_provider();
        let raw = r#"{"type":"data","data":{"transcript":"namaskaram",
                     "language_code":"ml-IN","request_id":"req-1",
                     "metrics":{"audio_duration":1.84,"processing_latency":0.21}}}"#;
        match p.parse(WsMessage::Text(raw)) {
            SttEvent::Final { text, language, audio_ms } => {
                assert_eq!(text, "namaskaram");
                assert_eq!(language.as_deref(), Some("ml-IN"));
                assert!((audio_ms.unwrap() - 1840.0).abs() < 0.001);
            }
            other => panic!("expected Final, got {other:?}"),
        }
    }

    #[test]
    fn parse_falls_back_to_configured_language() {
        let p = provider(SarvamSttConfig { language: Some("ml-IN".into()), ..Default::default() });
        match p.parse(WsMessage::Text(r#"{"type":"data","data":{"transcript":"hi"}}"#)) {
            SttEvent::Final { language, audio_ms, .. } => {
                assert_eq!(language.as_deref(), Some("ml-IN"));
                assert!(audio_ms.is_none(), "no metrics means no server duration");
            }
            other => panic!("expected Final, got {other:?}"),
        }
    }

    #[test]
    fn parse_empty_transcript_yields_empty_final_so_the_turn_closes() {
        let p = default_provider();
        let raw = r#"{"type":"data","data":{"transcript":"   ",
                     "metrics":{"audio_duration":0.04}}}"#;
        match p.parse(WsMessage::Text(raw)) {
            SttEvent::EmptyFinal { audio_ms } => {
                assert!((audio_ms.unwrap() - 40.0).abs() < 0.001);
            }
            other => panic!("expected EmptyFinal, got {other:?}"),
        }
    }

    #[test]
    fn parse_vad_events() {
        let p = default_provider();
        assert_eq!(
            p.parse(WsMessage::Text(r#"{"type":"events","data":{"signal_type":"START_SPEECH"}}"#)),
            SttEvent::SpeechStarted
        );
        assert_eq!(
            p.parse(WsMessage::Text(r#"{"type":"events","data":{"signal_type":"END_SPEECH"}}"#)),
            SttEvent::SpeechEnded
        );
    }

    #[test]
    fn parse_error_and_garbage() {
        let p = default_provider();
        assert!(matches!(
            p.parse(WsMessage::Text(r#"{"type":"error","data":{"message":"bad key"}}"#)),
            SttEvent::Error(_)
        ));
        assert_eq!(p.parse(WsMessage::Text("not json")), SttEvent::Ignore);
        assert_eq!(p.parse(WsMessage::Binary(&[0, 1, 2])), SttEvent::Ignore);
    }

    // ---- config split --------------------------------------------------------

    #[test]
    fn split_carries_core_settings_through() {
        let (core, p) = SarvamSttConfig {
            sample_rate: 8_000,
            pre_roll_ms: 250,
            stop_release_timeout_ms: 900,
            audio_gating: false,
            noise_backend: NoiseBackend::HushVani,
            ..Default::default()
        }
        .split();

        assert_eq!(core.sample_rate, 8_000);
        assert_eq!(core.pre_roll_ms, 250);
        assert_eq!(core.stop_release_timeout_ms, 900);
        assert!(!core.audio_gating);
        assert_eq!(core.noise_backend, NoiseBackend::HushVani);
        assert_eq!(core.interim_policy, InterimPolicy::Drop);
        assert_eq!(p.audio().sample_rate, 8_000);
        assert_eq!(p.audio().encoding, "wav");
    }
}
