//! Deepgram streaming Speech-to-Text (`nova-3`) as a [`SttProvider`] on the shared STT core:
//! partial transcripts while the speaker talks, finals per segment, the detected language per
//! transcript, local turns, audio gating, the noise front end, a turn limit and reconnects —
//! what [`SarvamRealtimeSttHandler`](super::sarvam_realtime::SarvamRealtimeSttHandler) has, on
//! Deepgram. (The older [`DeepgramSttHandler`](super::deepgram::DeepgramSttHandler) sends finals
//! only and drives its own socket.)
//!
//! Wiring:
//!   let stt = DeepgramRealtimeSttHandler::new(DeepgramRealtimeSttConfig {
//!       api_key: std::env::var("DEEPGRAM_API_KEY").unwrap(),
//!       language: None, // detect: Deepgram's `multi`
//!       ..Default::default()
//!   })
//!   .into_processor();
//!
//! Protocol (developers.deepgram.com, "Live Audio"):
//!   URL:   wss://api.deepgram.com/v1/listen?model=…&language=…&encoding=linear16&sample_rate=…
//!   Auth:  Authorization: Token <key>
//!   Send:  binary PCM; {"type":"Finalize"}, {"type":"KeepAlive"}, {"type":"CloseStream"}
//!   Get:   Results (is_final, from_finalize, channel.alternatives[0].{transcript, languages}),
//!          Metadata, SpeechStarted, UtteranceEnd, Error

use crate::frames::FrameProcessor;

use super::core::util::percent_encode;
use super::core::{
    AudioSpec, Handshake, InterimPolicy, NoiseBackend, Outgoing, SttCoreConfig, SttEvent,
    SttProvider, SttService, TurnLimit, WsMessage,
};
use std::time::Duration;

const URL: &str = "wss://api.deepgram.com/v1/listen";
/// Deepgram closes a socket that hears nothing for 10 s; with audio gating nothing is sent
/// between turns.
const KEEPALIVE_EVERY: Duration = Duration::from_secs(5);

/// Configuration for [`DeepgramRealtimeSttHandler`].
#[derive(Debug, Clone)]
pub struct DeepgramRealtimeSttConfig {
    pub api_key: String,
    /// `nova-3` (default).
    pub model: String,
    /// BCP-47 (`hi-IN`, `en-IN`, …), sent as Deepgram names it (`hi`, `en-IN`); `None`,
    /// `"unknown"` or `"multi"` detect it per segment (`multi`, Deepgram's code-switching
    /// mode).
    pub language: Option<String>,
    /// Numbers, dates and the like formatted (`smart_format`), with punctuation.
    pub smart_format: bool,
    /// Server-side endpointing in ms; `None` (default) leaves turns to local VAD, which sends
    /// `Finalize` when one ends.
    pub endpointing: Option<u32>,
    /// Terms to boost (`keyterm`, nova-3).
    pub keyterms: Vec<String>,
    /// 8000 or 16000; audio is resampled to it by the core's front-end.
    pub sample_rate: u32,

    // ---- provider-independent (see SttCoreConfig) ----
    pub noise_reduction: bool,
    pub noise_backend: NoiseBackend,
    pub agc: bool,
    pub audio_gating: bool,
    pub pre_roll_ms: u32,
    pub stop_release_timeout_ms: u64,
    pub max_turn_ms: Option<u64>,
    /// A turn limit changeable while running; overrides `max_turn_ms`. See [`SttCoreConfig::turn_limit`].
    pub turn_limit: Option<TurnLimit>,
    /// Push partial transcripts downstream (default `true`).
    pub partials: bool,
}

impl Default for DeepgramRealtimeSttConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: "nova-3".to_string(),
            language: None,
            smart_format: true,
            endpointing: None,
            keyterms: Vec::new(),
            sample_rate: 16_000,
            noise_reduction: true,
            noise_backend: NoiseBackend::Rnnoise,
            agc: true,
            audio_gating: true,
            pre_roll_ms: 500,
            stop_release_timeout_ms: 1_200,
            max_turn_ms: None,
            turn_limit: None,
            partials: true,
        }
    }
}

impl DeepgramRealtimeSttConfig {
    fn split(self) -> (SttCoreConfig, DeepgramRealtimeProvider) {
        let core = SttCoreConfig {
            sample_rate: self.sample_rate,
            noise_reduction: self.noise_reduction,
            noise_backend: self.noise_backend,
            agc: self.agc,
            agc_config: Default::default(),
            audio_gating: self.audio_gating,
            pre_roll_ms: self.pre_roll_ms,
            stop_release_timeout_ms: self.stop_release_timeout_ms,
            interim_policy: if self.partials { InterimPolicy::Emit } else { InterimPolicy::Drop },
            max_turn_ms: self.max_turn_ms,
            turn_limit: self.turn_limit.clone(),
        };
        let audio = AudioSpec::new(self.sample_rate, "linear16");
        (core, DeepgramRealtimeProvider { config: self, audio })
    }
}

/// Deepgram's live wire protocol. Holds configuration only.
pub struct DeepgramRealtimeProvider {
    config: DeepgramRealtimeSttConfig,
    audio: AudioSpec,
}

impl DeepgramRealtimeProvider {
    /// The `language` parameter: `multi` to detect, else Deepgram's code (English keeps its
    /// region, `en-IN`; other languages go by their primary subtag, `hi`).
    fn language_param(&self) -> String {
        match self.config.language.as_deref() {
            None | Some("") | Some("unknown") | Some("multi") => "multi".to_string(),
            Some(lang) if lang.to_ascii_lowercase().starts_with("en") => lang.to_string(),
            Some(lang) => lang.split(['-', '_']).next().unwrap_or(lang).to_string(),
        }
    }

    fn ws_url(&self) -> String {
        let c = &self.config;
        let mut params = vec![
            format!("model={}", percent_encode(&c.model)),
            format!("language={}", percent_encode(&self.language_param())),
            "encoding=linear16".to_string(),
            format!("sample_rate={}", c.sample_rate),
            "channels=1".to_string(),
            "interim_results=true".to_string(),
            "punctuate=true".to_string(),
            format!("smart_format={}", c.smart_format),
            match c.endpointing {
                Some(ms) => format!("endpointing={ms}"),
                None => "endpointing=false".to_string(),
            },
        ];
        for term in &c.keyterms {
            params.push(format!("keyterm={}", percent_encode(term)));
        }
        format!("{URL}?{}", params.join("&"))
    }

    /// The configured language, when not detecting.
    fn fixed_language(&self) -> Option<String> {
        Some(self.language_param()).filter(|l| l != "multi")
    }
}

impl SttProvider for DeepgramRealtimeProvider {
    fn name(&self) -> &'static str {
        "DeepgramRealtimeStt"
    }

    fn audio(&self) -> &AudioSpec {
        &self.audio
    }

    fn handshake(&self) -> Handshake {
        Handshake::new(self.ws_url()).header("Authorization", format!("Token {}", self.config.api_key))
    }

    fn encode_audio(&self, pcm_le: &[u8]) -> Outgoing {
        Outgoing::Binary(pcm_le.to_vec())
    }

    fn finalize_msg(&self) -> Option<Outgoing> {
        Some(Outgoing::Text(r#"{"type":"Finalize"}"#.to_string()))
    }

    fn close_msg(&self) -> Option<Outgoing> {
        Some(Outgoing::Text(r#"{"type":"CloseStream"}"#.to_string()))
    }

    fn keepalive(&self) -> Option<(Duration, Outgoing)> {
        Some((KEEPALIVE_EVERY, Outgoing::Text(r#"{"type":"KeepAlive"}"#.to_string())))
    }

    fn parse(&self, msg: WsMessage<'_>) -> SttEvent {
        let WsMessage::Text(text) = msg else {
            log::debug!("DeepgramRealtimeStt: unexpected binary message");
            return SttEvent::Ignore;
        };
        let message: serde_json::Value = match serde_json::from_str(text) {
            Ok(m) => m,
            Err(e) => {
                log::warn!("DeepgramRealtimeStt: parse error: {e} — raw: {text}");
                return SttEvent::Ignore;
            }
        };
        match message.get("type").and_then(|t| t.as_str()).unwrap_or_default() {
            "Results" => {
                let alt = &message["channel"]["alternatives"][0];
                let transcript = alt["transcript"].as_str().unwrap_or_default().trim().to_string();
                let language = alt["languages"][0]
                    .as_str()
                    .or_else(|| message["metadata"]["detected_language"].as_str())
                    .map(str::to_string)
                    .or_else(|| self.fixed_language());
                let is_final = message["is_final"].as_bool().unwrap_or(false);
                let from_finalize = message["from_finalize"].as_bool().unwrap_or(false);
                let audio_ms = message["duration"].as_f64().map(|s| s * 1000.0);
                match (is_final, transcript.is_empty()) {
                    (false, true) => SttEvent::Ignore,
                    (false, false) => SttEvent::Partial { text: transcript, language },
                    (true, false) => SttEvent::Final { text: transcript, language, audio_ms },
                    // An empty final answers our Finalize (the turn closes) or is silence
                    // Deepgram finalised on its own (nothing to do).
                    (true, true) if from_finalize => SttEvent::EmptyFinal { audio_ms },
                    (true, true) => SttEvent::Ignore,
                }
            }
            "SpeechStarted" => SttEvent::SpeechStarted,
            "UtteranceEnd" => SttEvent::SpeechEnded,
            "Error" => SttEvent::Error(
                ["description", "message", "err_msg"]
                    .iter()
                    .find_map(|k| message[*k].as_str())
                    .map(str::to_string)
                    .unwrap_or_else(|| message.to_string()),
            ),
            other => {
                log::debug!("DeepgramRealtimeStt: {other}: {text}");
                SttEvent::Ignore
            }
        }
    }
}

/// Deepgram live STT as a pipeline component.
pub struct DeepgramRealtimeSttHandler {
    service: SttService<DeepgramRealtimeProvider>,
}

impl DeepgramRealtimeSttHandler {
    pub fn new(config: DeepgramRealtimeSttConfig) -> Self {
        let (core, provider) = config.split();
        Self { service: SttService::new(provider, core) }
    }

    pub fn with_billing(mut self, billing: std::sync::Arc<dyn crate::billing::BillingCollector>) -> Self {
        self.service = self.service.with_billing(billing);
        self
    }

    pub fn into_processor(self) -> FrameProcessor {
        self.service.into_processor()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(config: DeepgramRealtimeSttConfig) -> DeepgramRealtimeProvider {
        config.split().1
    }

    #[test]
    fn url_detects_by_default_and_names_languages_as_deepgram_does() {
        let url = provider(DeepgramRealtimeSttConfig::default()).ws_url();
        assert!(url.starts_with("wss://api.deepgram.com/v1/listen?"), "{url}");
        for part in ["model=nova-3", "language=multi", "encoding=linear16", "sample_rate=16000", "interim_results=true", "endpointing=false", "smart_format=true"] {
            assert!(url.contains(part), "{part} missing from {url}");
        }
        let lang = |l: &str| provider(DeepgramRealtimeSttConfig { language: Some(l.into()), ..Default::default() }).language_param();
        assert_eq!((lang("hi-IN"), lang("en-IN"), lang("unknown"), lang("ta_IN")), ("hi".into(), "en-IN".into(), "multi".into(), "ta".into()));
        let terms = provider(DeepgramRealtimeSttConfig { keyterms: vec!["Kochi drone".into()], endpointing: Some(300), ..Default::default() }).ws_url();
        assert!(terms.contains("keyterm=Kochi%20drone") && terms.contains("endpointing=300"), "{terms}");
    }

    #[test]
    fn audio_is_binary_and_control_messages_are_json() {
        let p = provider(DeepgramRealtimeSttConfig::default());
        assert_eq!(p.encode_audio(&[1, 2]), Outgoing::Binary(vec![1, 2]));
        assert_eq!(p.finalize_msg(), Some(Outgoing::Text(r#"{"type":"Finalize"}"#.into())));
        assert_eq!(p.mid_turn_finalize_msg(), p.finalize_msg());
        assert!(p.turn_start_msg().is_none());
        assert!(matches!(p.keepalive(), Some((d, Outgoing::Text(t))) if d == KEEPALIVE_EVERY && t.contains("KeepAlive")));
    }

    #[test]
    fn results_become_partials_finals_and_turn_closes() {
        let p = provider(DeepgramRealtimeSttConfig::default());
        let parse = |s: &str| p.parse(WsMessage::Text(s));
        let results = |final_: bool, from_finalize: bool, text: &str| {
            format!(
                r#"{{"type":"Results","is_final":{final_},"from_finalize":{from_finalize},"duration":1.5,"channel":{{"alternatives":[{{"transcript":"{text}","languages":["hi"]}}]}}}}"#
            )
        };
        assert_eq!(parse(&results(false, false, "नमस्ते")), SttEvent::Partial { text: "नमस्ते".into(), language: Some("hi".into()) });
        assert_eq!(parse(&results(true, false, "नमस्ते दुनिया")), SttEvent::Final { text: "नमस्ते दुनिया".into(), language: Some("hi".into()), audio_ms: Some(1500.0) });
        assert_eq!(parse(&results(true, true, "")), SttEvent::EmptyFinal { audio_ms: Some(1500.0) });
        assert_eq!(parse(&results(true, false, " ")), SttEvent::Ignore);
        assert_eq!(parse(&results(false, false, "")), SttEvent::Ignore);
        assert_eq!(parse(r#"{"type":"SpeechStarted"}"#), SttEvent::SpeechStarted);
        assert_eq!(parse(r#"{"type":"Metadata","request_id":"x"}"#), SttEvent::Ignore);
        assert!(matches!(parse(r#"{"type":"Error","description":"bad key"}"#), SttEvent::Error(m) if m == "bad key"));
        let fixed = provider(DeepgramRealtimeSttConfig { language: Some("en-IN".into()), ..Default::default() });
        let no_lang = r#"{"type":"Results","is_final":true,"channel":{"alternatives":[{"transcript":"hello"}]}}"#;
        assert_eq!(fixed.parse(WsMessage::Text(no_lang)), SttEvent::Final { text: "hello".into(), language: Some("en-IN".into()), audio_ms: None });
    }
}
