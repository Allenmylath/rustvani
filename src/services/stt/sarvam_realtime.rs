//! Sarvam AI realtime Speech-to-Text (`saaras:v3-realtime`) — a [`SttProvider`]
//! with **partial transcripts**: text arrives while the speaker is still
//! talking, and a final when the utterance ends.
//!
//! The classic streaming endpoint ([`super::sarvam`]) returns one final per
//! utterance, so a caption waits for the speaker to pause. This one streams
//! `transcript.partial` events as the words come in, which is what live
//! captions need. Partials reach the pipeline as `TranscriptionFrame`s with
//! `finalized: false` ([`InterimPolicy::Emit`]).
//!
//! Wiring:
//!   let stt = SarvamRealtimeSttHandler::new(SarvamRealtimeSttConfig {
//!       api_key: std::env::var("SARVAM_API_KEY").unwrap(),
//!       language: Some("ml-IN".into()),
//!       ..Default::default()
//!   })
//!   .into_processor();
//!
//! Protocol (docs.sarvam.ai, "Realtime Streaming"):
//!   URL:   wss://api.sarvam.ai/speech-to-text-realtime/ws?model=…&language_code=…&stream_type=…
//!   Auth:  api-subscription-key header
//!   Send:  {"event":"audio_input","audio":"<base64 PCM s16le>"}, {"event":"flush"},
//!          {"event":"ping"}, {"event":"end"}
//!   Get:   session.begin, vad.speech_start, vad.speech_end, transcript.partial,
//!          transcript.final, pong, session.end, error

use crate::frames::FrameProcessor;

use super::core::util::percent_encode;
use super::core::{
    AudioSpec, Handshake, InterimPolicy, NoiseBackend, Outgoing, SttCoreConfig, SttEvent,
    SttProvider, SttService, TurnLimit, WsMessage,
};
use std::time::Duration;

const URL: &str = "wss://api.sarvam.ai/speech-to-text-realtime/ws";
/// The server closes an idle connection; with audio gating nothing is sent between turns.
const PING_EVERY: Duration = Duration::from_secs(5);

/// Configuration for [`SarvamRealtimeSttHandler`].
#[derive(Debug, Clone)]
pub struct SarvamRealtimeSttConfig {
    pub api_key: String,
    /// `saaras:v3-realtime` (default) or `saaras:v4`.
    pub model: String,
    /// BCP-47 (`ml-IN`, `hi-IN`, `en-IN`, …); `None` or `"unknown"` detects it (`auto`).
    pub language: Option<String>,
    /// `fast` (lowest partial latency), `balanced` (Sarvam's default: a little slower, more
    /// accurate), or `simulated` (no partials).
    pub stream_type: String,
    /// `transcribe` (default), `translate`, `verbatim`, `translit`, `codemix`; applies to finals.
    pub mode: Option<String>,
    /// Terminology hint for finals.
    pub prompt: Option<String>,
    /// Who ends an utterance. `manual` (default): local VAD does — `speech_start`
    /// when a turn opens, `speech_end` when it closes; a mid-turn
    /// [`max_turn_ms`](Self::max_turn_ms) cut is `speech_end` then `speech_start`
    /// (a `flush` alone ends the utterance, and Sarvam drops the audio after it). `vad`: Sarvam's server VAD, after
    /// `silence_duration_ms` of silence; it ignores `flush`, and with audio gating
    /// it hears no silence after a turn, so an utterance only ends when the next
    /// begins (measured: finals 7 s late, 10–18 s utterances on news speech).
    pub endpointing: String,
    /// Server VAD silence that ends an utterance (Sarvam's default 500).
    pub silence_duration_ms: Option<u32>,
    /// Server VAD threshold, 0–1 (Sarvam's default 0.3).
    pub threshold: Option<f32>,
    /// Server VAD minimum speech (Sarvam's default 250).
    pub min_speech_duration_ms: Option<u32>,
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
    /// Push partial transcripts downstream (default `true`: the reason to use this endpoint).
    pub partials: bool,
}

impl Default for SarvamRealtimeSttConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: "saaras:v3-realtime".to_string(),
            language: Some("unknown".to_string()),
            stream_type: "fast".to_string(),
            mode: None,
            prompt: None,
            endpointing: "manual".to_string(),
            silence_duration_ms: None,
            threshold: None,
            min_speech_duration_ms: None,
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

impl SarvamRealtimeSttConfig {
    fn split(self) -> (SttCoreConfig, SarvamRealtimeProvider) {
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
        (core, SarvamRealtimeProvider { config: self, audio })
    }
}

/// Sarvam's realtime wire protocol. Holds configuration only.
pub struct SarvamRealtimeProvider {
    config: SarvamRealtimeSttConfig,
    audio: AudioSpec,
}

impl SarvamRealtimeProvider {
    fn language_code(&self) -> String {
        match self.config.language.as_deref() {
            None | Some("unknown") | Some("") => "auto".to_string(),
            Some(lang) => lang.to_string(),
        }
    }

    fn ws_url(&self) -> String {
        let c = &self.config;
        let mut params = vec![
            format!("model={}", percent_encode(&c.model)),
            format!("language_code={}", percent_encode(&self.language_code())),
            format!("stream_type={}", percent_encode(&c.stream_type)),
            format!("endpointing={}", percent_encode(&c.endpointing)),
            "encoding=linear16".to_string(),
            format!("sample_rate={}", c.sample_rate),
        ];
        if let Some(mode) = &c.mode {
            params.push(format!("mode={}", percent_encode(mode)));
        }
        if let Some(prompt) = &c.prompt {
            params.push(format!("prompt={}", percent_encode(prompt)));
        }
        if let Some(ms) = c.silence_duration_ms {
            params.push(format!("silence_duration_ms={ms}"));
        }
        if let Some(t) = c.threshold {
            params.push(format!("threshold={t}"));
        }
        if let Some(ms) = c.min_speech_duration_ms {
            params.push(format!("min_speech_duration_ms={ms}"));
        }
        format!("{URL}?{}", params.join("&"))
    }

    fn manual(&self) -> bool {
        self.config.endpointing == "manual"
    }

    /// The configured language, when not detecting.
    fn fixed_language(&self) -> Option<String> {
        Some(self.language_code()).filter(|l| l != "auto")
    }
}

/// A field at the top level or under `data` (the docs show both shapes).
fn field<'a>(message: &'a serde_json::Value, name: &str) -> Option<&'a serde_json::Value> {
    message.get(name).or_else(|| message.get("data").and_then(|d| d.get(name)))
}

impl SttProvider for SarvamRealtimeProvider {
    fn name(&self) -> &'static str {
        "SarvamRealtimeStt"
    }

    fn audio(&self) -> &AudioSpec {
        &self.audio
    }

    fn handshake(&self) -> Handshake {
        Handshake::new(self.ws_url()).header("api-subscription-key", &self.config.api_key)
    }

    fn encode_audio(&self, pcm_le: &[u8]) -> Outgoing {
        use base64::engine::general_purpose::STANDARD as BASE64;
        use base64::Engine as _;
        Outgoing::Text(
            serde_json::json!({ "event": "audio_input", "audio": BASE64.encode(pcm_le) }).to_string(),
        )
    }

    fn turn_start_msg(&self) -> Option<Outgoing> {
        self.manual().then(|| Outgoing::Text(r#"{"event":"speech_start"}"#.to_string()))
    }

    fn finalize_msg(&self) -> Option<Outgoing> {
        Some(Outgoing::Text(if self.manual() {
            r#"{"event":"speech_end"}"#.to_string()
        } else {
            r#"{"event":"flush"}"#.to_string()
        }))
    }


    fn close_msg(&self) -> Option<Outgoing> {
        Some(Outgoing::Text(r#"{"event":"end"}"#.to_string()))
    }

    fn keepalive(&self) -> Option<(Duration, Outgoing)> {
        Some((PING_EVERY, Outgoing::Text(r#"{"event":"ping"}"#.to_string())))
    }

    fn parse(&self, msg: WsMessage<'_>) -> SttEvent {
        let WsMessage::Text(text) = msg else {
            log::debug!("SarvamRealtimeStt: unexpected binary message");
            return SttEvent::Ignore;
        };
        let message: serde_json::Value = match serde_json::from_str(text) {
            Ok(m) => m,
            Err(e) => {
                log::warn!("SarvamRealtimeStt: parse error: {e} — raw: {text}");
                return SttEvent::Ignore;
            }
        };
        let kind = message
            .get("type")
            .or_else(|| message.get("event"))
            .and_then(|k| k.as_str())
            .unwrap_or_default();
        let transcript = || field(&message, "text").and_then(|t| t.as_str()).unwrap_or_default().trim().to_string();
        let language = || {
            field(&message, "language")
                .and_then(|l| l.as_str())
                .map(str::to_string)
                .or_else(|| self.fixed_language())
        };
        match kind {
            "transcript.partial" => {
                let text = transcript();
                if text.is_empty() {
                    SttEvent::Ignore
                } else {
                    SttEvent::Partial { text, language: language() }
                }
            }
            "transcript.final" => {
                let text = transcript();
                if text.is_empty() {
                    SttEvent::EmptyFinal { audio_ms: None }
                } else {
                    SttEvent::Final { text, language: language(), audio_ms: None }
                }
            }
            "vad.speech_start" => SttEvent::SpeechStarted,
            "vad.speech_end" => SttEvent::SpeechEnded,
            "error" => SttEvent::Error(
                field(&message, "message")
                    .and_then(|m| m.as_str())
                    .map(str::to_string)
                    .unwrap_or_else(|| message.to_string()),
            ),
            "session.begin" | "session.end" | "pong" | "config.updated" => {
                log::debug!("SarvamRealtimeStt: {kind}: {text}");
                SttEvent::Ignore
            }
            other => {
                log::debug!("SarvamRealtimeStt: unknown message {other:?}: {text}");
                SttEvent::Ignore
            }
        }
    }
}

/// Sarvam realtime STT as a pipeline component.
pub struct SarvamRealtimeSttHandler {
    service: SttService<SarvamRealtimeProvider>,
}

impl SarvamRealtimeSttHandler {
    pub fn new(config: SarvamRealtimeSttConfig) -> Self {
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

    fn provider(config: SarvamRealtimeSttConfig) -> SarvamRealtimeProvider {
        config.split().1
    }

    #[test]
    fn url_carries_the_realtime_parameters() {
        let p = provider(SarvamRealtimeSttConfig {
            language: Some("ml-IN".into()),
            silence_duration_ms: Some(300),
            ..Default::default()
        });
        let url = p.ws_url();
        assert!(url.starts_with("wss://api.sarvam.ai/speech-to-text-realtime/ws?"), "{url}");
        for part in ["model=saaras%3Av3-realtime", "language_code=ml-IN", "stream_type=fast", "endpointing=manual", "encoding=linear16", "sample_rate=16000", "silence_duration_ms=300"] {
            assert!(url.contains(part), "{part} missing from {url}");
        }
        let auto = provider(SarvamRealtimeSttConfig::default()).ws_url();
        assert!(auto.contains("language_code=auto"));
    }

    #[test]
    fn manual_endpointing_follows_local_turns() {
        let text = |m: Option<Outgoing>| match m {
            Some(Outgoing::Text(t)) => serde_json::from_str::<serde_json::Value>(&t).unwrap()["event"].as_str().unwrap().to_string(),
            other => panic!("{other:?}"),
        };
        let manual = provider(SarvamRealtimeSttConfig::default());
        assert_eq!(text(manual.turn_start_msg()), "speech_start");
        assert_eq!(text(manual.finalize_msg()), "speech_end");
        assert_eq!(text(manual.mid_turn_finalize_msg()), "speech_end");
        let server = provider(SarvamRealtimeSttConfig { endpointing: "vad".into(), ..Default::default() });
        assert!(server.turn_start_msg().is_none());
        assert_eq!(text(server.finalize_msg()), "flush");
    }

    #[test]
    fn audio_goes_as_base64_json() {
        let p = provider(SarvamRealtimeSttConfig::default());
        let Outgoing::Text(text) = p.encode_audio(&[1, 2, 3, 4]) else { panic!("binary") };
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["event"], "audio_input");
        assert_eq!(v["audio"], "AQIDBA==");
    }

    #[test]
    fn partials_finals_and_events_parse_in_either_shape() {
        let p = provider(SarvamRealtimeSttConfig { language: Some("ml-IN".into()), ..Default::default() });
        let parse = |s: &str| p.parse(WsMessage::Text(s));
        assert_eq!(
            parse(r#"{"type":"transcript.partial","text":"നാളെ"}"#),
            SttEvent::Partial { text: "നാളെ".into(), language: Some("ml-IN".into()) }
        );
        assert_eq!(
            parse(r#"{"event":"transcript.final","data":{"text":"നാളെ മീറ്റിംഗ്","language":"ml-IN"}}"#),
            SttEvent::Final { text: "നാളെ മീറ്റിംഗ്".into(), language: Some("ml-IN".into()), audio_ms: None }
        );
        assert_eq!(parse(r#"{"type":"transcript.final","text":"  "}"#), SttEvent::EmptyFinal { audio_ms: None });
        assert_eq!(parse(r#"{"type":"vad.speech_start"}"#), SttEvent::SpeechStarted);
        assert_eq!(parse(r#"{"type":"pong"}"#), SttEvent::Ignore);
        assert!(matches!(parse(r#"{"type":"error","message":"quota","code":1003}"#), SttEvent::Error(m) if m == "quota"));
    }
}
