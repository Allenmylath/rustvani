use serde::Deserialize;
use serde_json::{json, Value};


pub const PROTOCOL_VERSION: &str = "1.2.0";
pub const MESSAGE_LABEL: &str = "ravi";

// ---------------------------------------------------------------------------
// Inbound (client → server)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct InboundMessage {
    pub label:    String,
    #[serde(rename = "type")]
    pub msg_type: String,
    pub id:       String,
    pub data:     Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct ClientReadyData {
    pub version: String,
    pub about:   Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct SendTextData {
    pub content: String,
    pub options: Option<SendTextOptions>,
}

#[derive(Debug, Deserialize)]
pub struct SendTextOptions {
    #[serde(default = "default_true")]
    pub run_immediately: bool,
    #[serde(default = "default_true")]
    pub audio_response:  bool,
}

fn default_true() -> bool { true }

impl Default for SendTextOptions {
    fn default() -> Self {
        Self { run_immediately: true, audio_response: true }
    }
}

// ---------------------------------------------------------------------------
// Outbound builders — return pre-serialised JSON strings
// ---------------------------------------------------------------------------

fn envelope(msg_type: &str, id: Option<&str>, data: Option<Value>) -> String {
    let mut obj = json!({ "label": MESSAGE_LABEL, "type": msg_type });
    if let Some(id) = id {
        obj["id"] = json!(id);
    }
    if let Some(d) = data {
        obj["data"] = d;
    }
    obj.to_string()
}

// ---- Handshake ----

pub fn msg_bot_ready(client_ready_id: &str, about: Option<Value>) -> String {
    envelope(
        "bot-ready",
        Some(client_ready_id),
        Some(json!({
            "version": PROTOCOL_VERSION,
            "about":   about.unwrap_or(Value::Null),
        })),
    )
}

pub fn msg_error_response(client_msg_id: &str, error: &str) -> String {
    envelope("error-response", Some(client_msg_id), Some(json!({ "error": error })))
}

pub fn msg_error(error: &str, fatal: bool) -> String {
    envelope("error", None, Some(json!({ "error": error, "fatal": fatal })))
}

// ---- Bot speaking ----

pub fn msg_bot_started_speaking() -> String { envelope("bot-started-speaking", None, None) }
pub fn msg_bot_stopped_speaking() -> String { envelope("bot-stopped-speaking", None, None) }

// ---- User speaking ----

pub fn msg_user_started_speaking() -> String { envelope("user-started-speaking", None, None) }
pub fn msg_user_stopped_speaking() -> String { envelope("user-stopped-speaking", None, None) }

// ---- User mute ----

pub fn msg_user_mute_started() -> String { envelope("user-mute-started", None, None) }
pub fn msg_user_mute_stopped() -> String { envelope("user-mute-stopped", None, None) }

// ---- Transcription ----

/// `language` is the transcript's language as the STT reported it (BCP-47, e.g. `ml-IN`), sent
/// as `data.language` when known: with auto-detection it can change phrase by phrase.
pub fn msg_user_transcription(
    text:      &str,
    user_id:   &str,
    timestamp: &str,
    is_final:  bool,
    language:  Option<&str>,
) -> String {
    let mut data = json!({
        "text":      text,
        "user_id":   user_id,
        "timestamp": timestamp,
        "final":     is_final,
    });
    if let Some(language) = language {
        data["language"] = json!(language);
    }
    envelope("user-transcription", None, Some(data))
}

// ---- Translation ----

/// A transcript translated into `target_language`, from a `TranslationFrame`: a final one, or an
/// interim one (`final: false`) the next replaces. `source_text` and `timestamp` are the
/// transcript's, so a client can place the translation under the `user-transcription` it belongs
/// to; `source_language`, `provider` and `latency_ms` are sent when known.
pub fn msg_user_translation(t: &crate::frames::TranslationData) -> String {
    let mut data = json!({
        "text":            t.text,
        "source_text":     t.source_text,
        "target_language": t.target_language,
        "user_id":         t.user_id,
        "timestamp":       t.timestamp,
        "final":           t.finalized,
    });
    if let Some(language) = &t.source_language {
        data["source_language"] = json!(language);
    }
    if let Some(provider) = &t.provider {
        data["provider"] = json!(provider);
    }
    if let Some(ms) = t.latency_ms {
        data["latency_ms"] = json!(ms);
    }
    envelope("user-translation", None, Some(data))
}

/// A transcript that could not be translated (`TranslationErrorFrame`): `kind` is `credits`,
/// `rate_limited`, `rejected`, `timeout` or `other`.
pub fn msg_user_translation_error(e: &crate::frames::TranslationErrorData) -> String {
    let data = json!({
        "source_text":     e.source_text,
        "source_language": e.source_language,
        "target_language": e.target_language,
        "user_id":         e.user_id,
        "timestamp":       e.timestamp,
        "final":           e.finalized,
        "provider":        e.provider,
        "error":           e.error,
        "kind":            e.kind.as_str(),
    });
    envelope("user-translation-error", None, Some(data))
}

/// What happened to a dubbed utterance (`DubEventFrame`): `event` is `synthesized`, `started`,
/// `ended`, `skipped` or `failed`; the timings and `rtp_timestamp` are sent when known.
pub fn msg_user_dub(d: &crate::frames::DubEventData) -> String {
    let mut data = json!({
        "event":           d.event.as_str(),
        "target_language": d.target_language,
        "text":            d.text,
        "source_text":     d.source_text,
        "user_id":         d.user_id,
        "timestamp":       d.timestamp,
    });
    for (key, value) in [("ready_ms", d.ready_ms), ("late_ms", d.late_ms), ("audio_ms", d.audio_ms)] {
        if let Some(v) = value {
            data[key] = json!(v);
        }
    }
    if let Some(ts) = d.rtp_timestamp {
        data["rtp_timestamp"] = json!(ts);
    }
    if let Some(error) = &d.error {
        data["error"] = json!(error);
    }
    if let Some(kind) = d.kind {
        data["kind"] = json!(kind.as_str());
    }
    envelope("user-dub", None, Some(data))
}

// ---- LLM ----

pub fn msg_bot_llm_started() -> String { envelope("bot-llm-started", None, None) }
pub fn msg_bot_llm_stopped() -> String { envelope("bot-llm-stopped", None, None) }

pub fn msg_bot_llm_text(text: &str) -> String {
    envelope("bot-llm-text", None, Some(json!({ "text": text })))
}

pub fn msg_bot_transcription(text: &str) -> String {
    envelope("bot-transcription", None, Some(json!({ "text": text })))
}

// ---- TTS ----

pub fn msg_bot_tts_started() -> String { envelope("bot-tts-started", None, None) }
pub fn msg_bot_tts_stopped() -> String { envelope("bot-tts-stopped", None, None) }

pub fn msg_bot_tts_text(text: &str) -> String {
    envelope("bot-tts-text", None, Some(json!({ "text": text })))
}

// ---- Custom ----

pub fn msg_server_message(data: Value) -> String {
    envelope("server-message", None, Some(data))
}

pub fn msg_server_response(client_msg_id: &str, msg_type: &str, data: Option<Value>) -> String {
    envelope(
        "server-response",
        Some(client_msg_id),
        Some(json!({ "t": msg_type, "d": data })),
    )
}

// ---- Audio levels ----

pub fn msg_user_audio_level(value: f32) -> String {
    envelope("user-audio-level", None, Some(json!({ "value": value })))
}

pub fn msg_bot_audio_level(value: f32) -> String {
    envelope("bot-audio-level", None, Some(json!({ "value": value })))
}

// ---- System log ----

pub fn msg_system_log(text: &str) -> String {
    envelope("system-log", None, Some(json!({ "text": text })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_transcription_carries_the_language_when_known() {
        let with: Value =
            serde_json::from_str(&msg_user_transcription("നാളെ", "u", "0", true, Some("ml-IN"))).unwrap();
        assert_eq!(with["type"], "user-transcription");
        assert_eq!(with["data"]["language"], "ml-IN");
        assert_eq!(with["data"]["final"], true);
        let without: Value = serde_json::from_str(&msg_user_transcription("hi", "u", "0", false, None)).unwrap();
        assert!(without["data"].get("language").is_none());
    }

    #[test]
    fn user_translation_names_both_languages_and_its_source() {
        use crate::frames::{TranscriptionData, TranslationData};
        let source = TranscriptionData::new("നാളെ", "u", "17.250").with_language("ml-IN").finalized();
        let mut t = TranslationData::new(&source, "en", "tomorrow");
        t.provider = Some("sarvam".into());
        t.latency_ms = Some(640);
        let msg: Value = serde_json::from_str(&msg_user_translation(&t)).unwrap();
        assert_eq!(msg["type"], "user-translation");
        assert_eq!(msg["data"]["text"], "tomorrow");
        assert_eq!(msg["data"]["source_text"], "നാളെ");
        assert_eq!(msg["data"]["source_language"], "ml-IN");
        assert_eq!(msg["data"]["target_language"], "en");
        assert_eq!(msg["data"]["timestamp"], "17.250");
        assert_eq!(msg["data"]["final"], true);
        assert_eq!((msg["data"]["provider"].clone(), msg["data"]["latency_ms"].clone()), (json!("sarvam"), json!(640)));
        let bare = TranslationData::new(&TranscriptionData::new("y", "u", "0"), "hi", "x");
        let unknown: Value = serde_json::from_str(&msg_user_translation(&bare)).unwrap();
        assert!(unknown["data"].get("source_language").is_none());
        assert!(unknown["data"].get("provider").is_none());
        assert_eq!(unknown["data"]["final"], false, "an interim transcript's translation is interim");
    }

    #[test]
    fn translation_errors_and_dub_events_say_what_happened() {
        use crate::frames::{DubEventData, DubEventKind, TranscriptionData, TranslationData, TranslationErrorData};
        let source = TranscriptionData::new("നാളെ", "u", "3").finalized();
        let e = TranslationErrorData::new(&source, "en", "sarvam", "SarvamTranslate: HTTP 402 Payment Required");
        let msg: Value = serde_json::from_str(&msg_user_translation_error(&e)).unwrap();
        assert_eq!(msg["type"], "user-translation-error");
        assert_eq!((msg["data"]["kind"].clone(), msg["data"]["timestamp"].clone()), (json!("credits"), json!("3")));

        let t = TranslationData::new(&source, "en", "tomorrow");
        let mut started = DubEventData::new(DubEventKind::Started, &t);
        started.late_ms = Some(120);
        started.rtp_timestamp = Some(48_000);
        let msg: Value = serde_json::from_str(&msg_user_dub(&started)).unwrap();
        assert_eq!(msg["type"], "user-dub");
        assert_eq!(msg["data"]["event"], "started");
        assert_eq!((msg["data"]["late_ms"].clone(), msg["data"]["rtp_timestamp"].clone()), (json!(120), json!(48_000)));
        assert!(msg["data"].get("error").is_none());
        let failed = DubEventData::new(DubEventKind::Failed, &t).failed("SarvamStreamTts: no final event");
        let msg: Value = serde_json::from_str(&msg_user_dub(&failed)).unwrap();
        assert_eq!((msg["data"]["event"].clone(), msg["data"]["kind"].clone()), (json!("failed"), json!("timeout")));
    }
}
