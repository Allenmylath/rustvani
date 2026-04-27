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

pub fn msg_user_transcription(
    text:      &str,
    user_id:   &str,
    timestamp: &str,
    is_final:  bool,
) -> String {
    envelope(
        "user-transcription",
        None,
        Some(json!({
            "text":      text,
            "user_id":   user_id,
            "timestamp": timestamp,
            "final":     is_final,
        })),
    )
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
