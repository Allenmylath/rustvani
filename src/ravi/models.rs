/// RAVI — Real-time Audio Voice Interface protocol message models.
///
/// Inbound types are plain structs derived from `Deserialize`.
/// Outbound messages are produced by builder functions that return a
/// pre-serialised `String` ready to drop into a `Frame::ravi_server_message`.
///
/// Using builder functions rather than typed structs for outbound messages
/// keeps the serialisation concern in one place and avoids a proliferation of
/// single-use wrapper types.
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const PROTOCOL_VERSION: &str = "1.2.0";
pub const MESSAGE_LABEL: &str = "ravi";

// ---------------------------------------------------------------------------
// Inbound (client → server)
// ---------------------------------------------------------------------------

/// Top-level envelope for any inbound RAVI client message.
#[derive(Debug, Deserialize)]
pub struct InboundMessage {
    pub label:    String,
    #[serde(rename = "type")]
    pub msg_type: String,
    pub id:       String,
    pub data:     Option<Value>,
}

/// Data payload of the `client-ready` message.
#[derive(Debug, Deserialize)]
pub struct ClientReadyData {
    pub version: String,
    pub about:   Option<Value>,
}

/// Data payload of the `send-text` message.
#[derive(Debug, Deserialize)]
pub struct SendTextData {
    pub content: String,
    pub options: Option<SendTextOptions>,
}

#[derive(Debug, Deserialize)]
pub struct SendTextOptions {
    /// If true, interrupt the bot and run the LLM immediately.
    #[serde(default = "default_true")]
    pub run_immediately: bool,
    /// If true, produce a spoken (TTS) response.
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
// Outbound builder helpers
// ---------------------------------------------------------------------------
//
// Every builder produces a String that is already valid JSON — the full RAVI
// envelope including `label`, `type`, optional `id`, and optional `data`.
//
// Naming convention: `msg_<snake_case_type>`.

#[inline]
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

/// `bot-ready` — sent in response to `client-ready`.
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

/// `error-response` — sent when a client request cannot be fulfilled.
pub fn msg_error_response(client_msg_id: &str, error: &str) -> String {
    envelope(
        "error-response",
        Some(client_msg_id),
        Some(json!({ "error": error })),
    )
}

/// `error` — pipeline-level error (may be fatal).
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

/// `user-transcription` — emitted for each recognised user utterance.
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

/// `bot-llm-text` — one streamed token from the LLM.
pub fn msg_bot_llm_text(text: &str) -> String {
    envelope("bot-llm-text", None, Some(json!({ "text": text })))
}

/// `bot-transcription` — complete LLM sentence (deprecated in RTVI 1.2 but
/// kept for backward-compatibility with older clients).
pub fn msg_bot_transcription(text: &str) -> String {
    envelope("bot-transcription", None, Some(json!({ "text": text })))
}

// ---- TTS ----

pub fn msg_bot_tts_started() -> String { envelope("bot-tts-started", None, None) }
pub fn msg_bot_tts_stopped() -> String { envelope("bot-tts-stopped", None, None) }

/// `bot-tts-text` — text chunk being sent to TTS.
pub fn msg_bot_tts_text(text: &str) -> String {
    envelope("bot-tts-text", None, Some(json!({ "text": text })))
}

// ---- Custom server messages / responses ----

/// `server-message` — arbitrary data pushed from the server to the client.
pub fn msg_server_message(data: Value) -> String {
    envelope("server-message", None, Some(data))
}

/// `server-response` — reply to a `client-message` request.
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

// ---------------------------------------------------------------------------
// Inbound parsing helper
// ---------------------------------------------------------------------------

/// Parse the raw JSON string stored in `RaviClientMessage.data` into an
/// `InboundMessage`.  Returns `None` and logs a warning on failure.
pub fn parse_inbound(raw: &str) -> Option<InboundMessage> {
    match serde_json::from_str(raw) {
        Ok(m) => Some(m),
        Err(e) => {
            log::warn!("RAVI: failed to parse inbound message: {}", e);
            None
        }
    }
}
