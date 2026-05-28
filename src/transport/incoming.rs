use tokio::sync::mpsc;

use crate::frames::{Frame, FrameDirection};

/// Parse an incoming text message and push the appropriate frame(s).
///
/// Recognised protocols / message types:
///
/// | `label` | `type`                   | Action                              |
/// |---------|--------------------------|-------------------------------------|
/// | `"ravi"`| any                      | `RaviClientMessage` downstream      |
/// | —       | `"client_interruption"`  | `InterruptionFrame` downstream      |
/// | —       | `"client_vad_start"`     | `ClientVADUserStartedSpeaking`      |
/// | —       | `"client_vad_stop"`      | `ClientVADUserStoppedSpeaking`      |
///
/// The optional `"timestamp"` field (f64 seconds) is read for VAD messages.
/// If absent or unparseable, `0.0` is used.
///
/// This function is transport-agnostic: every transport that accepts text
/// messages from a client should route them through here so new message types
/// only need to be added in one place.
pub async fn dispatch_text_message(
    text: &str,
    push_tx: &mpsc::Sender<(Frame, FrameDirection)>,
) {
    let Ok(msg) = serde_json::from_str::<serde_json::Value>(text) else {
        log::warn!("transport: ignoring non-JSON text message");
        return;
    };

    let msg_type = msg.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let label    = msg.get("label").and_then(|v| v.as_str()).unwrap_or("");

    // ---- RAVI protocol ----
    if label == "ravi" {
        let Some(msg_id) = msg.get("id").and_then(|v| v.as_str()) else {
            log::warn!("transport: RAVI message missing 'id' field — dropping");
            return;
        };

        let data_str = msg.get("data").map(|d| d.to_string());
        let frame = Frame::ravi_client_message(msg_id, msg_type, data_str);
        let _ = push_tx.send((frame, FrameDirection::Downstream)).await;

        log::trace!("transport: RAVI '{}' (id={})", msg_type, msg_id);
        return;
    }

    // ---- Bare (non-RAVI) message types ----
    match msg_type {
        "client_vad_start" => {
            let ts = timestamp_from_msg(&msg);
            log::info!("transport: client VAD start (ts={})", ts);
            let _ = push_tx
                .send((Frame::client_vad_user_started_speaking(ts), FrameDirection::Downstream))
                .await;
        }
        "client_vad_stop" => {
            let ts = timestamp_from_msg(&msg);
            log::info!("transport: client VAD stop (ts={})", ts);
            let _ = push_tx
                .send((Frame::client_vad_user_stopped_speaking(ts), FrameDirection::Downstream))
                .await;
        }
        "client_interruption" => {
            log::info!("transport: legacy client-initiated interruption");
            let _ = push_tx
                .send((Frame::interruption(), FrameDirection::Downstream))
                .await;
        }
        other => {
            log::warn!("transport: unknown text message type '{}' — ignoring", other);
        }
    }
}

fn timestamp_from_msg(msg: &serde_json::Value) -> f64 {
    msg.get("timestamp").and_then(|v| v.as_f64()).unwrap_or(0.0)
}
