//! Signaling protocol for `vaniwebrtc`: SDP offer/answer + trickle ICE over a
//! WebSocket, plus Opus `fmtp` munging of the answer SDP.

use serde::{Deserialize, Serialize};

use super::params::VaniWebRTCParams;

/// Messages exchanged over the signaling WebSocket (tagged JSON).
///
/// Field names match the browser `RTCIceCandidate` / `RTCSessionDescription`
/// JSON shape so a vanilla WebRTC client needs no translation layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SignalMsg {
    /// Client → server: SDP offer.
    Offer { sdp: String },
    /// Server → client: SDP answer.
    Answer { sdp: String },
    /// Either direction: a trickled ICE candidate.
    Ice {
        candidate: String,
        #[serde(rename = "sdpMid", default, skip_serializing_if = "Option::is_none")]
        sdp_mid: Option<String>,
        #[serde(rename = "sdpMLineIndex", default, skip_serializing_if = "Option::is_none")]
        sdp_mline_index: Option<u16>,
    },
    /// Either direction: graceful close.
    Bye,
}

/// Rewrite the Opus `a=fmtp` line of an answer SDP to force high-bitrate,
/// full-band Opus from the remote encoder, so the inbound 48 kHz denoise stage
/// (DeepFilterNet) isn't starved of high frequencies.
///
/// Idempotent and conservative: if no Opus codec is present the SDP is returned
/// unchanged. Existing `fmtp` parameters are preserved and ours appended.
pub fn munge_answer_sdp(sdp: &str, params: &VaniWebRTCParams) -> String {
    // Find the dynamic payload type for Opus from its rtpmap line.
    let opus_pt = sdp.lines().find_map(|line| {
        let rest = line.strip_prefix("a=rtpmap:")?;
        rest.to_ascii_lowercase()
            .contains("opus/48000")
            .then(|| rest.split_whitespace().next())
            .flatten()
    });
    let Some(pt) = opus_pt else {
        return sdp.to_string();
    };

    let extra = build_fmtp_params(params);
    let fmtp_prefix = format!("a=fmtp:{} ", pt);
    let rtpmap_prefix = format!("a=rtpmap:{} ", pt);
    let has_fmtp = sdp.lines().any(|l| l.starts_with(&fmtp_prefix));

    let mut out = String::with_capacity(sdp.len() + extra.len() + 16);
    for line in sdp.lines() {
        if has_fmtp && line.starts_with(&fmtp_prefix) {
            let mut merged = line.trim_end().to_string();
            if !merged.ends_with(';') {
                merged.push(';');
            }
            merged.push_str(&extra);
            out.push_str(&merged);
            out.push_str("\r\n");
        } else {
            out.push_str(line);
            out.push_str("\r\n");
            // No existing fmtp → synthesise one right after the rtpmap line.
            if !has_fmtp && line.starts_with(&rtpmap_prefix) {
                out.push_str(&format!("a=fmtp:{} {}\r\n", pt, extra));
            }
        }
    }
    out
}

fn build_fmtp_params(p: &VaniWebRTCParams) -> String {
    let mut parts = vec![
        format!("maxaveragebitrate={}", p.opus_max_avg_bitrate),
        "stereo=0".to_string(),
        format!("usedtx={}", u8::from(p.opus_dtx)),
        "cbr=0".to_string(),
    ];
    if p.opus_fullband {
        parts.push("maxplaybackrate=48000".to_string());
    }
    parts.join(";")
}
