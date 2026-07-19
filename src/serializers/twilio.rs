//! Twilio Media Streams WebSocket protocol serializer.
//!
//! Ported from pipecat's `pipecat/serializers/twilio.py`. Converts between
//! rustvani [`Frame`]s and Twilio's Media Streams JSON protocol: 8 kHz µ-law
//! audio (base64 in `media` events), `clear` barge-in events, inbound `dtmf`
//! events, and automatic call hang-up via Twilio's REST API on end/cancel.

use async_trait::async_trait;
use base64::Engine;
use serde_json::json;

use crate::audio_process::resamplers::{ResamplerQuality, StreamResampler};
use crate::error::{PipecatError, Result};
use crate::frames::{ControlFrame, DataFrame, Frame, FrameInner, KeypadEntry, SystemFrame};

use super::g711::{pcm_to_ulaw, ulaw_to_pcm};
use super::{FrameSerializer, SerializedInput, SerializedOutput};

/// Resampler quality for the 8 kHz ↔ pipeline-rate conversions. Telephony audio
/// is narrowband, so a balanced preset keeps latency low without audible loss.
const RESAMPLER_QUALITY: ResamplerQuality = ResamplerQuality::Medium;

// ---------------------------------------------------------------------------
// InputParams
// ---------------------------------------------------------------------------

/// Configuration for [`TwilioFrameSerializer`], mirroring pipecat's
/// `TwilioFrameSerializer.InputParams`.
#[derive(Debug, Clone)]
pub struct TwilioInputParams {
    /// Sample rate used by Twilio on the wire. Defaults to 8000 Hz.
    pub twilio_sample_rate: u32,
    /// Optional override for the pipeline input sample rate. When `None`, the
    /// rate passed to [`setup`](FrameSerializer::setup) is used.
    pub sample_rate: Option<u32>,
    /// Whether to terminate the Twilio call via REST on End/Cancel.
    pub auto_hang_up: bool,
    /// Seconds of inactivity after which a streaming resampler would clear its
    /// history. Retained for parity with pipecat; rustvani's resampler does not
    /// currently expose stale-history clearing.
    pub resampler_clear_after_secs: Option<f32>,
}

impl Default for TwilioInputParams {
    fn default() -> Self {
        Self {
            twilio_sample_rate: 8000,
            sample_rate: None,
            auto_hang_up: true,
            resampler_clear_after_secs: Some(0.2),
        }
    }
}

// ---------------------------------------------------------------------------
// TwilioStart — parsed `start` handshake message
// ---------------------------------------------------------------------------

/// The identifiers Twilio sends in its initial `start` event.
///
/// A Twilio Media Stream opens with a `connected` event, then a `start` event
/// carrying the SIDs. pipecat constructs the serializer from these; the
/// transport/route layer reads them off the socket via [`TwilioStart::parse`]
/// before building a [`TwilioFrameSerializer`].
#[derive(Debug, Clone)]
pub struct TwilioStart {
    pub stream_sid: String,
    pub call_sid: Option<String>,
    pub account_sid: Option<String>,
}

impl TwilioStart {
    /// Parse a Twilio WebSocket text message, returning `Some` only for a
    /// `start` event.
    pub fn parse(text: &str) -> Option<Self> {
        let msg: serde_json::Value = serde_json::from_str(text).ok()?;
        if msg.get("event")?.as_str()? != "start" {
            return None;
        }
        let start = msg.get("start")?;
        let stream_sid = start
            .get("streamSid")
            .or_else(|| msg.get("streamSid"))
            .and_then(|v| v.as_str())?
            .to_string();
        Some(Self {
            stream_sid,
            call_sid: start.get("callSid").and_then(|v| v.as_str()).map(String::from),
            account_sid: start.get("accountSid").and_then(|v| v.as_str()).map(String::from),
        })
    }
}

// ---------------------------------------------------------------------------
// URL builder for auto hang-up
// ---------------------------------------------------------------------------

/// Build the REST URL for a Call resource (used by auto hang-up).
///
/// With `base_url` set it is used verbatim as the API root; otherwise the host
/// is derived from Twilio's `region`/`edge` FQDN format. Mirrors
/// `_build_call_resource_url` in pipecat.
fn build_call_resource_url(
    account_sid: &str,
    call_sid: &str,
    base_url: Option<&str>,
    region: Option<&str>,
    edge: Option<&str>,
) -> String {
    let root = match base_url {
        Some(b) => b.trim_end_matches('/').to_string(),
        None => {
            let region_prefix = region.map(|r| format!("{r}.")).unwrap_or_default();
            let edge_prefix = edge.map(|e| format!("{e}.")).unwrap_or_default();
            format!("https://api.{edge_prefix}{region_prefix}twilio.com")
        }
    };
    format!("{root}/2010-04-01/Accounts/{account_sid}/Calls/{call_sid}.json")
}

// ---------------------------------------------------------------------------
// TwilioFrameSerializer
// ---------------------------------------------------------------------------

/// Serializer for the Twilio Media Streams WebSocket protocol.
///
/// When `auto_hang_up` is enabled (default) the serializer terminates the
/// Twilio call via REST when an End/Cancel frame is serialized, and therefore
/// requires the call/account SIDs and auth token at construction.
pub struct TwilioFrameSerializer {
    stream_sid: String,
    call_sid: Option<String>,
    account_sid: Option<String>,
    auth_token: Option<String>,
    region: Option<String>,
    edge: Option<String>,
    base_url: Option<String>,

    params: TwilioInputParams,
    twilio_sample_rate: u32,
    /// Pipeline input rate; set in [`setup`](FrameSerializer::setup).
    sample_rate: u32,

    /// 8 kHz → pipeline rate, for inbound audio. `None` when rates are equal.
    input_resampler: Option<StreamResampler>,
    /// (source rate, pipeline rate → 8 kHz), for outbound audio. Built lazily on
    /// the first audio frame; rebuilt if a later frame has a different rate.
    output_resampler: Option<(u32, StreamResampler)>,

    hangup_attempted: bool,
}

impl TwilioFrameSerializer {
    /// Construct a Twilio serializer.
    ///
    /// Replicates pipecat's constructor validation: when `auto_hang_up` is
    /// enabled, `call_sid`/`account_sid`/`auth_token` are required, and
    /// `region`/`edge` must be set together (unless `base_url` is given).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stream_sid: impl Into<String>,
        call_sid: Option<String>,
        account_sid: Option<String>,
        auth_token: Option<String>,
        region: Option<String>,
        edge: Option<String>,
        base_url: Option<String>,
        params: TwilioInputParams,
    ) -> Result<Self> {
        if params.auto_hang_up {
            let mut missing = Vec::new();
            if call_sid.is_none() {
                missing.push("call_sid");
            }
            if account_sid.is_none() {
                missing.push("account_sid");
            }
            if auth_token.is_none() {
                missing.push("auth_token");
            }
            if !missing.is_empty() {
                return Err(PipecatError::pipeline(format!(
                    "auto_hang_up is enabled but missing required parameters: {}",
                    missing.join(", ")
                )));
            }

            if base_url.is_none()
                && ((region.is_some() && edge.is_none()) || (edge.is_some() && region.is_none()))
            {
                return Err(PipecatError::pipeline(format!(
                    "Both edge and region parameters are required if one is set. \
                     Twilio's FQDN format requires both: api.{{edge}}.{{region}}.twilio.com. \
                     Got: region={region:?}, edge={edge:?}"
                )));
            }
        }

        let twilio_sample_rate = params.twilio_sample_rate;

        Ok(Self {
            stream_sid: stream_sid.into(),
            call_sid,
            account_sid,
            auth_token,
            region,
            edge,
            base_url,
            params,
            twilio_sample_rate,
            sample_rate: 0,
            input_resampler: None,
            output_resampler: None,
            hangup_attempted: false,
        })
    }

    /// Convenience constructor from a parsed [`TwilioStart`] handshake plus
    /// account credentials.
    pub fn from_start(
        start: TwilioStart,
        auth_token: Option<String>,
        params: TwilioInputParams,
    ) -> Result<Self> {
        Self::new(
            start.stream_sid,
            start.call_sid,
            start.account_sid,
            auth_token,
            None,
            None,
            None,
            params,
        )
    }

    /// Terminate the Twilio call via the REST API (POST `Status=completed`).
    ///
    /// Takes owned parameters rather than `&self` so the returned future does
    /// not borrow the serializer across the `.await` — the serializer holds a
    /// resampler that is not `Sync`, which would make a `&self`-borrowing future
    /// non-`Send`.
    #[cfg(feature = "serializer-twilio")]
    async fn hang_up_call(
        account_sid: String,
        auth_token: String,
        call_sid: String,
        base_url: Option<String>,
        region: Option<String>,
        edge: Option<String>,
    ) {
        let endpoint = build_call_resource_url(
            &account_sid,
            &call_sid,
            base_url.as_deref(),
            region.as_deref(),
            edge.as_deref(),
        );

        let client = reqwest::Client::new();
        let result = client
            .post(&endpoint)
            .basic_auth(account_sid, Some(auth_token))
            .form(&[("Status", "completed")])
            .send()
            .await;

        match result {
            Ok(resp) if resp.status().is_success() => {
                log::info!("TwilioFrameSerializer: terminated Twilio call {call_sid}");
            }
            Ok(resp) if resp.status().as_u16() == 404 => {
                // Error code 20404 ("resource not found") = call already ended.
                let already_ended = resp
                    .json::<serde_json::Value>()
                    .await
                    .ok()
                    .and_then(|b| b.get("code").and_then(|c| c.as_u64()))
                    == Some(20404);
                if already_ended {
                    log::debug!("TwilioFrameSerializer: call {call_sid} was already terminated");
                } else {
                    log::error!("TwilioFrameSerializer: failed to terminate call {call_sid} (404)");
                }
            }
            Ok(resp) => {
                log::error!(
                    "TwilioFrameSerializer: failed to terminate call {call_sid}: status {}",
                    resp.status()
                );
            }
            Err(e) => {
                log::error!("TwilioFrameSerializer: failed to hang up call: {e}");
            }
        }
    }

    /// Stub when the reqwest-backed hang-up feature is disabled.
    #[cfg(not(feature = "serializer-twilio"))]
    async fn hang_up_call(
        _account_sid: String,
        _auth_token: String,
        _call_sid: String,
        _base_url: Option<String>,
        _region: Option<String>,
        _edge: Option<String>,
    ) {
        log::debug!(
            "TwilioFrameSerializer: auto hang-up requested but the `serializer-twilio` \
             feature is disabled; skipping REST call"
        );
    }

    /// µ-law-encode an outbound audio chunk at `from_rate`, resampling to the
    /// Twilio wire rate first (lazily building/rebuilding the output resampler).
    fn encode_output_audio(&mut self, pcm: &[u8], from_rate: u32) -> Vec<u8> {
        if from_rate == self.twilio_sample_rate {
            return pcm_to_ulaw(pcm, None);
        }

        let needs_rebuild = self.output_resampler.as_ref().map(|(r, _)| *r) != Some(from_rate);
        if needs_rebuild {
            self.output_resampler = Some((
                from_rate,
                StreamResampler::new(from_rate, self.twilio_sample_rate, RESAMPLER_QUALITY),
            ));
        }

        let resampler = self.output_resampler.as_mut().map(|(_, r)| r);
        pcm_to_ulaw(pcm, resampler)
    }
}

#[async_trait]
impl FrameSerializer for TwilioFrameSerializer {
    async fn setup(&mut self, audio_in_sample_rate: u32, _audio_out_sample_rate: u32) {
        self.sample_rate = self.params.sample_rate.unwrap_or(audio_in_sample_rate);

        self.input_resampler = if self.twilio_sample_rate == self.sample_rate {
            None
        } else {
            Some(StreamResampler::new(
                self.twilio_sample_rate,
                self.sample_rate,
                RESAMPLER_QUALITY,
            ))
        };
    }

    async fn serialize(&mut self, frame: &Frame) -> Option<SerializedOutput> {
        // 1. End / Cancel → auto hang-up (once).
        let is_end_or_cancel = matches!(
            &frame.inner,
            FrameInner::Control(ControlFrame::End { .. }) | FrameInner::System(SystemFrame::Cancel { .. })
        );
        if self.params.auto_hang_up && !self.hangup_attempted && is_end_or_cancel {
            self.hangup_attempted = true;
            if let (Some(account_sid), Some(auth_token), Some(call_sid)) =
                (self.account_sid.clone(), self.auth_token.clone(), self.call_sid.clone())
            {
                Self::hang_up_call(
                    account_sid,
                    auth_token,
                    call_sid,
                    self.base_url.clone(),
                    self.region.clone(),
                    self.edge.clone(),
                )
                .await;
            }
            return None;
        }

        match &frame.inner {
            // 2. Interruption → barge-in clear.
            FrameInner::System(SystemFrame::Interruption) => {
                Some(SerializedOutput::Text(
                    json!({ "event": "clear", "streamSid": self.stream_sid }).to_string(),
                ))
            }

            // 3. Output audio → 8 kHz µ-law media event.
            FrameInner::Data(DataFrame::OutputAudioRaw(audio)) => {
                let ulaw = self.encode_output_audio(&audio.audio, audio.sample_rate);
                if ulaw.is_empty() {
                    return None;
                }
                let payload = base64::engine::general_purpose::STANDARD.encode(&ulaw);
                Some(SerializedOutput::Text(
                    json!({
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": { "payload": payload },
                    })
                    .to_string(),
                ))
            }

            // Everything else is not carried over Twilio's protocol.
            _ => None,
        }
    }

    async fn deserialize(&mut self, data: &SerializedInput) -> Option<Frame> {
        let text = match data {
            SerializedInput::Text(t) => t,
            SerializedInput::Binary(_) => return None, // Twilio speaks JSON text
        };

        let msg: serde_json::Value = serde_json::from_str(text).ok()?;
        match msg.get("event").and_then(|v| v.as_str())? {
            "media" => {
                let payload = msg.get("media")?.get("payload")?.as_str()?;
                let ulaw = base64::engine::general_purpose::STANDARD.decode(payload).ok()?;
                let pcm = ulaw_to_pcm(&ulaw, self.input_resampler.as_mut());
                if pcm.is_empty() {
                    return None;
                }
                Some(Frame::input_audio(pcm, self.sample_rate, 1))
            }
            "dtmf" => {
                let digit = msg.get("dtmf")?.get("digit")?.as_str()?;
                KeypadEntry::from_digit(digit).map(Frame::input_dtmf)
            }
            // start / stop / mark / connected are not turned into frames.
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn serializer() -> TwilioFrameSerializer {
        TwilioFrameSerializer::new(
            "MZ_test_stream",
            None,
            None,
            None,
            None,
            None,
            None,
            TwilioInputParams { auto_hang_up: false, ..Default::default() },
        )
        .unwrap()
    }

    #[test]
    fn auto_hang_up_requires_credentials() {
        let err = TwilioFrameSerializer::new(
            "MZ",
            None,
            None,
            None,
            None,
            None,
            None,
            TwilioInputParams::default(), // auto_hang_up = true
        );
        assert!(err.is_err());
    }

    #[test]
    fn single_region_or_edge_is_rejected() {
        let err = TwilioFrameSerializer::new(
            "MZ",
            Some("CA1".into()),
            Some("AC1".into()),
            Some("tok".into()),
            Some("au1".into()),
            None, // edge missing
            None,
            TwilioInputParams::default(),
        );
        assert!(err.is_err());
    }

    #[test]
    fn url_builder_variants() {
        assert_eq!(
            build_call_resource_url("AC1", "CA1", None, None, None),
            "https://api.twilio.com/2010-04-01/Accounts/AC1/Calls/CA1.json"
        );
        assert_eq!(
            build_call_resource_url("AC1", "CA1", None, Some("au1"), Some("sydney")),
            "https://api.sydney.au1.twilio.com/2010-04-01/Accounts/AC1/Calls/CA1.json"
        );
        assert_eq!(
            build_call_resource_url("AC1", "CA1", Some("https://example.com/"), None, None),
            "https://example.com/2010-04-01/Accounts/AC1/Calls/CA1.json"
        );
    }

    #[tokio::test]
    async fn serialize_interruption_is_clear_event() {
        let mut s = serializer();
        s.setup(8000, 8000).await;
        let out = s.serialize(&Frame::interruption()).await.unwrap();
        assert_eq!(
            out,
            SerializedOutput::Text(r#"{"event":"clear","streamSid":"MZ_test_stream"}"#.into())
        );
    }

    #[tokio::test]
    async fn serialize_output_audio_is_media_event() {
        let mut s = serializer();
        s.setup(8000, 8000).await; // 8k pipeline → no resampling
        // Two 16-bit PCM samples at 8 kHz.
        let pcm: Vec<u8> = [0i16, 1000].iter().flat_map(|x| x.to_le_bytes()).collect();
        let out = s.serialize(&Frame::output_audio(pcm, 8000, 1)).await.unwrap();
        let SerializedOutput::Text(json) = out else { panic!("expected text") };
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["event"], "media");
        assert_eq!(v["streamSid"], "MZ_test_stream");
        let payload = v["media"]["payload"].as_str().unwrap();
        let ulaw = base64::engine::general_purpose::STANDARD.decode(payload).unwrap();
        // µ-law of [0, 1000] with the standard encoder.
        assert_eq!(ulaw, vec![0xFF, 0xCE]);
    }

    #[tokio::test]
    async fn deserialize_media_yields_input_audio() {
        let mut s = serializer();
        s.setup(8000, 8000).await;
        let ulaw = vec![0xFFu8, 0xCE];
        let payload = base64::engine::general_purpose::STANDARD.encode(&ulaw);
        let msg = json!({ "event": "media", "media": { "payload": payload } }).to_string();
        let frame = s.deserialize(&SerializedInput::Text(msg)).await.unwrap();
        assert_eq!(frame.name(), "InputAudioRawFrame");
    }

    #[tokio::test]
    async fn deserialize_dtmf_and_unknown() {
        let mut s = serializer();
        s.setup(8000, 8000).await;

        let dtmf = json!({ "event": "dtmf", "dtmf": { "digit": "5" } }).to_string();
        let frame = s.deserialize(&SerializedInput::Text(dtmf)).await.unwrap();
        assert_eq!(frame.name(), "InputDTMFFrame");

        let bad = json!({ "event": "dtmf", "dtmf": { "digit": "X" } }).to_string();
        assert!(s.deserialize(&SerializedInput::Text(bad)).await.is_none());

        let start = json!({ "event": "start", "start": {} }).to_string();
        assert!(s.deserialize(&SerializedInput::Text(start)).await.is_none());
    }

    #[test]
    fn twilio_start_parses_sids() {
        let msg = json!({
            "event": "start",
            "streamSid": "MZ123",
            "start": { "streamSid": "MZ123", "callSid": "CA1", "accountSid": "AC1" }
        })
        .to_string();
        let start = TwilioStart::parse(&msg).unwrap();
        assert_eq!(start.stream_sid, "MZ123");
        assert_eq!(start.call_sid.as_deref(), Some("CA1"));
        assert_eq!(start.account_sid.as_deref(), Some("AC1"));

        let connected = json!({ "event": "connected" }).to_string();
        assert!(TwilioStart::parse(&connected).is_none());
    }
}
