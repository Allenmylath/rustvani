//! Sarvam Bulbul over REST: `POST https://api.sarvam.ai/text-to-speech`, one utterance a call.
//!
//! The [`TtsSynthesizer`] for Sarvam. Unlike the streaming handler it takes a `pace` per call,
//! returns the whole audio (so its length is known), and can ask for 48 kHz, which goes straight
//! into Opus. Models: `bulbul:v3` (default; pace 0.5–2.0), `bulbul:v2` (pace 0.3–3.0, pitch and
//! loudness). Languages: the eleven Bulbul speaks (`bn en gu hi kn ml mr od pa ta te`, as `xx-IN`).

use std::time::Duration;

use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use bytes::Bytes;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};

use super::sarvam::{get_model_config, TtsModelConfig, MODEL_BULBUL_V3};
use super::synth::{Pcm, TtsSynthesizer};
use crate::error::{PipecatError, Result};
use crate::utils::sentence_splitter::extract_sentences;

/// Languages Bulbul speaks: primary subtag → Sarvam code.
pub const TTS_LANGUAGES: &[(&str, &str)] = &[
    ("bn", "bn-IN"),
    ("en", "en-IN"),
    ("gu", "gu-IN"),
    ("hi", "hi-IN"),
    ("kn", "kn-IN"),
    ("ml", "ml-IN"),
    ("mr", "mr-IN"),
    ("od", "od-IN"),
    ("or", "od-IN"), // ISO 639-1 for Odia
    ("pa", "pa-IN"),
    ("ta", "ta-IN"),
    ("te", "te-IN"),
];

/// The most text one request takes (bulbul:v3: 2500; v2: 1500). Longer texts are split at
/// sentence ends and the pieces joined.
const MAX_CHARS: usize = 1500;

#[derive(Debug, Clone)]
pub struct SarvamRestTtsConfig {
    /// Sent as `api-subscription-key`.
    pub api_key: String,
    /// `bulbul:v3` (default) or `bulbul:v2`.
    pub model: String,
    pub base_url: String,
    /// `bulbul:v3` only: 0.01–1.0; lower is steadier.
    pub temperature: Option<f32>,
    /// `bulbul:v2` only.
    pub pitch: Option<f32>,
    pub loudness: Option<f32>,
    pub timeout: Duration,
}

impl Default for SarvamRestTtsConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: "bulbul:v3".to_string(),
            base_url: "https://api.sarvam.ai".to_string(),
            temperature: None,
            pitch: None,
            loudness: None,
            timeout: Duration::from_secs(20),
        }
    }
}

pub struct SarvamRestTts {
    config: SarvamRestTtsConfig,
    model: &'static TtsModelConfig,
    client: Client,
}

#[derive(Deserialize)]
struct TtsResponse {
    #[serde(default)]
    audios: Vec<String>,
}

impl SarvamRestTts {
    pub fn new(config: SarvamRestTtsConfig) -> Self {
        let model = get_model_config(&config.model).unwrap_or(&MODEL_BULBUL_V3);
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .unwrap_or_else(|_| Client::new());
        Self { config, model, client }
    }

    pub fn config(&self) -> &SarvamRestTtsConfig {
        &self.config
    }

    /// Sarvam's code for a BCP-47 tag (`hi` or `hi-IN` → `hi-IN`), if Bulbul speaks it.
    pub fn code_for(tag: &str) -> Option<&'static str> {
        let primary = tag.trim().split(['-', '_']).next().unwrap_or("").to_ascii_lowercase();
        TTS_LANGUAGES.iter().find(|(p, _)| *p == primary).map(|(_, code)| *code)
    }

    /// The body of one call.
    fn request(&self, text: &str, code: &str, voice: &str, pace: f32, sample_rate: u32) -> Value {
        let mut body = json!({
            "text": text,
            "target_language_code": code,
            "speaker": if voice.is_empty() { self.model.default_speaker } else { voice },
            "model": self.config.model,
            "pace": pace,
            "speech_sample_rate": sample_rate,
            "output_audio_codec": "wav",
        });
        if self.model.supports_temperature {
            if let Some(t) = self.config.temperature {
                body["temperature"] = json!(t);
            }
        }
        if self.model.supports_pitch {
            if let Some(p) = self.config.pitch {
                body["pitch"] = json!(p);
            }
        }
        if self.model.supports_loudness {
            if let Some(l) = self.config.loudness {
                body["loudness"] = json!(l);
            }
        }
        body
    }

    /// Tries a call this many times on a rate limit or a server error, waiting [`RETRY_WAITS`].
    const ATTEMPTS: u32 = 3;

    async fn call(&self, body: &Value) -> Result<Vec<u8>> {
        let url = format!("{}/text-to-speech", self.config.base_url.trim_end_matches('/'));
        let mut attempt = 0;
        loop {
            attempt += 1;
            let sent = self
                .client
                .post(&url)
                .header("api-subscription-key", &self.config.api_key)
                .json(body)
                .send()
                .await;
            let retry = |why: String| -> Result<()> {
                if attempt < Self::ATTEMPTS {
                    log::debug!("SarvamRestTts: {why}; retrying");
                    Ok(())
                } else {
                    Err(PipecatError::pipeline(format!("SarvamRestTts: {why}")))
                }
            };
            match sent {
                Err(e) => retry(format!("request failed: {e}"))?,
                Ok(response) => {
                    let status = response.status();
                    if status.is_success() {
                        let parsed: TtsResponse = response
                            .json()
                            .await
                            .map_err(|e| PipecatError::pipeline(format!("SarvamRestTts: bad response: {e}")))?;
                        let audio = parsed
                            .audios
                            .first()
                            .ok_or_else(|| PipecatError::pipeline("SarvamRestTts: no audio in the response"))?;
                        return BASE64
                            .decode(audio)
                            .map_err(|e| PipecatError::pipeline(format!("SarvamRestTts: base64: {e}")));
                    }
                    let text = response.text().await.unwrap_or_default();
                    if status.as_u16() == 429 || status.is_server_error() {
                        retry(format!("HTTP {status} — {text}"))?;
                    } else {
                        return Err(PipecatError::pipeline(format!("SarvamRestTts: HTTP {status} — {text}")));
                    }
                }
            }
            tokio::time::sleep(RETRY_WAITS[(attempt as usize - 1).min(RETRY_WAITS.len() - 1)]).await;
        }
    }
}

/// Before the second and third attempts: a rate limit (429) with several languages dubbing at
/// once clears in about a second.
const RETRY_WAITS: [Duration; 2] = [Duration::from_millis(600), Duration::from_millis(1500)];

/// The PCM in a WAV file, with its format, from the `fmt ` and `data` chunks. Without a RIFF
/// header the bytes are taken as raw PCM at `assume_rate`.
pub fn parse_wav(bytes: &[u8], assume_rate: u32) -> Result<Pcm> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Ok(Pcm { samples: Bytes::copy_from_slice(bytes), rate: assume_rate, channels: 1 });
    }
    let mut pos = 12;
    let mut rate = assume_rate;
    let mut channels = 1u16;
    let mut bits = 16u16;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]]) as usize;
        let body = pos + 8;
        match id {
            b"fmt " if body + 16 <= bytes.len() => {
                channels = u16::from_le_bytes([bytes[body + 2], bytes[body + 3]]);
                rate = u32::from_le_bytes([bytes[body + 4], bytes[body + 5], bytes[body + 6], bytes[body + 7]]);
                bits = u16::from_le_bytes([bytes[body + 14], bytes[body + 15]]);
            }
            b"data" => {
                if bits != 16 {
                    return Err(PipecatError::pipeline(format!("SarvamRestTts: {bits}-bit WAV, expected 16")));
                }
                let end = (body + size).min(bytes.len());
                return Ok(Pcm { samples: Bytes::copy_from_slice(&bytes[body..end]), rate, channels: channels.max(1) });
            }
            _ => {}
        }
        pos = body + size + (size & 1);
    }
    Err(PipecatError::pipeline("SarvamRestTts: WAV without a data chunk"))
}

#[async_trait]
impl TtsSynthesizer for SarvamRestTts {
    fn name(&self) -> &'static str {
        "sarvam"
    }

    fn model(&self) -> String {
        self.config.model.clone()
    }

    fn supports(&self, lang: &str) -> bool {
        Self::code_for(lang).is_some()
    }

    fn pace_range(&self) -> (f32, f32) {
        (self.model.pace_min, self.model.pace_max)
    }

    async fn synthesize(&self, text: &str, lang: &str, voice: &str, pace: f32, sample_rate: u32) -> Result<Pcm> {
        let code = Self::code_for(lang)
            .ok_or_else(|| PipecatError::pipeline(format!("SarvamRestTts: Bulbul does not speak {lang:?}")))?;
        let pace = pace.clamp(self.model.pace_min, self.model.pace_max);
        let text = text.trim();
        if text.is_empty() {
            return Ok(Pcm { samples: Bytes::new(), rate: sample_rate, channels: 1 });
        }
        // Pieces of at most MAX_CHARS, at sentence ends.
        let mut pieces: Vec<String> = Vec::new();
        if text.chars().count() <= MAX_CHARS {
            pieces.push(text.to_string());
        } else {
            let mut buffer = text.to_string();
            pieces.extend(extract_sentences(&mut buffer, MAX_CHARS));
            if !buffer.trim().is_empty() {
                pieces.push(buffer.trim().to_string());
            }
        }
        let mut out: Option<Pcm> = None;
        for piece in pieces {
            let body = self.request(&piece, code, voice, pace, sample_rate);
            let wav = self.call(&body).await?;
            let pcm = parse_wav(&wav, sample_rate)?;
            match &mut out {
                None => out = Some(pcm),
                Some(acc) => {
                    if acc.rate != pcm.rate || acc.channels != pcm.channels {
                        return Err(PipecatError::pipeline("SarvamRestTts: pieces differ in format"));
                    }
                    let mut joined = Vec::with_capacity(acc.samples.len() + pcm.samples.len());
                    joined.extend_from_slice(&acc.samples);
                    joined.extend_from_slice(&pcm.samples);
                    acc.samples = Bytes::from(joined);
                }
            }
        }
        Ok(out.unwrap_or(Pcm { samples: Bytes::new(), rate: sample_rate, channels: 1 }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(rate: u32, channels: u16, pcm: &[u8]) -> Vec<u8> {
        let mut w = Vec::new();
        w.extend_from_slice(b"RIFF");
        w.extend_from_slice(&((36 + pcm.len()) as u32).to_le_bytes());
        w.extend_from_slice(b"WAVEfmt ");
        w.extend_from_slice(&16u32.to_le_bytes());
        w.extend_from_slice(&1u16.to_le_bytes());
        w.extend_from_slice(&channels.to_le_bytes());
        w.extend_from_slice(&rate.to_le_bytes());
        w.extend_from_slice(&(rate * u32::from(channels) * 2).to_le_bytes());
        w.extend_from_slice(&(channels * 2).to_le_bytes());
        w.extend_from_slice(&16u16.to_le_bytes());
        w.extend_from_slice(b"data");
        w.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
        w.extend_from_slice(pcm);
        w
    }

    #[test]
    fn wav_is_parsed_and_raw_pcm_is_taken_as_is() {
        let pcm = [1u8, 0, 2, 0, 3, 0, 4, 0];
        let parsed = parse_wav(&wav(48_000, 1, &pcm), 24_000).unwrap();
        assert_eq!((parsed.rate, parsed.channels, parsed.frames()), (48_000, 1, 4));
        assert_eq!(&parsed.samples[..], &pcm);
        assert!((parsed.duration_secs() - 4.0 / 48_000.0).abs() < 1e-9);
        let raw = parse_wav(&pcm, 24_000).unwrap();
        assert_eq!((raw.rate, raw.frames()), (24_000, 4));
        assert!(parse_wav(b"RIFF\0\0\0\0WAVE", 1).is_err(), "no data chunk");
    }

    #[test]
    fn languages_and_request_shape() {
        assert_eq!(SarvamRestTts::code_for("hi"), Some("hi-IN"));
        assert_eq!(SarvamRestTts::code_for("ML-in"), Some("ml-IN"));
        assert_eq!(SarvamRestTts::code_for("or"), Some("od-IN"));
        assert_eq!(SarvamRestTts::code_for("ur"), None, "Bulbul has no Urdu");
        let tts = SarvamRestTts::new(SarvamRestTtsConfig { api_key: "k".into(), temperature: Some(0.4), ..Default::default() });
        assert!(tts.supports("ta") && !tts.supports("de"));
        assert_eq!(tts.pace_range(), (0.5, 2.0));
        let body = tts.request("नमस्ते", "hi-IN", "", 1.2, 48_000);
        assert_eq!(body["speaker"], "shubh", "the model's default voice");
        assert_eq!(body["model"], "bulbul:v3");
        assert_eq!(body["target_language_code"], "hi-IN");
        assert_eq!(body["speech_sample_rate"], 48_000);
        assert!((body["temperature"].as_f64().unwrap() - 0.4).abs() < 1e-6);
        assert!(body.get("pitch").is_none(), "v3 has no pitch");
        let v2 = SarvamRestTts::new(SarvamRestTtsConfig { model: "bulbul:v2".into(), pitch: Some(0.2), ..Default::default() });
        let body = v2.request("x", "hi-IN", "anushka", 1.0, 22_050);
        assert!((body["pitch"].as_f64().unwrap() - 0.2).abs() < 1e-6);
        assert_eq!(v2.pace_range(), (0.3, 3.0));
    }

    /// Against the real API, in every language Bulbul speaks:
    /// `SARVAM_API_KEY=… cargo test --features tts-sarvam sarvam_rest_tts_live -- --ignored --nocapture`.
    #[tokio::test]
    #[ignore]
    async fn sarvam_rest_tts_live() {
        let Ok(api_key) = std::env::var("SARVAM_API_KEY") else {
            eprintln!("SARVAM_API_KEY not set");
            return;
        };
        let tts = SarvamRestTts::new(SarvamRestTtsConfig { api_key, ..Default::default() });
        let samples = [
            ("hi", "नमस्ते, आज मौसम बहुत अच्छा है।"),
            ("ml", "നാളെ മീറ്റിംഗ് ഉണ്ടോ?"),
            ("ta", "இன்று காலை நல்ல வானிலை."),
            ("te", "ఈ రోజు వాతావరణం బాగుంది."),
            ("kn", "ಇಂದು ಹವಾಮಾನ ಚೆನ್ನಾಗಿದೆ."),
            ("bn", "আজ আবহাওয়া খুব ভালো।"),
            ("mr", "आज हवामान खूप छान आहे."),
            ("gu", "આજે હવામાન ખૂબ સરસ છે."),
            ("pa", "ਅੱਜ ਮੌਸਮ ਬਹੁਤ ਵਧੀਆ ਹੈ।"),
            ("od", "ଆଜି ପାଗ ବହୁତ ଭଲ ଅଛି।"),
            ("en", "Good morning everyone, the drone survey starts now."),
        ];
        for (lang, text) in samples {
            let t0 = std::time::Instant::now();
            let pcm = tts.synthesize(text, lang, "", 1.0, 48_000).await.unwrap_or_else(|e| panic!("{lang}: {e}"));
            println!("{lang}: {:.2} s of audio at {} Hz in {:.2} s", pcm.duration_secs(), pcm.rate, t0.elapsed().as_secs_f64());
            assert_eq!(pcm.rate, 48_000, "{lang}");
            assert!(pcm.duration_secs() > 0.5, "{lang}: too short");
        }
        // Pace shortens the audio.
        let slow = tts.synthesize("यह एक परीक्षण वाक्य है।", "hi", "", 0.8, 48_000).await.unwrap();
        let fast = tts.synthesize("यह एक परीक्षण वाक्य है।", "hi", "", 1.5, 48_000).await.unwrap();
        println!("pace 0.8: {:.2} s, pace 1.5: {:.2} s", slow.duration_secs(), fast.duration_secs());
        assert!(fast.duration_secs() < slow.duration_secs() * 0.8);
    }
}
