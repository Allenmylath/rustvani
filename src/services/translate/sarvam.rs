//! Sarvam Translate: `POST https://api.sarvam.ai/translate`.
//!
//! Models: `sarvam-translate:v1` (default; English and 22 Indian languages, up to 2000 characters
//! a call, formal register) and `mayura:v1` (11 languages, 1000 characters, colloquial modes,
//! output scripts, and the only one that detects the source language with `auto`).
//!
//! Languages are given to the service as BCP-47 tags however the caller names them (`hi`,
//! `en-IN`, `ml_IN`); this provider maps them to Sarvam's `xx-IN` codes and declines the rest
//! through [`TranslationProvider::supports`].

use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};

use super::{primary_subtag, Translated, TranslationProvider};
use crate::error::{PipecatError, Result};

/// Languages `mayura:v1` serves (and detects): primary subtag → Sarvam code.
const MAYURA_LANGUAGES: &[(&str, &str)] = &[
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

/// The languages `sarvam-translate:v1` adds.
const SARVAM_TRANSLATE_EXTRA: &[(&str, &str)] = &[
    ("as", "as-IN"),
    ("brx", "brx-IN"),
    ("doi", "doi-IN"),
    ("kok", "kok-IN"),
    ("ks", "ks-IN"),
    ("mai", "mai-IN"),
    ("mni", "mni-IN"),
    ("ne", "ne-IN"),
    ("sa", "sa-IN"),
    ("sat", "sat-IN"),
    ("sd", "sd-IN"),
    ("ur", "ur-IN"),
];

pub const SARVAM_TRANSLATE_V1: &str = "sarvam-translate:v1";
pub const MAYURA_V1: &str = "mayura:v1";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SarvamTranslateConfig {
    /// Sent as `api-subscription-key`.
    pub api_key: String,
    /// `sarvam-translate:v1` (default) or `mayura:v1`.
    pub model: String,
    pub base_url: String,
    /// `formal` (default; the only one `sarvam-translate:v1` accepts), `modern-colloquial`,
    /// `classic-colloquial`, `code-mixed`.
    pub mode: Option<String>,
    /// `international` (0-9, default) or `native`.
    pub numerals_format: Option<String>,
    /// `Male` or `Female`, for languages that inflect by speaker.
    pub speaker_gender: Option<String>,
    /// `mayura:v1` only: `roman`, `fully-native`, `spoken-form-in-native`.
    pub output_script: Option<String>,
    /// Per-call HTTP timeout.
    pub timeout: Duration,
    /// Sarvam detects the source language (`source_language_code: "auto"`) only on `mayura:v1`.
    /// When a transcript's language is unknown, the call goes to this model instead; `None`
    /// fails such calls.
    pub auto_detect_model: Option<String>,
}

impl Default for SarvamTranslateConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: SARVAM_TRANSLATE_V1.to_string(),
            base_url: "https://api.sarvam.ai".to_string(),
            mode: None,
            numerals_format: None,
            speaker_gender: None,
            output_script: None,
            timeout: Duration::from_secs(10),
            auto_detect_model: Some(MAYURA_V1.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

pub struct SarvamTranslateProvider {
    config: SarvamTranslateConfig,
    client: Client,
}

#[derive(Deserialize)]
struct TranslateResponse {
    translated_text: String,
    #[serde(default)]
    source_language_code: Option<String>,
}

impl SarvamTranslateProvider {
    pub fn new(config: SarvamTranslateConfig) -> Self {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .unwrap_or_else(|_| Client::new());
        Self { config, client }
    }

    pub fn config(&self) -> &SarvamTranslateConfig {
        &self.config
    }

    /// Sarvam's code for a BCP-47 tag (`hi` or `hi-IN` → `hi-IN`), if `model` serves it.
    pub fn code_for(model: &str, tag: &str) -> Option<&'static str> {
        let primary = primary_subtag(tag.trim()).to_ascii_lowercase();
        let find = |table: &[(&str, &'static str)]| {
            table.iter().find(|(p, _)| *p == primary).map(|(_, code)| *code)
        };
        find(MAYURA_LANGUAGES).or_else(|| {
            if model == MAYURA_V1 {
                None
            } else {
                find(SARVAM_TRANSLATE_EXTRA)
            }
        })
    }

    /// The body of one call, and the model it goes to: the configured one, or
    /// `auto_detect_model` when the source language is unknown.
    fn request(&self, text: &str, from: Option<&str>, to: &str) -> Result<(String, Value)> {
        let (model, source) = match from.and_then(|from| Self::code_for(&self.config.model, from)) {
            Some(source) => (self.config.model.clone(), source.to_string()),
            None => {
                let model = self.config.auto_detect_model.clone().ok_or_else(|| {
                    PipecatError::pipeline(format!(
                        "SarvamTranslate: source language {from:?} is unknown to {} and no auto_detect_model is set",
                        self.config.model
                    ))
                })?;
                (model, "auto".to_string())
            }
        };
        let target = Self::code_for(&model, to).ok_or_else(|| {
            PipecatError::pipeline(format!("SarvamTranslate: {model} does not translate into {to:?}"))
        })?;
        let mut body = json!({
            "input": text,
            "source_language_code": source,
            "target_language_code": target,
            "model": model,
        });
        if let Some(mode) = &self.config.mode {
            body["mode"] = json!(mode);
        }
        if let Some(numerals) = &self.config.numerals_format {
            body["numerals_format"] = json!(numerals);
        }
        if let Some(gender) = &self.config.speaker_gender {
            body["speaker_gender"] = json!(gender);
        }
        if let Some(script) = &self.config.output_script {
            body["output_script"] = json!(script);
        }
        Ok((model, body))
    }

    async fn call(&self, body: &Value) -> Result<TranslateResponse> {
        let url = format!("{}/translate", self.config.base_url.trim_end_matches('/'));
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
                if attempt < 2 {
                    log::debug!("SarvamTranslate: {why}; retrying");
                    Ok(())
                } else {
                    Err(PipecatError::pipeline(format!("SarvamTranslate: {why}")))
                }
            };
            match sent {
                Err(e) => retry(format!("request failed: {e}"))?,
                Ok(response) => {
                    let status = response.status();
                    if status.is_success() {
                        return response
                            .json::<TranslateResponse>()
                            .await
                            .map_err(|e| PipecatError::pipeline(format!("SarvamTranslate: bad response: {e}")));
                    }
                    let text = response.text().await.unwrap_or_default();
                    if status.as_u16() == 429 || status.is_server_error() {
                        retry(format!("HTTP {status} — {text}"))?;
                    } else {
                        return Err(PipecatError::pipeline(format!("SarvamTranslate: HTTP {status} — {text}")));
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }
}

#[async_trait]
impl TranslationProvider for SarvamTranslateProvider {
    fn name(&self) -> &'static str {
        "sarvam"
    }

    fn model(&self) -> String {
        self.config.model.clone()
    }

    fn supports(&self, lang: &str) -> bool {
        Self::code_for(&self.config.model, lang).is_some()
    }

    fn max_chars(&self) -> usize {
        if self.config.model == MAYURA_V1 { 1000 } else { 2000 }
    }

    async fn translate(&self, text: &str, from: Option<&str>, to: &str) -> Result<Translated> {
        let (model, body) = self.request(text, from, to)?;
        log::debug!(
            "SarvamTranslate: {} chars {} → {} ({model})",
            text.chars().count(),
            body["source_language_code"],
            body["target_language_code"]
        );
        let response = self.call(&body).await?;
        let source_language = response
            .source_language_code
            .filter(|code| code != "auto")
            .or_else(|| from.map(str::to_string));
        Ok(Translated {
            text: response.translated_text,
            source_language,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(config: SarvamTranslateConfig) -> SarvamTranslateProvider {
        SarvamTranslateProvider::new(SarvamTranslateConfig { api_key: "k".into(), ..config })
    }

    #[test]
    fn tags_map_to_sarvam_codes_per_model() {
        assert_eq!(SarvamTranslateProvider::code_for(SARVAM_TRANSLATE_V1, "hi"), Some("hi-IN"));
        assert_eq!(SarvamTranslateProvider::code_for(SARVAM_TRANSLATE_V1, "en-IN"), Some("en-IN"));
        assert_eq!(SarvamTranslateProvider::code_for(SARVAM_TRANSLATE_V1, "en-US"), Some("en-IN"));
        assert_eq!(SarvamTranslateProvider::code_for(SARVAM_TRANSLATE_V1, "ML_in"), Some("ml-IN"));
        assert_eq!(SarvamTranslateProvider::code_for(SARVAM_TRANSLATE_V1, "or"), Some("od-IN"));
        assert_eq!(SarvamTranslateProvider::code_for(SARVAM_TRANSLATE_V1, "ur"), Some("ur-IN"));
        assert_eq!(SarvamTranslateProvider::code_for(MAYURA_V1, "ur"), None, "mayura has no Urdu");
        assert_eq!(SarvamTranslateProvider::code_for(SARVAM_TRANSLATE_V1, "de"), None);
        let p = provider(SarvamTranslateConfig::default());
        assert!(p.supports("ta") && !p.supports("fr"));
        assert_eq!(p.max_chars(), 2000);
        assert_eq!(provider(SarvamTranslateConfig { model: MAYURA_V1.into(), ..Default::default() }).max_chars(), 1000);
    }

    #[test]
    fn a_known_source_goes_to_the_configured_model() {
        let p = provider(SarvamTranslateConfig {
            mode: Some("formal".into()),
            numerals_format: Some("native".into()),
            ..Default::default()
        });
        let (model, body) = p.request("നാളെ", Some("ml-IN"), "en").unwrap();
        assert_eq!(model, SARVAM_TRANSLATE_V1);
        assert_eq!(
            body,
            serde_json::json!({
                "input": "നാളെ",
                "source_language_code": "ml-IN",
                "target_language_code": "en-IN",
                "model": "sarvam-translate:v1",
                "mode": "formal",
                "numerals_format": "native",
            })
        );
    }

    #[test]
    fn an_unknown_source_goes_to_the_auto_detect_model_or_fails() {
        let p = provider(SarvamTranslateConfig::default());
        let (model, body) = p.request("hello", None, "hi").unwrap();
        assert_eq!(model, MAYURA_V1);
        assert_eq!(body["source_language_code"], "auto");
        assert_eq!(body["model"], "mayura:v1");
        // A source the model does not know counts as unknown too.
        let (model, _) = p.request("hallo", Some("de"), "hi").unwrap();
        assert_eq!(model, MAYURA_V1);
        // Detection cannot target a language mayura lacks.
        assert!(p.request("hello", None, "ur").is_err());
        let strict = provider(SarvamTranslateConfig { auto_detect_model: None, ..Default::default() });
        assert!(strict.request("hello", None, "hi").is_err());
        assert!(strict.request("hello", Some("hi"), "fr").is_err(), "unsupported target");
    }

    /// Against the real API: `SARVAM_API_KEY=… cargo test -p rustvani sarvam_translate_live -- --ignored`.
    #[tokio::test]
    #[ignore]
    async fn sarvam_translate_live() {
        let Ok(api_key) = std::env::var("SARVAM_API_KEY") else {
            eprintln!("SARVAM_API_KEY not set");
            return;
        };
        let p = SarvamTranslateProvider::new(SarvamTranslateConfig { api_key, ..Default::default() });
        let known = p.translate("Good morning everyone.", Some("en-IN"), "hi").await.unwrap();
        println!("en→hi: {known:?}");
        assert!(!known.text.trim().is_empty());
        assert_eq!(known.source_language.as_deref(), Some("en-IN"));
        let detected = p.translate("നാളെ മീറ്റിംഗ് ഉണ്ടോ?", None, "en").await.unwrap();
        println!("auto→en: {detected:?}");
        assert!(!detected.text.trim().is_empty());
        assert_eq!(detected.source_language.as_deref(), Some("ml-IN"));
    }
}
