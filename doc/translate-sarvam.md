# Translation — `TranslationService` and Sarvam Translate

**Files:** `src/services/translate/mod.rs` (service, provider trait), `src/services/translate/sarvam.rs` (Sarvam backend)
**Feature:** `translate-sarvam` (enabled by default) gates the Sarvam backend only; the service and the `TranslationProvider` trait are always compiled
**Protocol:** HTTP (`POST https://api.sarvam.ai/translate`)

Translates every **final** transcript into one or more target languages, as a pipeline stage after
the STT. Each translation goes downstream as a `TranslationFrame` (`TranslationData`), which the
`RaviObserver` sends to the client as a `user-translation` message.

## Pipeline position

```
… → STT → TranslationService → (LLMUserAggregator → …) or transport output
          TranscriptionFrame (final) ─▶ provider ─▶ TranslationFrame per target
```

Every frame passes through unchanged. Final transcripts (a `TranscriptionFrame` with `finalized`,
or the turn-closing transcript the STT turn gate bundles onto `VADUserStoppedSpeaking`) are also
queued for translation. Interim transcripts are never translated.

Calls run off the pipeline's tasks, one ordered queue per target language: a slow provider never
holds up the transcript or another language, and a language's translations come out in the order
their transcripts did. A transcript whose language already is the target is passed through as its
own translation.

## Usage

```rust
use rustvani::{SarvamTranslateConfig, SarvamTranslateProvider, TranslationService, TranslationTargets};

let targets = TranslationTargets::new(["en", "hi"]);
let translate = TranslationService::new(
    SarvamTranslateProvider::new(SarvamTranslateConfig {
        api_key: std::env::var("SARVAM_API_KEY").unwrap(),
        ..Default::default()
    }),
    targets.clone(),
)
.with_billing(billing.clone())
.into_processor();

let task = PipelineTask::new(vec![transport.input(), stt, translate, transport.output()], params);

// Later, from anywhere: translate only what is being watched.
targets.set(["hi"]);
```

Language tags are BCP-47 however you name them (`hi`, `en-IN`, `ml_IN`); the provider maps them to
Sarvam's `xx-IN` codes. Ask `provider.supports("ta")` before offering a language.

`.with_default_source_language("ml-IN")` names the source when a transcript carries no language
(an STT configured for one language reports none per phrase); without it the provider detects it.

## `SarvamTranslateConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `api_key` | `String` | — | Sent as `api-subscription-key` |
| `model` | `String` | `"sarvam-translate:v1"` | Or `"mayura:v1"` |
| `base_url` | `String` | `"https://api.sarvam.ai"` | |
| `mode` | `Option<String>` | `None` (Sarvam's `formal`) | `formal`, `modern-colloquial`, `classic-colloquial`, `code-mixed` (`mayura:v1` only for the last three) |
| `numerals_format` | `Option<String>` | `None` (`international`) | Or `native` |
| `speaker_gender` | `Option<String>` | `None` | `Male` / `Female` |
| `output_script` | `Option<String>` | `None` | `mayura:v1` only: `roman`, `fully-native`, `spoken-form-in-native` |
| `timeout` | `Duration` | 10 s | Per call |
| `auto_detect_model` | `Option<String>` | `Some("mayura:v1")` | Used, with `source_language_code: "auto"`, when a transcript's language is unknown: Sarvam detects the source only on `mayura:v1`. `None` fails such calls. |

### Models and languages

| Model | Languages | Max input | Detects source |
|---|---|---|---|
| `sarvam-translate:v1` | English + 22 Indian languages (`bn gu hi kn ml mr od pa ta te as brx doi kok ks mai mni ne sa sat sd ur`) | 2000 chars | no |
| `mayura:v1` | `bn en gu hi kn ml mr od pa ta te` | 1000 chars | yes (`auto`) |

Longer transcripts are split at sentence ends (then spaces) and the pieces joined. `429` and `5xx`
are retried once after 500 ms.

## Frames

| Consumed | Produced |
|---|---|
| `TranscriptionFrame` (`finalized`) | `TranslationFrame` per target language |
| `VADUserStoppedSpeaking` with a bundled transcript | `TranslationFrame` per target language |
| `EndFrame`, `CancelFrame`, `StopFrame` | (stops the queues) |

`TranslationData` carries `text`, `target_language` (the tag as you configured it), and the source
transcript's `source_text`, `source_language` (the STT's, else what the provider detected),
`user_id` and `timestamp`, so a client can pair each translation with its transcript.

## RAVI

```json
{"label":"ravi","type":"user-translation","data":{
  "text":"Is there a meeting tomorrow?","source_text":"നാളെ മീറ്റിംഗ് ഉണ്ടോ?",
  "source_language":"ml-IN","target_language":"en","user_id":"","timestamp":"1761100000.250","final":true}}
```

`RaviObserverParams::user_translation_enabled` (default `true`) switches it off.

## Billing

With `.with_billing(..)`, each call records `BillingEvent::TranslationUsage { provider, model,
source_language, target_language, char_count, .. }` where `char_count` is the source text's
characters (what translation APIs bill by). `SessionSummary` sums them as `translation_chars` /
`translation_calls`; the Postgres schema gains the same columns and `billing_events` rows of type
`translation` with `source_language` / `target_language`.

## Writing a provider

```rust
#[async_trait]
impl TranslationProvider for MyBackend {
    fn name(&self) -> &'static str { "mine" }
    fn model(&self) -> String { "m1".into() }
    fn supports(&self, lang: &str) -> bool { … }
    fn max_chars(&self) -> usize { 5000 }
    async fn translate(&self, text: &str, from: Option<&str>, to: &str) -> Result<Translated> { … }
}
```

`from` is `None` when the STT did not report a language; return the detected one in
`Translated::source_language` when you know it.

## Test

```sh
cargo test translate                       # the service with a stub provider, and the Sarvam request shapes
SARVAM_API_KEY=… cargo test sarvam_translate_live -- --ignored --nocapture   # the real API
```
