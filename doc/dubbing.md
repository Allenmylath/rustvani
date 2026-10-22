# Dubbing — `DubbingService`, `TtsSynthesizer` and Sarvam Bulbul over REST

**Files:** `src/services/dub/mod.rs` (service), `src/services/tts/synth.rs` (one-shot synthesis
trait), `src/services/tts/sarvam_rest.rs` (Sarvam backend)
**Feature:** `tts-sarvam` gates the Sarvam backend (it now also pulls `reqwest`); the service and
the trait are always compiled
**Protocol:** HTTP (`POST https://api.sarvam.ai/text-to-speech`)

Speaks every final translation in its language and hands the audio on as a frame, so a host can
play it where the words were said. Built for live dubbing: the
[`TranslationService`](translate-sarvam.md) translates a speaker's finals, the `DubbingService`
speaks them, one ordered queue per language, and the host decides when each utterance plays.

## Pipeline position

```
… → STT → TranslationService → DubbingService → (a sink the host adds) → transport output
                                 TranslationFrame (target among the dub targets) ─▶ TTS ─▶ DubbedSpeechFrame
```

Every frame passes through unchanged. A `TranslationFrame` with `finalized` whose
`target_language` is in the service's `TranslationTargets` is spoken by the `TtsSynthesizer` at the
pace a `PaceHint` returns, and comes out as one **`DubbedSpeechFrame`** (`DubbedSpeechData`): the
whole utterance's PCM (s16le), its `sample_rate` and `channels`, the `voice` and `pace` used, and
the source transcript's `timestamp` and `source_text` to pair it with its caption. Synthesis runs
off the pipeline's tasks. Partials are never spoken.

## Usage

```rust
use rustvani::{DubConfig, DubbingService, SarvamRestTts, SarvamRestTtsConfig, TranslationTargets};

let targets = TranslationTargets::new(["hi", "ta"]);
let synth = Arc::new(SarvamRestTts::new(SarvamRestTtsConfig {
    api_key: std::env::var("SARVAM_API_KEY").unwrap(),
    ..Default::default()
}));
let dub = DubbingService::new(synth, targets.clone(), DubConfig {
    default_voice: "shubh".into(),
    ..Default::default()
})
.with_pace_hint(Arc::new(|t| if t.text.chars().count() > 80 { 1.3 } else { 1.0 }))
.with_billing(billing.clone())
.into_processor();

let task = PipelineTask::new(vec![transport.input(), stt, translate, dub, my_sink, transport.output()], params);
```

The dub's targets are usually a subset of the translation's; both handles can change at run time.

## `TtsSynthesizer`

```rust
#[async_trait]
pub trait TtsSynthesizer: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn model(&self) -> String;
    fn supports(&self, lang: &str) -> bool;
    fn pace_range(&self) -> (f32, f32) { (1.0, 1.0) }
    async fn synthesize(&self, text: &str, lang: &str, voice: &str, pace: f32, sample_rate: u32) -> Result<Pcm>;
}
```

One utterance in, its whole audio (`Pcm { samples, rate, channels }`, with `duration_secs()`) out,
so a caller knows how long it is before deciding when to play it. The streaming handlers
(`SarvamTtsHandler`, Deepgram, Piper) are the other shape, for a bot's answer as it is generated.

## `SarvamRestTts`

| Field | Type | Default | Description |
|---|---|---|---|
| `api_key` | `String` | — | Sent as `api-subscription-key` |
| `model` | `String` | `"bulbul:v3"` | Or `"bulbul:v2"` (pitch and loudness; pace 0.3–3.0) |
| `base_url` | `String` | `"https://api.sarvam.ai"` | |
| `temperature` | `Option<f32>` | `None` | `bulbul:v3`: 0.01–1.0, lower is steadier |
| `pitch`, `loudness` | `Option<f32>` | `None` | `bulbul:v2` only |
| `timeout` | `Duration` | 20 s | Per call |

- Languages: the eleven Bulbul speaks, as BCP-47 tags however named (`hi`, `hi-IN`, `ml_IN`):
  `bn en gu hi kn ml mr od pa ta te`; `supports()` says which.
- Voices: the model's (`shubh` and 24 more on v3, `anushka` and 6 more on v2); an empty voice is
  the model's default. Names are per model.
- Pace: clamped to the model's range (v3: 0.5–2.0). Sample rates up to 48 kHz; the WAV is decoded
  to PCM (`parse_wav`). Text over 1500 characters is split at sentence ends and the pieces joined.
  One retry on 429/5xx.

Measured on 2026-10-22: 1.5–2.6 s of 48 kHz audio for a short sentence in each language, in
0.6–1.2 s a call; pace 1.5 made a Hindi sentence 43% shorter than pace 0.8.

## `DubConfig`

| Field | Default | Description |
|---|---|---|
| `voices` | empty | Voice per language tag, as the targets name them |
| `default_voice` | empty (the provider's default) | For languages without one |
| `sample_rate` | 48 000 | Asked of the provider |

## Billing

With `.with_billing(..)`, each utterance records `BillingEvent::TtsUsage { provider, voice,
language: Some(lang), char_count, .. }`. `TtsUsage` gained the `language` field for this; it is
`None` from the streaming handlers and left out of the JSON when unset.

## Fitting speech to its time

The service doesn't decide when audio plays; a host does, with the frame's `timestamp` and
`source_text` pairing it to the transcript's span. The `PaceHint` is where the host fits speech to
the time the words took: predict the duration from a running seconds-per-character estimate per
language, ask for `predicted / slot` clamped to a sensible range, measure what came back
(`Pcm::duration_secs()`), and call `synthesize` again faster if it still overruns. The StreamScope
speech engine does exactly this (`sfu/processor-rustvani/src/dub.rs`).

## Test

```sh
cargo test dub                                   # the service with a stub synthesizer; the WAV parser; request shapes
SARVAM_API_KEY=… cargo test --features tts-sarvam sarvam_rest_tts_live -- --ignored --nocapture   # all 11 languages, live
```
