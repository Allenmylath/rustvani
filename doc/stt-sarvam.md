# Sarvam STT

**File:** `src/services/stt/sarvam.rs`
**Feature:** `stt-sarvam` (enabled by default)
**Protocol:** WebSocket (`wss://api.sarvam.ai/speech-to-text/ws`)

Sarvam AI streaming STT with Indian language support and multiple output modes.

Sarvam is built on the shared STT core — see [stt-abstraction.md](stt-abstraction.md).
`sarvam.rs` contributes only the wire protocol (`SarvamProvider: SttProvider`)
and this config struct; the turn gate, audio front-end, billing and WebSocket
plumbing come from `services/stt/core`.

## Pipeline Position

```
transport.input() → SarvamSttHandler → LLMUserAggregator → llm → tts → transport.output()
```

`LLMUserAggregator` must sit immediately downstream: it consumes the
`VADUserStoppedSpeaking` that the turn gate releases with the transcript
bundled onto it.

## Usage

```rust
use rustvani::services::stt::sarvam::{SarvamSttConfig, SarvamSttHandler};

let stt = SarvamSttHandler::new(SarvamSttConfig {
    api_key: std::env::var("SARVAM_API_KEY").unwrap(),
    model: "saaras:v3".to_string(),
    language: Some("unknown".to_string()),
    mode: Some("transcribe".to_string()),
    ..Default::default()
}).into_processor();
```

## Configuration

### Protocol

| Field | Type | Default | Description |
|---|---|---|---|
| `api_key` | `String` | — | Sarvam API subscription key (`api-subscription-key` header) |
| `model` | `String` | `"saaras:v3"` | `saaras:v3`, `saarika:v2.5`, `saaras:v2.5` |
| `language` | `Option<String>` | `Some("unknown")` | BCP-47 code; `unknown` = auto-detect. Sent as `language-code` (hyphen) |
| `mode` | `Option<String>` | `Some("transcribe")` | `transcribe`, `translate`, `verbatim`, `translit`, `codemix` (v3 only) |
| `prompt` | `Option<String>` | `None` | Biasing string — domain terms, names |
| `sample_rate` | `u32` | `16000` | Audio at any other rate is resampled to this |
| `encoding` | `String` | `"wav"` | `wav`, `pcm_s16le`, `pcm_l16`, `pcm_raw` |

### Server-side VAD

| Field | Type | Default | Description |
|---|---|---|---|
| `high_vad_sensitivity` | `bool` | `false` | Shorter silence before flush |
| `vad_signals` | `bool` | `false` | Receive server `START_SPEECH` / `END_SPEECH` events (advisory only) |
| `vad_tuning` | `SarvamVadTuning` | all `None` | Fine-grained tuning, **saaras:v3 only** |

`SarvamVadTuning` fields, all `Option` and emitted only when `Some`:
`positive_speech_threshold`, `negative_speech_threshold`, `min_speech_frames`,
`first_turn_min_speech_frames`, `negative_frames_count`,
`negative_frames_window`, `start_speech_volume_threshold`,
`interrupt_min_speech_frames`, `pre_speech_pad_frames`,
`num_initial_ignored_frames`.

These tune Sarvam's *server* VAD. The local Silero VAD in `transport.input()`
owns turn boundaries; with `audio_gating` on (default) the server never sees
inter-turn audio, so these mostly matter when gating is off.

### Audio front-end and turn gating

Handled by the core — see [stt-abstraction.md](stt-abstraction.md).

| Field | Type | Default | Description |
|---|---|---|---|
| `noise_reduction` | `bool` | `true` | Noise suppression before sending |
| `noise_backend` | `NoiseBackend` | `Rnnoise` | `Rnnoise` or `HushVani` (falls back to RNNoise if init fails) |
| `agc` | `bool` | `true` | High-pass before the denoiser, AGC + soft limiter after |
| `audio_gating` | `bool` | `true` | Send audio only during local-VAD-attested turns |
| `pre_roll_ms` | `u32` | `500` | Pre-speech audio replayed on VAD start, so the first syllable isn't clipped |
| `stop_release_timeout_ms` | `u64` | `1200` | How long to hold a gated VAD stop waiting for the transcript |

Turning `audio_gating` off restores continuous streaming. Spurious server-VAD
transcripts then become possible again, so set the aggregator's
`LateTranscriptPolicy::Discard`.

## Model-Specific Endpoints

- `saaras:v2.5` → `/speech-to-text-translate/ws` (legacy translation to English; takes no `language-code`)
- All other models → `/speech-to-text/ws`

## Frames

**Consumed:**
- `StartFrame` → resets the gate, connects the WebSocket
- `InputAudioRaw` → forwarded, then resample → high-pass → denoise → AGC → gate (send now, or buffer as pre-roll)
- `VADUserStartedSpeaking` → drops any pending stop (barge-in), opens a new epoch, sends pre-roll
- `VADUserStoppedSpeaking` → **consumed**; front-end tail sent, then `{"type":"flush"}`, then a release timer armed
- `EndFrame` / `CancelFrame` → resets the gate, disconnects

**Produced:**
- `VADUserStoppedSpeaking` with the transcript bundled on, released when the transcript arrives (or on timeout)
- `TranscriptionFrame` (downstream) for a transcript arriving mid-turn
- `ErrorFrame` (upstream) on connection / server errors

Sarvam's streaming API returns finals only, so `interim_policy` is fixed to
`Drop`. An empty or whitespace-only transcript is still the answer to our flush
and closes the turn immediately rather than waiting for the release timeout.

## Billing

`BillingEvent::SttUsage { provider: "SarvamStt", .. }` per final transcript,
using the server-reported `metrics.audio_duration` when present and the gate's
ledger estimate otherwise.

## Environment Variables

```bash
SARVAM_API_KEY=your_key
```

## Cargo Feature

Enabled by default. `stt-sarvam` pulls `tokio-tungstenite`. Note that Sarvam's
**LLM** service is a separate feature, `llm-sarvam` (HTTP + SSE, pulls
`reqwest`); both read the same `SARVAM_API_KEY`.

To build only Sarvam STT:

```toml
[dependencies]
rustvani = { version = "0.4", default-features = false, features = ["stt-sarvam"] }
```
