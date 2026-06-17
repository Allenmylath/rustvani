# Sarvam STT

**File:** `src/services/stt/sarvam.rs`  
**Feature:** `stt-sarvam` (enabled by default)  
**Protocol:** WebSocket (`wss://api.sarvam.ai/speech-to-text/ws`)

Sarvam AI streaming STT with Indian language support, multiple output modes, and optional RNNoise denoising.

## Pipeline Position

```
transport.input() → SarvamSttHandler → llm → tts → transport.output()
```

## Usage

```rust
use rustvani::services::stt::sarvam::{SarvamSttConfig, SarvamSttHandler};

let stt = SarvamSttHandler::new(SarvamSttConfig {
    api_key: std::env::var("SARVAM_API_KEY").unwrap(),
    model: "saaras:v3".to_string(),
    language: Some("unknown".to_string()),
    mode: Some("transcribe".to_string()),
    sample_rate: 16_000,
    encoding: "wav".to_string(),
    noise_reduction: true,
    ..Default::default()
}).into_processor();
```

## Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `api_key` | `String` | — | Sarvam API subscription key |
| `model` | `String` | `"saaras:v3"` | `saaras:v3`, `saarika:v2.5`, `saaras:v2.5` |
| `language` | `Option<String>` | `Some("unknown")` | BCP-47 code; `unknown` = auto-detect |
| `mode` | `Option<String>` | `Some("transcribe")` | `transcribe`, `translate`, `verbatim`, `translit`, `codemix` (v3 only) |
| `sample_rate` | `u32` | `16000` | Must match transport audio rate |
| `encoding` | `String` | `"wav"` | `wav`, `pcm_s16le`, `pcm_l16`, `pcm_raw` |
| `high_vad_sensitivity` | `bool` | `false` | Shorter silence before flush |
| `vad_signals` | `bool` | `false` | Receive server VAD start/end events |
| `noise_reduction` | `bool` | `true` | RNNoise before sending |

## Model-Specific Endpoints

- `saaras:v2.5` → `/speech-to-text-translate/ws` (legacy translation to English)
- All other models → `/speech-to-text/ws`

## Frames

**Consumed:**
- `StartFrame` → connects WebSocket
- `InputAudioRaw` → denoise → base64 encode → send
- `VADUserStoppedSpeaking` → flush noise filter → send flush signal
- `EndFrame` / `CancelFrame` → disconnect

**Produced:**
- `TranscriptionFrame` (downstream) on transcript
- `ErrorFrame` (upstream) on errors

## Environment Variables

```bash
SARVAM_API_KEY=your_key
```

## Cargo Feature

Enabled by default. To disable:

```toml
[dependencies]
rustvani = { version = "0.2", default-features = false, features = ["vad-silero-ort", "transport-websocket", "stt-60db", "llm-openai", "tts-deepgram", "dhara"] }
```
