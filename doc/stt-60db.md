# 60db STT

**File:** `src/services/stt/sixtydb.rs`  
**Feature:** `stt-60db` (enabled by default)  
**Protocol:** WebSocket (`wss://api.60db.ai/ws/stt`)

Real-time streaming STT with 39 languages, two-phase finals (fast dictation + LLM-refined canonical), RNNoise denoising, automatic resampling, and optional speaker diarization.

## Pipeline Position

```
transport.input() → SixtyDbSttHandler → llm → tts → transport.output()
```

## Usage

```rust
use rustvani::services::stt::sixtydb::{SixtyDbSttConfig, SixtyDbSttHandler, SixtyDbEncoding};

let stt = SixtyDbSttHandler::new(SixtyDbSttConfig {
    api_key: std::env::var("SIXTYDB_API_KEY").unwrap(),
    languages: vec!["en".to_string()],
    encoding: SixtyDbEncoding::Linear,
    sample_rate: 16_000,
    noise_reduction: true,
    ..Default::default()
}).into_processor();
```

## Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `api_key` | `String` | — | 60db API key (query param) |
| `languages` | `Vec<String>` | `["en"]` | Language codes; empty = auto-detect |
| `context` | `Option<SixtyDbContext>` | `None` | Context hints for LLM refinement |
| `encoding` | `SixtyDbEncoding` | `Linear` | `Linear` (16-bit PCM) or `Mulaw` (G.711) |
| `sample_rate` | `u32` | `16000` | Target sample rate; auto-resampled if input differs |
| `utterance_end_ms` | `u32` | `500` | Silence duration before finalizing (clamped ≥300 ms) |
| `continuous_mode` | `bool` | `true` | Keep session alive between utterances |
| `interim_results_frequency` | `Option<u32>` | `Some(300)` | Partial result interval; `None` = disabled |
| `audio_enhancement` | `SixtyDbAudioEnhancement` | `Off` | `Off` / `Light` / `Adaptive` |
| `diarize` | `bool` | `false` | Speaker diarization |
| `min_speakers` | `Option<i32>` | `None` | Lower bound when `diarize=true` |
| `max_speakers` | `Option<i32>` | `None` | Upper bound when `diarize=true` |
| `insecure` | `bool` | `false` | Use `ws://` instead of `wss://` |
| `noise_reduction` | `bool` | `true` | RNNoise denoising before sending |
| `resampler_quality` | `ResamplerQuality` | `Quick` | `Quick` / `Medium` / `High` |

## Context Hints

```rust
use rustvani::services::stt::sixtydb::{SixtyDbContext, SixtyDbContextItem};

let context = SixtyDbContext {
    general: vec![
        SixtyDbContextItem { key: "domain".into(), value: "healthcare".into() },
    ],
    text: Some("appointment scheduling".into()),
    terms: vec!["hypertension".into(), "diabetes".into()],
};
```

## Frames

**Consumed:**
- `StartFrame` → connects WebSocket
- `InputAudioRaw` → denoise → resample → encode → send
- `EndFrame` / `CancelFrame` → disconnect

**Produced:**
- `TranscriptionFrame` (downstream) on transcript
- `UserStartedSpeaking` (downstream) on `speech_started` (barge-in)
- `UserStoppedSpeaking` (downstream) on canonical final
- `ErrorFrame` (upstream) on errors

## Environment Variables

```bash
SIXTYDB_API_KEY=your_key
```

## Cargo Feature

Enabled by default. To disable:

```toml
[dependencies]
rustvani = { version = "0.2", default-features = false, features = ["vad-silero-ort", "transport-websocket", "stt-sarvam", "llm-openai", "tts-deepgram", "dhara"] }
```
