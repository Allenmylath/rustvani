# Gnani STT (Vachana)

**File:** `src/services/stt/gnani.rs`  
**Feature:** `stt-gnani`  
**Protocol:** WebSocket (`wss://api.vachana.ai/stt/v3/stream`)

Connects to Gnani's Vachana streaming STT API. Emits `TranscriptionFrame` downstream when transcripts arrive.

## Pipeline Position

```
transport.input() → GnaniSttHandler → llm → tts → transport.output()
```

## Usage

```rust
use rustvani::services::stt::gnani::{GnaniSttConfig, GnaniSttHandler};

let stt = GnaniSttHandler::new(GnaniSttConfig {
    api_key: std::env::var("GNANI_API_KEY").unwrap(),
    language_code: "hi-IN".to_string(),
    sample_rate: 16_000,
    format: "verbatim".to_string(),
    itn_native_numerals: false,
}).into_processor();
```

## Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `api_key` | `String` | — | Vachana API key (header: `x-api-key-id`) |
| `language_code` | `String` | `"en-IN"` | BCP-47 language code e.g. `hi-IN`, `ta-IN` |
| `sample_rate` | `u32` | `16000` | Audio sample rate. Accepted: `8000`, `16000`, `44100`, `48000` |
| `format` | `String` | `"verbatim"` | `"verbatim"` = raw spoken-form; `"transcribe"` = ITN enabled |
| `itn_native_numerals` | `bool` | `false` | Render digits in native script when `format="transcribe"` |

## Frames

**Consumed:**
- `StartFrame` → connects WebSocket
- `InputAudioRaw` → buffers into 1024-byte chunks → sends binary
- `VADUserStoppedSpeaking` → flushes remaining buffered audio
- `EndFrame` / `CancelFrame` → disconnects WebSocket

**Produced:**
- `TranscriptionFrame` (downstream) on transcript
- `ErrorFrame` (upstream) on connection / parse errors

## Audio Format

- Raw PCM signed 16-bit little-endian, mono
- Chunked into 1024-byte (512 sample) binary WebSocket frames

## Environment Variables

```bash
GNANI_API_KEY=your_key
```

## Cargo Feature

Enable in `Cargo.toml`:

```toml
[dependencies]
rustvani = { version = "0.2", features = ["stt-gnani"] }
```

Or use the feature directly in your crate if depending on rustvani as a library.
