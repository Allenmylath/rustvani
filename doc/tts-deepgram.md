# Deepgram TTS

**File:** `src/services/tts/deepgram.rs`  
**Feature:** `tts-deepgram` (enabled by default)  
**Protocol:** WebSocket (`wss://api.deepgram.com/v1/speak`)

Deepgram Aura streaming text-to-speech. Config is passed via URL query params; audio arrives as binary WS messages. Supports interruption via `Clear` message without reconnecting.

## Pipeline Position

```
llm → assistant_agg → DeepgramTtsHandler → transport.output()
```

## Usage

```rust
use rustvani::services::tts::deepgram::{DeepgramTtsConfig, DeepgramTtsHandler, DeepgramEncoding};

let tts = DeepgramTtsHandler::new(DeepgramTtsConfig {
    api_key: std::env::var("DEEPGRAM_API_KEY").unwrap(),
    voice: "aura-2-helena-en".to_string(),
    encoding: DeepgramEncoding::Linear16,
    sample_rate: 24_000,
    mip_opt_out: Some(true),
    ..Default::default()
}).unwrap().into_processor();
```

## Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `api_key` | `String` | — | Deepgram API key (header: `Authorization: Token …`) |
| `voice` | `String` | `"aura-2-helena-en"` | Deepgram voice model |
| `encoding` | `DeepgramEncoding` | `Linear16` | `Linear16`, `Mulaw`, `Alaw` |
| `sample_rate` | `u32` | `24000` | Output sample rate |
| `mip_opt_out` | `Option<bool>` | `None` | Opt out of Model Improvement Program |
| `base_url` | `String` | `"wss://api.deepgram.com"` | Custom endpoint if needed |

## Frames

**Consumed:**
- `StartFrame` → connects WebSocket
- `LLMFullResponseStart` → begins buffering text
- `LLMText` → sends `Speak` message per token
- `LLMFullResponseEnd` → sends `Flush` to trigger synthesis
- `Interruption` → sends `Clear` (no reconnect needed)
- `EndFrame` / `CancelFrame` → disconnect

**Produced:**
- `OutputAudioRaw` (downstream) as binary PCM/mu-law/alaw

## Timing Logs

The handler prints timing markers to stdout:

```
[123.456] [tts] speak  (42 chars): "Hello world"
[123.457] [tts] flush  ← synthesis starts now
[123.512] [tts] first_audio  ← Deepgram latency: 0.055s
```

## Environment Variables

```bash
DEEPGRAM_API_KEY=your_key
```

## Cargo Feature

Enabled by default. To disable:

```toml
[dependencies]
rustvani = { version = "0.2", default-features = false, features = ["vad-silero", "transport-websocket", "stt-sarvam", "stt-60db", "llm-openai", "tts-sarvam", "dhara"] }
```
