# Sarvam TTS

**File:** `src/services/tts/sarvam.rs`  
**Feature:** `tts-sarvam` (enabled by default)  
**Protocol:** WebSocket (`wss://api.sarvam.ai/text-to-speech/ws`)

Sarvam Bulbul streaming TTS with 25+ voices, sentence-aware buffering, and Indian numbering preprocessing.

## Pipeline Position

```
llm → assistant_agg → SarvamTtsHandler → transport.output()
```

## Usage

```rust
use rustvani::services::tts::sarvam::{SarvamTtsConfig, SarvamTtsHandler};

let tts = SarvamTtsHandler::new(SarvamTtsConfig {
    api_key: std::env::var("SARVAM_API_KEY").unwrap(),
    model: "bulbul:v3".to_string(),
    voice: "shubh".to_string(),
    language: "en-IN".to_string(),
    pace: 1.0,
    ..Default::default()
}).unwrap().into_processor();
```

## Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `api_key` | `String` | — | Sarvam API subscription key |
| `model` | `String` | `"bulbul:v2"` | `bulbul:v2`, `bulbul:v3-beta`, `bulbul:v3` |
| `voice` | `String` | model default | Speaker name (see tables below) |
| `language` | `String` | `"en-IN"` | Target language code |
| `sample_rate` | `Option<u32>` | `None` | Overrides model default if set |
| `pace` | `f32` | `1.0` | Speech speed (model-dependent range) |
| `pitch` | `Option<f32>` | `None` | Pitch shift (v2 only) |
| `loudness` | `Option<f32>` | `None` | Loudness boost (v2 only) |
| `temperature` | `Option<f32>` | `None` | Variability (v3 only) |
| `enable_preprocessing` | `bool` | `false` | Number-to-words preprocessing |
| `min_buffer_size` | `usize` | `50` | Min chars before sentence split |
| `max_chunk_length` | `usize` | `150` | Max chars per TTS chunk |
| `url` | `String` | Sarvam WSS endpoint | Custom endpoint |

## Model Defaults

| Model | Default Rate | Default Speaker | Pitch | Loudness | Temperature | Preprocessing |
|---|---|---|---|---|---|---|
| `bulbul:v2` | 22050 Hz | `anushka` | ✅ | ✅ | ❌ | Optional |
| `bulbul:v3-beta` | 24000 Hz | `shubh` | ❌ | ❌ | ✅ | Forced |
| `bulbul:v3` | 24000 Hz | `shubh` | ❌ | ❌ | ✅ | Forced |

## v2 Speakers

`anushka`, `abhilash`, `manisha`, `vidya`, `arya`, `karun`, `hitesh`

## v3 Speakers

`aditya`, `ritu`, `priya`, `neha`, `rahul`, `pooja`, `rohan`, `simran`, `kavya`, `amit`, `dev`, `ishita`, `shreya`, `ratan`, `varun`, `manan`, `sumit`, `roopa`, `kabir`, `aayan`, `shubh`, `ashutosh`, `advait`, `amelia`, `sophia`

## Frames

**Consumed:**
- `StartFrame` → connects and sends config
- `LLMFullResponseStart` → begins buffering
- `LLMText` → buffers, sentence-splits, sends text chunks
- `LLMFullResponseEnd` → flushes remaining text, sends flush
- `Interruption` → clears buffer, reconnects
- `EndFrame` / `CancelFrame` → disconnect

**Produced:**
- `OutputAudioRaw` (downstream) as 16-bit PCM

## Timing Logs

```
[123.456] [tts] send_text_chunk  (42 chars): "Hello world"
[123.457] [tts] send_flush  ← synthesis starts now
[123.512] [tts] first_audio  ← Sarvam synthesis latency: 0.055s
```

## Environment Variables

```bash
SARVAM_API_KEY=your_key
```

## Cargo Feature

Enabled by default. To disable:

```toml
[dependencies]
rustvani = { version = "0.2", default-features = false, features = ["vad-silero", "transport-websocket", "stt-sarvam", "stt-60db", "llm-openai", "tts-deepgram", "dhara"] }
```
