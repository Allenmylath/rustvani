# Deepgram STT

**File:** `src/services/stt/deepgram.rs`  
**Feature:** `stt-deepgram` (enabled by default)  
**Protocol:** WebSocket (`wss://api.deepgram.com/v1/listen`)

Deepgram streaming speech-to-text. Raw PCM is sent as binary WebSocket messages; final transcripts become `TranscriptionFrame`s. The low-latency default for English and for telephony, where its narrowband handling is strong.

## Pipeline Position

```
transport.input() → DeepgramSttHandler → user_agg → llm → tts → transport.output()
```

## Usage

```rust
use rustvani::services::{DeepgramSttConfig, DeepgramSttHandler};

let stt = DeepgramSttHandler::new(DeepgramSttConfig {
    api_key: std::env::var("DEEPGRAM_API_KEY").unwrap(),
    model: "nova-3".to_string(),
    language: "en-US".to_string(),
    sample_rate: 16_000,
    ..DeepgramSttConfig::default()
})
.into_processor();
```

`new` returns the handler directly (not a `Result`) — the WebSocket is opened on `StartFrame`, not at construction.

## Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `api_key` | `String` | — | Deepgram API key (header: `Authorization: Token …`) |
| `model` | `String` | `"nova-3"` | `"nova-3"`, `"nova-2"`, `"base"`, `"enhanced"` |
| `language` | `String` | `"en-US"` | BCP-47 code, or `"multi"` for multilingual |
| `encoding` | `String` | `"linear16"` | `"linear16"` for raw PCM i16 LE |
| `sample_rate` | `u32` | `16000` | **Must match what the transport sends** |
| `channels` | `u32` | `1` | |
| `interim_results` | `bool` | `true` | Emit non-final results; finals are always emitted |
| `punctuate` | `bool` | `true` | |
| `smart_format` | `bool` | `false` | Formats numbers, dates, currency |
| `endpointing` | `Option<u32>` | `Some(300)` | Silence (ms) before Deepgram finalises. `None` disables server endpointing. |
| `utterance_end_ms` | `Option<u32>` | `None` | Enables `utterance_end` events |
| `base_url` | `Option<String>` | `None` | Override for on-prem / proxy deployments |

### Sample rate must match the transport

`sample_rate` is written into the connection URL, so Deepgram interprets the byte stream at that rate. If it disagrees with `TransportParams::audio_in_sample_rate`, transcripts come back garbled or empty rather than erroring. When a serializer is in play (e.g. Twilio), the serializer has already resampled by the time frames reach STT — use the **pipeline** rate here, not the provider's wire rate.

## Frames

**Consumed:**
- `StartFrame` → connects WebSocket
- `InputAudioRaw` → sends raw binary PCM
- `VADUserStoppedSpeaking` → sends `Finalize` to force a final transcript
- `EndFrame` / `CancelFrame` → disconnect (`CloseStream` + task abort)

**Produced:**
- `TranscriptionFrame` (downstream) on final transcript
- `ErrorFrame` (upstream) on connection / parse errors

## Endpointing vs. local VAD

Deepgram's server-side endpointing and rustvani's local VAD both decide "the user stopped". Running both at aggressive settings produces duplicate turn boundaries. The usual arrangement is to let local VAD own turn-taking (it also drives interruption) and leave `endpointing` at its default as a backstop — the handler already sends `Finalize` on `VADUserStoppedSpeaking`, so the final transcript arrives without waiting for the server timer.

## Environment Variables

```bash
DEEPGRAM_API_KEY=your_key
```

## Cargo Feature

Enabled by default. To build without it:

```toml
[dependencies]
rustvani = { version = "0.4.0-dev.10", default-features = false, features = [
    "transport-websocket", "stt-sarvam", "llm-openai", "tts-deepgram",
] }
```

## See Also

- [Speech Enhancement](audio-enhancement.md) — the HPF/denoise/AGC chain
- [Twilio serializer](serializer-twilio.md) — Deepgram is the usual STT for phone calls
- [Sarvam STT](stt-sarvam.md) · [60db STT](stt-60db.md) · [Gnani STT](stt-gnani.md)
