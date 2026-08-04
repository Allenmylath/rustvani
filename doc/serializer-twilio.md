# Twilio Media Streams Serializer

**Files:** `src/serializers/twilio.rs`, `src/serializers/g711.rs`  
**Feature:** builds under `transport-websocket`; `serializer-twilio` (default) adds REST auto-hangup  
**Protocol:** Twilio Media Streams JSON over WebSocket

Point a Twilio phone number at your rustvani server and you have a phone agent. A `FrameSerializer` sits between `WebSocketTransport` and the provider: outgoing frames are encoded into provider messages, incoming provider messages are decoded into frames. The rest of the pipeline is unchanged — STT, LLM, TTS and VAD never learn that the caller is on a phone.

## How it fits

```
Twilio WS ──► WebSocketTransport.input()   (µ-law 8k → PCM 16k, VAD)
                → DeepgramStt
                → LLMUserAggregator
                → OpenAILLM
                → LLMAssistantAggregator
                → DeepgramTts
                → WebSocketTransport.output()
Twilio WS ◄─────────────────────────────    (PCM 16k → µ-law 8k)
```

## Usage

Twilio sends a `start` event as the first text message on the socket. Parse it, build the serializer from it, install it **before** `run_socket`:

```rust
use rustvani::serializers::{TwilioFrameSerializer, TwilioInputParams, TwilioStart};

// First text message on the socket.
let start = TwilioStart::parse(&text).expect("expected a Twilio `start` event");

let serializer = TwilioFrameSerializer::from_start(
    start,
    std::env::var("TWILIO_AUTH_TOKEN").ok(),  // None disables REST hang-up
    TwilioInputParams { auto_hang_up: true, ..TwilioInputParams::default() },
)?;

transport.set_serializer(Box::new(serializer));   // before run_socket
transport.run_socket(socket, push_tx).await;
```

`TwilioStart` carries `stream_sid`, `call_sid`, and `account_sid`. The lower-level `TwilioFrameSerializer::new(stream_sid, call_sid, account_sid, auth_token, region, edge, base_url, params)` is available when you need to override the REST region/edge/base URL.

## Configuration — `TwilioInputParams`

| Field | Type | Default | Description |
|---|---|---|---|
| `twilio_sample_rate` | `u32` | `8000` | Twilio's wire rate |
| `sample_rate` | `Option<u32>` | `None` | Override the pipeline input rate. `None` uses the rate passed to `setup()`. |
| `auto_hang_up` | `bool` | `true` | Terminate the call over the Twilio REST API on `EndFrame` / `CancelFrame` |
| `resampler_clear_after_secs` | `Option<f32>` | `Some(0.2)` | Retained for pipecat parity; rustvani's resampler does not expose stale-history clearing |

## Transport params for telephony

Telephony audio is narrowband and quiet after µ-law, which changes two settings from their defaults:

```rust
TransportParams {
    audio_in_sample_rate:  Some(16_000),  // pipeline rate, not Twilio's 8k
    audio_out_sample_rate: Some(TTS_SAMPLE_RATE),
    audio_out_10ms_chunks: 2,             // 20 ms — Twilio's media-event cadence
    vad_params: VadParams {
        confidence: 0.45,   // gate on VAD confidence…
        min_volume: 0.0,    // …not raw volume, which µ-law flattens
        ..VadParams::default()
    },
    ..TransportParams::default()
}
```

Downstream services see the **pipeline** rate (16 kHz), because the serializer has already upsampled by the time frames reach STT. Configure `DeepgramSttConfig::sample_rate` to 16 kHz, not 8 kHz.

## Wire behaviour

| Twilio event | Direction | Handling |
|---|---|---|
| `start` | in | Parsed by `TwilioStart::parse` to build the serializer |
| `media` | in | base64 µ-law @ 8 kHz → resampled to pipeline rate → `InputAudioRaw` |
| `dtmf` | in | → `InputDTMFFrame` (see `KeypadEntry`) |
| `media` | out | `OutputAudioRaw` → resampled to 8 kHz → µ-law → base64 |
| `clear` | out | Emitted on interruption — barge-in without a reconnect |

## G.711 codec

`serializers::g711` is a standalone μ-law/A-law codec, usable outside the Twilio path:

```rust
use rustvani::serializers::g711::{pcm_to_ulaw, ulaw_to_pcm, linear_to_ulaw, ulaw_to_linear};
```

`pcm_to_ulaw` / `ulaw_to_pcm` take an optional `&mut StreamResampler` so rate conversion and encoding happen in one pass.

## Hang-up semantics

There are two independent ways the call ends, and you generally want both:

1. **TwiML `<Connect><Stream>`** — the call ends as soon as the WebSocket closes. Ending the pipeline hangs up the caller. This works with no credentials.
2. **REST auto hang-up** — with `auto_hang_up: true` and an auth token, the serializer also POSTs `Status=completed` on `EndFrame`/`CancelFrame`. Requires the `serializer-twilio` feature (on by default), which pulls `reqwest` for the REST call.

Without an auth token, `auto_hang_up` is effectively disabled and (1) does the work.

## Environment Variables

```bash
DEEPGRAM_API_KEY=your_key    # or whichever STT/TTS you wire up
TWILIO_AUTH_TOKEN=…          # optional; enables REST auto hang-up
PUBLIC_HOST=abc123.ngrok.app # optional; defaults to the inbound Host header
```

## Running it locally

The reference server is [`src/bin/twilio_coordinator_server.rs`](../src/bin/twilio_coordinator_server.rs). It builds on default features — no `--features` flag needed.

```bash
cargo run --bin twilio_coordinator_server
ngrok http 8080
# Set the number's "A call comes in" webhook to https://<ngrok-host>/twiml
```

It exposes two routes: `POST|GET /twiml` (the voice webhook, returns TwiML connecting the call to `/ws`) and `GET /ws` (the Media Streams socket Twilio dials back into).

## See Also

- [Transport](transport.md) — `WebSocketTransport` and `run_socket`
- [Deepgram STT](stt-deepgram.md) — the usual STT for phone audio
- [VAD](vad.md) — tuning `confidence` / `min_volume` for narrowband audio
