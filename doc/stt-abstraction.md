# STT abstraction

Every speech-to-text backend is the same program with a different wire format.
`src/services/stt/core/` owns the program; a provider supplies the wire format.

```
                ┌──────────────────── SttService<P> ────────────────────┐
 InputAudioRaw ─┤  AudioFrontend → TurnGate → P::encode_audio → socket  │
                │         ▲                                       │     │
 Transcription ─┤   P::parse ── SttEvent ────────────────────────-─┘     │
                └───────────────────────────────────────────────────────┘
```

`SttService<P>` is the **only** `FrameHandler` in the STT subsystem. Before the
core existed, each provider re-derived the WebSocket handshake, the send and
receive tasks, the byte helpers, the frame-dispatch match and the billing block
— roughly 250–350 lines of plumbing apiece.

## Files

| File | What |
| --- | --- |
| `core/provider.rs` | The `SttProvider` base trait plus `SttEvent`, `Outgoing`, `Handshake`, `AudioSpec`, `WsMessage` |
| `core/driver.rs` | `SttService<P>`, `SttCoreConfig`, `InterimPolicy`; the single `FrameHandler` impl and the receive loop |
| `core/turn_gate.rs` | `TurnGate` — audio gating, pre-roll, stashed VAD stop, duration ledger |
| `core/frontend.rs` | `AudioFrontend` — resample → high-pass → denoise → AGC + limiter; `NoiseBackend` |
| `core/ws.rs` | Handshake builder, `connect`, send task, keepalive task, `WsSink`/`WsStream` |
| `core/util.rs` | `bytes_to_i16`, `i16_to_bytes`, `timestamp`, `percent_encode` |

`src/services/stt/sarvam.rs` is the worked example.

## Adding a provider

Implement `SttProvider`. It answers four questions and owns no state machine:

```rust
use rustvani::{AudioSpec, Handshake, Outgoing, SttEvent, SttProvider, WsMessage};

struct MyProvider { cfg: MyConfig, audio: AudioSpec }

impl SttProvider for MyProvider {
    fn name(&self) -> &'static str { "MyStt" }
    fn audio(&self) -> &AudioSpec { &self.audio }

    fn handshake(&self) -> Handshake {
        Handshake::new(self.cfg.ws_url())
            .header("Authorization", format!("Token {}", self.cfg.api_key))
    }

    fn encode_audio(&self, pcm_le: &[u8]) -> Outgoing {
        Outgoing::Binary(pcm_le.to_vec())
    }

    fn finalize_msg(&self) -> Option<Outgoing> {
        Some(Outgoing::Text(r#"{"type":"Finalize"}"#.into()))
    }

    fn keepalive(&self) -> Option<(Duration, Outgoing)> {
        Some((Duration::from_secs(8), Outgoing::Text(r#"{"type":"KeepAlive"}"#.into())))
    }

    fn parse(&self, msg: WsMessage<'_>) -> SttEvent {
        // → Partial / Final / EmptyFinal / SpeechStarted / SpeechEnded /
        //   Error / Ignore
    }
}
```

Then wrap it:

```rust
let stt = SttService::new(provider, SttCoreConfig::default())
    .with_billing(collector)
    .into_processor();
```

Convention is to keep a `MyProviderSttConfig` with a `split()` returning
`(SttCoreConfig, MyProvider)`, and a thin `MyProviderSttHandler` wrapper — see
`SarvamSttConfig::split`.

### What the provider must NOT do

- Open sockets, spawn tasks, or hold session state. All methods take `&self`.
- Handle VAD frames, order transcripts against turn boundaries, or emit frames.
- Denoise, resample, or level audio — it arrives already at `audio().sample_rate`.
- Record billing. The driver does it, keyed on `name()`.

## What you get for free

**Turn ordering.** `TurnGate` consumes `VADUserStoppedSpeaking` and re-emits it
only after the transcript arrives, with the text *bundled onto the stop frame*.
System and data frames travel different queue lanes and two separate frames
*will* reorder (see `frames/mod.rs`), so one frame is the only structural fix.
`LLMUserAggregator` depends on this ordering.

**Audio gating.** With `audio_gating` on (default), audio reaches the provider
only during a local-VAD-attested turn, plus `pre_roll_ms` of pre-speech captured
while the VAD was still confirming. Server-side VAD therefore has no inter-turn
noise to hallucinate from — spurious transcripts are impossible by construction,
not by bookkeeping. It also cuts cost.

**Billing attribution.** The gate keeps a FIFO of `(epoch, ms-sent)`. A provider
that reports a per-transcript audio duration (`SttEvent::Final { audio_ms }`)
gets exact turn attribution and exact billing; one that reports `None` falls back
to consuming the oldest closed turn whole.

**Audio front-end.** Incoming `AudioRawData` at any sample rate is resampled to
`audio().sample_rate`, then high-passed, denoised (RNNoise or hush-vani), and
levelled by AGC + soft limiter. Before the core, only Sarvam had any of this, and
no provider resampled — a transport/config rate mismatch silently mis-transcribed.

## Interim transcripts

`SttCoreConfig::interim_policy` decides what happens to `SttEvent::Partial`:

- `InterimPolicy::Drop` (default) — discarded.
- `InterimPolicy::Emit` — pushed downstream as `TranscriptionFrame` with
  `finalized: false`, for live captions.

`Emit` is safe because `LLMUserAggregator` forwards non-finalized transcripts
without aggregating them. Without that rule a provider streaming partials would
append every prefix into the LLM turn ("he hello hello there").

## Migration status

| Provider | On the core? |
| --- | --- |
| Sarvam | yes |
| Deepgram | not yet |
| Gnani | not yet |
| 60db | not yet |

The three remaining providers still carry their own plumbing and, because they
forward the VAD stop separately from the transcript, are exposed to the
reordering race the gate exists to eliminate. Migrating them means writing an
`SttProvider` impl and deleting the rest.
