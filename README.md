[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Allenmylath/rustvani)
[![Crates.io](https://img.shields.io/crates/v/rustvani.svg)](https://crates.io/crates/rustvani)
[![License: BSD-2-Clause](https://img.shields.io/badge/license-BSD--2--Clause-blue.svg)](LICENSE)

<p align="center">
  <img src="assets/rustvanisupercolor.png" alt="rustvani hero" width="100%" />
</p>

# rustvani — वाणी

**High-performance voice agent pipeline framework in Rust.** A from-scratch port of [Pipecat](https://github.com/pipecat-ai/pipecat) designed for production voice AI deployments where latency, memory, and concurrency matter.

> *vānī* (वाणी) — voice, speech, language

```
User speaks → VAD → STT → LLM → TTS → User hears
              ↑                          ↑
         client + server            <500ms
         coordinated VAD           end-to-end
```

---

## Install

```toml
[dependencies]
rustvani = "0.4.0-dev.9"
```

`0.4.0-dev.*` is a **prerelease**, so Cargo needs the version spelled out — `cargo add rustvani` alone will resolve to the last stable, `0.3.0`. Use:

```bash
cargo add rustvani@0.4.0-dev.9
```

Everything in this README describes `0.4.0-dev.9`. The hush-vani noise backend, WebRTC transport, Twilio serializer, and agent swarm do **not** exist in `0.3.0`.

---

## Feature Flags

rustvani ships a lot of providers, so services are behind Cargo features. **Half of what people try first is opt-in** — check this table before filing a "cannot find `SarvamTtsHandler`" issue.

**Enabled by default:** `vad-silero-ort`, `transport-websocket`, `serializer-twilio`, `stt-sarvam`, `stt-60db`, `stt-deepgram`, `llm-openai`, `tts-deepgram`, `dhara`, `db-postgres`.

| Feature | Default | Gates | Notes |
|---|:---:|---|---|
| `vad-silero-ort` | ✅ | `SileroVadOrt` | ONNX Runtime backend (8 kHz + 16 kHz). `SileroVadNative` is **always** compiled and needs no feature. |
| `transport-websocket` | ✅ | `WebSocketTransport`, `ravi`, `serializers` | axum 0.7 + tungstenite |
| `serializer-twilio` | ✅ | Twilio REST auto-hangup | The `TwilioFrameSerializer` itself builds under `transport-websocket` |
| `stt-sarvam` | ✅ | `SarvamSttHandler`, `SarvamLLMHandler` | |
| `stt-60db` | ✅ | 60db STT | |
| `stt-deepgram` | ✅ | `DeepgramSttHandler` | |
| `llm-openai` | ✅ | `OpenAILLMHandler`, `FunctionRegistry` | Any OpenAI-compatible endpoint |
| `tts-deepgram` | ✅ | `DeepgramTtsHandler` | Aura-2 |
| `dhara` | ✅ | `DharaManager` | Implies `llm-openai` + `transport-websocket` |
| `db-postgres` | ✅ | `NeonPostgresTool`, Postgres billing/audio storage | |
| **`tts-sarvam`** | ❌ | `SarvamTtsHandler` | Bulbul v2/v3 |
| **`tts-piper`** | ❌ | `PiperTtsHandler` | Local ONNX; runtime needs `espeak-ng` |
| **`stt-gnani`** | ❌ | `GnaniSttHandler` | Vachana API |
| **`vaniwebrtc`** | ❌ | `VaniWebRTCTransport` | P2P WebRTC. Large dep tree; needs cmake + a C compiler (MSVC on Windows) for `audiopus`/libopus. |

A common "Sarvam end to end" setup:

```toml
[dependencies]
rustvani = { version = "0.4.0-dev.9", features = ["tts-sarvam"] }
```

---

## Why rustvani over Pipecat?

If you've built voice agents with Pipecat (Python), you know the architecture is excellent — frame-based pipelines, clean processor abstractions, interrupt handling. But Python's async runtime, GIL contention, and memory overhead become real problems at scale.

rustvani keeps Pipecat's architecture and fixes the runtime:

| | Pipecat (Python) | rustvani (Rust) |
|---|---|---|
| Runtime | asyncio + threads | Tokio (work-stealing, zero-cost futures) |
| VAD inference | Threadpool executor | `spawn_blocking` on true OS threads |
| Memory per session | ~80–150 MB | ~8–15 MB |
| Frame dispatch | Dynamic dict lookups | Enum dispatch, compiler-verified exhaustive |
| Cold start | 2–5s (interpreter + imports) | <100ms (static binary) |
| Deployment | Docker + Python env | Single static binary, ~15 MB |
| Concurrent sessions | GIL-limited | Truly parallel across all cores |
| Frontend integration | Limited | Deep Dioxus/WASM native binding |

This isn't a wrapper or binding — it's a ground-up Rust implementation that mirrors Pipecat's mental model so you can reason about both codebases interchangeably.

### What rustvani has that Pipecat doesn't

**Built-in speech enhancement DSP chain, with two neural denoisers.** Every audio frame is cleaned in-process before it reaches STT: high-pass filter → neural noise suppression → automatic gain control → soft limiter. Pick **RNNoise** (default, true streaming) or **hush-vani** (DeepFilterNet3-style, stronger suppression) at runtime with one config field. Pure Rust, zero external services, no paid noise-suppression SDK. Pipecat points you at Krisp (paid SDK) or leaves you to wire filters yourself. See [Speech Enhancement](#speech-enhancement--the-audio-front-end).

**Client + Server VAD coordination.** rustvani is designed for deep Dioxus frontend integration. The browser client runs its own lightweight VAD and sends `ClientVADUserStartedSpeaking` events directly into the server pipeline. A toggle-switch CAS gate ensures exactly one `VADUserStartedSpeaking` is emitted regardless of which side fires first — no double-triggers, no race conditions. Pipecat has no equivalent.

**Dhara conversation flow engine.** Node-based state machine where each node owns its own system prompt, tool set, and context strategy. Handlers return `Stay` or `Transition { next_node }` — full multi-turn flow control without orchestration boilerplate.

**Zero-dependency VAD and end-of-turn detection.** The native Silero backend and the SmartTurn end-of-turn model are both pure Rust — no ONNX Runtime, no dynamic libraries, no `.so` files to bundle. One binary, everything included.

**P2P WebRTC without an SFU.** `vaniwebrtc` carries Opus over real peer-to-peer SRTP with no media server in the path.

**Production-tested.** Deployed for a Kerala government voice agent serving real users across Malayalam, Hindi, and English.

---

## Quick Start

A complete voice agent server. This is [`examples/quickstart.rs`](examples/quickstart.rs) verbatim — a real example in this repo, so you can check it yourself. It builds with **default features only**:

```bash
cargo build --example quickstart
```

```rust
use std::sync::Arc;

use rustvani::axum::{
    extract::{ws::WebSocket, State, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
    Router,
};
use rustvani::processors::{
    llm_assistant_aggregator::LLMAssistantAggregator, llm_user_aggregator::LLMUserAggregator,
};
use rustvani::services::{
    DeepgramTtsConfig, DeepgramTtsHandler, OpenAILLMConfig, OpenAILLMHandler, SarvamSttConfig,
    SarvamSttHandler,
};
use rustvani::transport::{TransportParams, WebSocketParams, WebSocketTransport};
use rustvani::{
    shared_context, system_clock, PipelineParams, PipelineTask, SileroVadNative, VadParams,
};

#[derive(Clone)]
struct AppState {
    sarvam_key: String,
    openai_key: String,
    deepgram_key: String,
}

/// One fully isolated pipeline per connection.
async fn handle_connection(socket: WebSocket, state: AppState) {
    // 1. VAD — pure Rust, no ONNX Runtime, no .so files to bundle.
    let vad = match SileroVadNative::new(16_000) {
        Ok(v) => Arc::new(v),
        Err(e) => return log::error!("VAD init failed: {e}"),
    };

    // 2. Transport — owns the VAD and the audio I/O.
    let transport = WebSocketTransport::new(
        "quickstart",
        WebSocketParams {
            transport: TransportParams {
                audio_in_enabled: true,
                audio_in_sample_rate: Some(16_000),
                audio_out_enabled: true,
                audio_out_sample_rate: Some(24_000), // Deepgram TTS default
                vad_analyzer: Some(vad),
                vad_params: VadParams { confidence: 0.4, min_volume: 0.1, ..Default::default() },
                ..TransportParams::default()
            },
        },
    );

    // 3. Shared conversation context — the aggregators read and write it.
    let context = shared_context(Some(
        "You are a helpful voice assistant. Keep answers to one or two sentences.".into(),
    ));

    // 4. Services. The speech-enhancement chain (HPF → RNNoise → AGC → limiter)
    //    is already on inside the STT handler; nothing to wire up.
    let stt = SarvamSttHandler::new(SarvamSttConfig {
        api_key: state.sarvam_key,
        model: "saaras:v3".into(),
        language: Some("en-IN".into()),
        ..SarvamSttConfig::default()
    })
    .into_processor();

    let llm = OpenAILLMHandler::new(OpenAILLMConfig {
        api_key: state.openai_key,
        model: "gpt-4o-mini".into(),
        ..OpenAILLMConfig::default()
    })
    .into_processor();

    let tts = match DeepgramTtsHandler::new(DeepgramTtsConfig {
        api_key: state.deepgram_key,
        ..DeepgramTtsConfig::default()
    }) {
        Ok(t) => t.into_processor(),
        Err(e) => return log::error!("TTS init failed: {e}"),
    };

    // 5. Aggregators bridge VAD/STT ↔ LLM. `new` already returns a
    //    FrameProcessor — no `.into_processor()` here.
    let user_agg = LLMUserAggregator::new(context.clone());
    let assistant_agg = LLMAssistantAggregator::new(context.clone());

    // 6. Assemble and run.
    let task = PipelineTask::new(
        vec![
            transport.input(),
            stt,
            user_agg,
            llm,
            assistant_agg,
            tts,
            transport.output(),
        ],
        PipelineParams { allow_interruptions: true, ..PipelineParams::default() },
    );

    // Take the injection handle *before* run() — it can only be taken once.
    let push_tx = task.push_sender();

    tokio::join!(
        async { task.run(system_clock(), None).await.ok(); },
        transport.run_socket(socket, push_tx),
    );
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_connection(socket, state))
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let state = AppState {
        sarvam_key: std::env::var("SARVAM_API_KEY").expect("SARVAM_API_KEY"),
        openai_key: std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY"),
        deepgram_key: std::env::var("DEEPGRAM_API_KEY").expect("DEEPGRAM_API_KEY"),
    };

    let app = Router::new().route("/ws", get(ws_handler)).with_state(state);
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();

    log::info!("listening on ws://0.0.0.0:8080/ws");
    rustvani::axum::serve(listener, app).await.unwrap();
}
```

Client → server is raw i16 LE PCM, 16 kHz mono, over binary WebSocket frames. Server → client is raw i16 LE PCM at the TTS sample rate.

> **axum version pinning:** `WebSocketTransport::run_socket` takes an `axum::extract::ws::WebSocket` by value. Build your router off the `rustvani::axum` re-export (as above) so your axum and rustvani's can't drift into the confusing "expected `WebSocket`, found `WebSocket`" error.

More complete programs live in [`examples/`](examples/) and [`src/bin/`](src/bin/) — including a Twilio phone agent, a WebRTC server, and full billing + recording wiring.

---

## Deploy in 5 Minutes

### Docker (single static binary)

```dockerfile
FROM rust:1-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
COPY . .
RUN cargo build --release --bin your-bot

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/your-bot /usr/local/bin/
WORKDIR /app
CMD ["your-bot"]
```

No Python, no virtualenv, no `requirements.txt`. The image is ~50 MB total.

Two build-time caveats:

- **`tts-piper`** additionally needs `espeak-ng` in the *runtime* image.
- **`vaniwebrtc`** additionally needs `cmake` and a C/C++ toolchain in the *builder* image (libopus via `audiopus`).

### Environment variables

Only the keys for services you actually enable are required.

```bash
SARVAM_API_KEY=your_key     # Sarvam STT / TTS / LLM
DEEPGRAM_API_KEY=your_key   # Deepgram STT / TTS
SIXTYDB_API_KEY=your_key    # 60db STT
GNANI_API_KEY=your_key      # Gnani STT
OPENAI_API_KEY=your_key     # or any OpenAI-compatible endpoint
DATABASE_URL=postgres://…   # Postgres tool, billing storage, audio metadata
TWILIO_ACCOUNT_SID=…        # Twilio serializer auto-hangup
TWILIO_AUTH_TOKEN=…
```

### Fly.io (scale-to-zero)

```toml
# fly.toml
[build]
  dockerfile = "Dockerfile"

[[services]]
  internal_port = 8080
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0

[[services.ports]]
  port = 443
  handlers = ["tls", "http"]
```

```bash
fly launch
fly secrets set SARVAM_API_KEY=… OPENAI_API_KEY=… DEEPGRAM_API_KEY=…
fly deploy
```

Your voice agent is live. Zero idle cost when no users are connected.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  PipelineTask                                                    │
│                                                                  │
│  [TaskSource] → Transport.Input → STT → UserAgg →                │
│                 LLM → AssistantAgg → TTS → Transport.Output →    │
│                 [TaskSink]                                       │
│                                                                  │
│  Upstream  ◄────────────────────────────────────────────────     │
│  Downstream ────────────────────────────────────────────────►    │
└──────────────────────────────────────────────────────────────────┘

VAD sits in Transport.Input — fires VADUserStartedSpeaking /
VADUserStoppedSpeaking frames that drive the STT and aggregation.
```

### Core concepts (1:1 with Pipecat)

**Frames** — Typed messages that flow through the pipeline. Three categories: System (lifecycle, VAD signals, audio input), Control (end, LLM response boundaries), and Data (transcriptions, LLM text, audio output, function calls). Every frame has a unique ID and optional sibling ID for broadcast deduplication.

**FrameProcessor** — The universal building block. Every component (VAD, STT, LLM, TTS, transport, pipeline itself) is a `FrameProcessor`. Each has two async queues: an input queue (system frames get priority) and a process queue (data/control frames). This two-queue design ensures lifecycle frames like `InterruptionFrame` are never blocked behind a backlog of audio chunks.

**Pipeline** — Chains processors into a linked list with source/sink sentinels. A Pipeline IS a FrameProcessor, so pipelines nest inside pipelines.

**PipelineTask** — Lifecycle wrapper. Manages setup, StartFrame injection, heartbeats, idle timeout, and graceful shutdown. Exposes callback hooks (`on_pipeline_started`, `on_pipeline_finished`, `on_idle_timeout`) and a `push_sender()` for external frame injection from your transport.

→ Deep dive: [architecture.md](architecture.md)

---

## Modules

```
src/
├── adapters/          LLM provider adapters (OpenAI wire format)
│   └── schemas/       Provider-agnostic tool/function schemas
├── agents/            Multi-agent swarm — bus, registry, runner, coordinator
├── audio_capture/     Per-turn WAV recording (user + bot tracks) + storage
├── audio_process/     Speech enhancement + resampling
│   ├── agc/           High-pass filter, AGC, soft limiter
│   ├── noisefilter/   RNNoise (nnnoiseless)
│   ├── hushfilter/    DeepFilterNet3-style (hush-vani)
│   └── resamplers/    Streaming sample-rate conversion (rubato)
├── billing/           Usage tracking + storage backends
│   └── storage/       LogBillingStorage (JSON logs) + PostgresBillingStorage
├── clock/             Pipeline clock abstraction (BaseClock, SystemClock)
├── context/           Shared LLMContext (messages, tools, tool_choice)
├── dhara/             Conversation flow engine (node-based state machine)
├── error/             PipecatError + Result
├── frames/            Frame types, FrameProcessor, priority queues
├── metrics/           TTFB / processing / token-usage metric hooks
├── observer/          BaseObserver — frame-level tracing hooks
├── pipeline/          Pipeline assembly + PipelineTask lifecycle
├── processors/        LLM user/assistant aggregators
├── ravi/              RAVI protocol (real-time audio/video interface)
├── serializers/       Wire-protocol adapters — Twilio Media Streams, G.711
├── services/
│   ├── llm/           OpenAI + Sarvam LLM (SSE streaming, function calling)
│   ├── stt/           Sarvam + 60db + Deepgram + Gnani STT (WebSocket streaming)
│   └── tts/           Sarvam + Deepgram TTS (WebSocket) + Piper TTS (local ONNX)
├── tools/             Built-in tools (Neon Postgres with pgvector)
├── transport/         WebSocket (axum), P2P WebRTC, base I/O, ChannelTransport
│   ├── websocket/
│   └── vaniwebrtc/
├── turn/              SmartTurn end-of-turn model (pure Rust Whisper features)
├── utils/             Sentence splitter, text preprocessor, model cache
└── vad/               Silero VAD (native Rust + ONNX) + state machine
```

---

## Features

### Speech Enhancement — the Audio Front-End

STT accuracy lives or dies on input audio quality. Real users call from noisy streets on cheap phone mics — too quiet, too loud, full of rumble and background noise. rustvani runs every audio frame through a studio-style processing chain *before* it reaches the STT provider:

```
raw mic audio
   │
   ▼
┌──────────────────────┐   DC offset, rumble, handling noise below 90 Hz
│ High-pass filter     │   (2nd-order Butterworth)
└──────────────────────┘
   │
   ▼
┌──────────────────────┐   Neural noise suppression — pure Rust.
│ RNNoise or hush-vani │   Backend selected at runtime; resampling handled
└──────────────────────┘   transparently (16k ↔ 48k for RNNoise).
   │
   ▼
┌──────────────────────┐   Quiet speakers boosted (up to +30 dB), loud ones
│ AGC                  │   tamed — normalized to −20 dBFS. Fast attack (10 ms),
└──────────────────────┘   slow release (400 ms), gain held during silence so
   │                       the noise floor is never pumped up between words
   ▼
┌──────────────────────┐   Peaks compressed smoothly toward full scale —
│ Soft limiter         │   hard digital clipping is impossible by construction
└──────────────────────┘
   │
   ▼
clean, consistently-levelled audio → STT
```

The entire chain is **pure Rust, in-process, and on by default**. No Krisp SDK, no external denoising service, no per-minute cleanup fees.

```rust
use rustvani::{NoiseBackend, SarvamSttConfig, SarvamSttHandler};

let stt = SarvamSttHandler::new(SarvamSttConfig {
    api_key: std::env::var("SARVAM_API_KEY").unwrap(),
    noise_reduction: true,                    // denoiser on        (default: true)
    noise_backend:   NoiseBackend::Rnnoise,   // which one          (default: Rnnoise)
    agc:             true,                    // HPF + AGC + limiter (default: true)
    ..SarvamSttConfig::default()
})
.into_processor();
```

#### Choosing a noise backend

Both backends implement the `StreamingDenoiser` trait (`filter` / `flush` / `reset`), so the STT path swaps them behind one `Box<dyn>`.

| | `NoiseBackend::Rnnoise` *(default)* | `NoiseBackend::HushVani` |
|---|---|---|
| Model | RNNoise (`nnnoiseless`) | DeepFilterNet3-style (`hush-vani`) |
| Nature | True streaming, 10 ms frames | Batch API, wrapped in a sliding window |
| Suppression | Good | Stronger, especially on non-stationary noise |
| Added latency | ~10 ms (one frame) | ~10 ms (160 samples held back per call) |
| Native rate | 48 kHz (resampled in/out) | 16 kHz (resampled in/out) |

**How hush-vani is made streaming.** `hush-vani` only exposes a batch `enhance()` whose GRUs start from zero on every call — feed it 20 ms chunks and you get 20 ms of cold-start artifacts, every chunk. rustvani wraps it in a sliding window instead: each call re-runs `enhance` over **200 ms of prior audio plus the new samples**, which re-primes the GRUs to roughly the state a true streaming decode would hold. The context region of the output is discarded and only the new samples are emitted. Because `enhance` output lags its input by 160 samples, the last frame of each call is held back and drained in `flush()` at end of utterance — so total output length ≈ total input length. Re-computing the context every call is affordable: `hush-vani` runs at roughly 100× real-time.

`hush-vani` is a regular dependency — there is no feature flag to enable, just set `noise_backend`.

The pieces (`RNNoiseFilter`, `HushVaniFilter`, `AudioEnhancer`) are also usable standalone, and the AGC is fully tunable via `AgcConfig` — target level, max gain, attack/release, noise gate, limiter knee. The adapted gain is remembered across utterances, so the same speaker isn't re-learned from scratch every sentence.

→ Full guide: [doc/audio-enhancement.md](doc/audio-enhancement.md)

### Voice Activity Detection

Two backends, same API:

```rust
// Pure Rust — zero ONNX Runtime dependency, 16kHz only. Always available.
let vad = SileroVadNative::new(16_000)?;

// ONNX Runtime — 8kHz + 16kHz, same model as Pipecat. Needs `vad-silero-ort`.
let vad = SileroVadOrt::new(16_000)?;
```

- 4-state machine: `Quiet → Starting → Speaking → Stopping → Quiet`
- `VadParams`: `confidence`, `start_secs`, `stop_secs`, `min_volume`
- Volume calculation using dBFS approximation of EBU R128
- Inference runs on `spawn_blocking` — never stalls the Tokio executor

Installed from crates.io, `SileroVadNative` has its weights **embedded at compile time**, so deployed binaries never touch the network. Built from a source checkout without the bundled model, it falls back to fetching them once into the rustvani cache directory. `SileroVadOrt` always fetches `silero.onnx` into the cache on first use.

→ [doc/vad.md](doc/vad.md)

### SmartTurn — ML End-of-Turn Detection

VAD tells you the user went quiet. It cannot tell you whether they *finished*. SmartTurn is a local end-of-turn model that defers the stop event on hesitation pauses ("my number is… uh… 98…") instead of letting the bot barge in mid-thought.

Entirely pure Rust: Whisper-style mel feature extraction (`src/turn/whisper_features.rs`) with mel filters embedded via `include_bytes!`, and a hand-rolled inference engine. **No ONNX Runtime, no Python, no `.so` files.** The gzipped weights are fetched once into the rustvani cache directory on first use, or you can ship them yourself and point `weights_path` at the file for a fully offline deployment.

```rust
use rustvani::turn::SmartTurnConfig;

let params = TransportParams {
    vad_analyzer: Some(vad),                       // required
    turn_config:  Some(SmartTurnConfig::default()), // None = VAD-only
    ..TransportParams::default()
};
```

`SmartTurnConfig` exposes `stop_secs`, `pre_speech_ms`, `max_duration_secs`, `precision` (F32/F16), `resampler_quality`, and `weights_path`. Weights default to the rustvani cache directory (`~/.rustvani/cache/`, or `%LOCALAPPDATA%\rustvani\cache` on Windows). `TurnMetrics` reports `is_complete`, `probability`, and `e2e_processing_time_ms`.

→ [doc/turn-acid.md](doc/turn-acid.md) · [doc/turn-acid-phase2.md](doc/turn-acid-phase2.md)

### Client + Server VAD Coordination (Dioxus Integration)

No other voice framework has this: the browser/Dioxus client runs its own lightweight VAD and pushes events directly into the server pipeline. A shared atomic toggle ensures exactly one `VADUserStartedSpeaking` is emitted per utterance regardless of which side detects speech first.

The methods live on the **input** transport (`BaseInputTransport`), which is what `transport.input()` wraps:

```rust
// Called from your WebSocket handler when the Dioxus client reports speech.
input_transport.push_client_vad_started(&processor, timestamp).await;
input_transport.push_client_vad_stopped(&processor, timestamp).await;
```

The coordination rule: `emitted_speaking` is an `AtomicBool` shared between client and server paths. The first source to win `compare_exchange(false, true)` emits the event; the second is a no-op. This eliminates double-triggers with zero locking overhead.

### Speech-to-Text
- **Sarvam AI** streaming WebSocket STT (`saaras:v3`) — transcription, translation, verbatim, transliteration, codemix modes; `ml-IN`, `hi-IN`, `en-IN`, auto-detect (`unknown`)
- **Deepgram** — WebSocket streaming (`nova-3`), the low-latency default for English and telephony
- **60db STT** — real-time WebSocket streaming with 39 languages, two-phase finals (fast dictation + LLM-refined canonical), and diarization
- **Gnani (Vachana) STT** — WebSocket streaming for Indic languages (`hi-IN`, `ta-IN`, `en-IN`, etc.) — feature `stt-gnani`
- Integrated **speech enhancement chain**, on by default (see [Speech Enhancement](#speech-enhancement--the-audio-front-end))
- **Audio gating** — audio is forwarded to the provider only during VAD-attested turns plus pre-roll, which eliminates spurious server-VAD transcripts by construction and cuts STT cost
- Transparent resampling if source rate ≠ target rate (via `rubato`)

→ Per-service guides: [Sarvam](doc/stt-sarvam.md) · [Deepgram](doc/stt-deepgram.md) · [60db](doc/stt-60db.md) · [Gnani](doc/stt-gnani.md)

### Large Language Models
- **OpenAI-compatible** API with SSE streaming
- **Sarvam LLM** (`sarvam-m`, `sarvam-30b`) with optional CoT thinking mode
- Full function calling with re-invocation loop (model calls tool → execute → re-invoke)
- Configurable max tool rounds to prevent infinite loops
- Automatic context-window trimming with a per-model token table, overridable via `context_window_tokens`
- Provider adapter system — add new providers by implementing `LLMAdapter`

→ [OpenAI](doc/llm-openai.md) · [Sarvam](doc/llm-sarvam.md)

### Text-to-Speech
- **Deepgram Aura** TTS — WebSocket streaming with Aura-2 voices, interruption via `Clear` without reconnect *(default feature)*
- **Sarvam Bulbul** TTS (v2, v3-beta, v3) — WebSocket streaming with 25+ voices *(feature `tts-sarvam`)*
- **Piper TTS** — fully local ONNX inference, zero network calls *(feature `tts-piper`)*
  - espeak-ng phonemization → Piper ONNX → chunked PCM streaming
  - Multiple quality levels (Low/Medium/High)
  - Shared model across pipeline instances via `Arc<Mutex<PiperModel>>`
- Sentence-aware text buffering with abbreviation detection (Mr., Dr., IPC., etc.)
- Indian numbering system preprocessing for TTS (10000 → "ten thousand")

→ [Sarvam](doc/tts-sarvam.md) · [Deepgram](doc/tts-deepgram.md) · [Piper (local)](doc/tts-piper.md)

### Telephony — Twilio Media Streams

Point a Twilio phone number at your rustvani server and you have a phone agent. The `FrameSerializer` layer sits between `WebSocketTransport` and the provider: outgoing frames become provider messages, incoming provider messages become frames.

```rust
use rustvani::{TwilioFrameSerializer, TwilioInputParams, TwilioStart};

// `start` is the parsed Twilio `start` handshake off the WebSocket.
let serializer = TwilioFrameSerializer::from_start(
    start,
    twilio_auth_token,   // Option<String> — None disables REST hang-up
    TwilioInputParams { auto_hang_up: true, ..TwilioInputParams::default() },
)?;
transport.set_serializer(Box::new(serializer));
```

- G.711 μ-law/A-law codec (`serializers::g711`) with transparent 8 kHz ↔ pipeline-rate resampling
- Barge-in maps to Twilio's `clear` event — no reconnect
- `auto_hang_up` terminates the call over the Twilio REST API on `EndFrame`/`CancelFrame` *(feature `serializer-twilio`, on by default)*
- Set `audio_out_10ms_chunks: 2` to match Twilio's 20 ms media cadence

Working server: [`src/bin/twilio_coordinator_server.rs`](src/bin/twilio_coordinator_server.rs).
→ [doc/serializer-twilio.md](doc/serializer-twilio.md)

### WebRTC Transport (P2P, no SFU)

`vaniwebrtc` carries audio over a real peer-to-peer WebRTC connection — Opus over RTP/SRTP with **no SFU or media server in the path**. Signaling (SDP offer/answer + trickle ICE) runs over a WebSocket; control messages ride a WebRTC data channel.

```rust
use rustvani::transport::{VaniWebRTCParams, VaniWebRTCTransport};

let transport = VaniWebRTCTransport::new("webrtc", VaniWebRTCParams {
    transport: TransportParams { /* … same params as any transport … */ },
    ..Default::default()
});
```

`TurnServer` configures TURN/STUN, and `build_shared_udp_mux` lets many sessions share one UDP port — the thing that makes a single container hold hundreds of calls.

Opt-in (`vaniwebrtc`): pulls a large dep tree and needs cmake + a C compiler for libopus.
Server: [`src/bin/vaniwebrtc_server.rs`](src/bin/vaniwebrtc_server.rs) · Browser client: [`examples/vaniwebrtc_client.html`](examples/vaniwebrtc_client.html)
→ [doc/vaniwebrtc.md](doc/vaniwebrtc.md)

### Function Calling & Tools

```rust
let mut registry = FunctionRegistry::new();

// Simple — result string goes directly to LLM context
registry.register("get_weather", |args: String| async move {
    let city = serde_json::from_str::<Value>(&args)?["city"].as_str().unwrap_or("unknown");
    format!("Weather in {city}: 28°C, partly cloudy")
});

// Data — summary to LLM, full structured data as a downstream frame for UI/logging
registry.register_data("search_cases", |args: String| async move {
    let rows = db_query(&args).await?;
    ToolCallOutput::with_data(format!("Found {} cases", rows.len()), json!(rows))
});

let llm = OpenAILLMHandler::with_shared_registry(config, registry);
```

Built-in **Neon Postgres tool** (schema caching, parameterized queries, pgvector similarity search, structured filters — the LLM never writes raw SQL):

```rust
let pg = Arc::new(NeonPostgresTool::from_env()); // reads DATABASE_URL
llm.add_tool(pg);
// Registers: pg_schema, pg_query, pg_refine, pg_vector_search
```

→ [doc/tools-postgres.md](doc/tools-postgres.md)

### Dhara — Conversation Flow Engine

> *dhara* (ധാര) — flow, stream

Node-based conversation flow where each node owns its system prompt, tools, and context strategy:

```rust
let mut dhara = DharaManager::new(context.clone(), registry.clone());

dhara.register_node("greeting", greeting_node, vec![
    ("check_availability", availability_handler),
    ("transfer_to_billing", |_| async { TransitionResult::Transition { next: "billing".into() } }),
]);
dhara.register_node("billing", billing_node, vec![...]);
dhara.set_initial_node("greeting");

llm.set_transition_hook(dhara.create_transition_hook());
```

### Agent Swarm — Multi-Agent Coordination

For workloads a single pipeline shouldn't own — a voice agent that hands research off to a background worker, or several specialists behind one caller. Each agent owns its own `PipelineTask` and they communicate over an `AgentBus`, orchestrated by an `AgentRunner`. No global state.

```rust
use rustvani::agents::{AgentRunner, BaseAgent, LocalAgentBus, TaskContext};
```

- **`LocalAgentBus`** — two-priority fan-out. System messages (End/Cancel/Activate/urgent replies/registry) are never dropped and always overtake queued data; data messages are dropped-and-counted rather than blocking when a subscriber is full. Control never drops, data never blocks.
- **`BaseAgent`** — task router. Register job handlers with `on_task(name, handler)`; each handler runs in its own tokio task with a `TaskRequestCtx` for `complete` / `stream_start` / `stream_data` / `stream_end`.
- **`TaskContext::dispatch`** — ready-gated: waits (watch-based, no polling) for the target to appear in the registry instead of silently dropping the request.
- **`BusOutputEdge`** — a tail-of-pipeline `FrameProcessor` that republishes frames to peer agents, so one agent's pipeline output becomes another's input.
- **`CoordinatorProcessor`** — a bus-connected frame processor for agentless coordination inside a single pipeline.

→ Full guide: [agents.md](agents.md)

### Billing & Usage Tracking

Production-grade, non-blocking billing that captures exactly what you need to cost and invoice voice sessions — session duration, LLM tokens, TTS characters, STT audio duration, and full conversation transcripts, all linked by `session_id` and written to PostgreSQL or structured JSON logs.

| Signal | Source | Accuracy |
|---|---|---|
| Session duration (seconds) | Pipeline start/end hooks | Exact |
| LLM input + output tokens | OpenAI `stream_options.include_usage` | Exact |
| TTS characters synthesised | Per-flush confirmation (Deepgram / Sarvam) | Exact |
| STT audio duration | Server-reported or PCM byte counter | Exact / computed |

`record()` is a single send onto an **unbounded** channel drained by a background task — it never blocks and never drops, so billing overhead is invisible to audio latency. (`SessionBilling::new`'s `channel_capacity` argument is retained for API compatibility and is ignored.) Wiring is one builder call per service:

```rust
let (billing, drain_handle) = SessionBilling::new(session_id, storage, 256);

let stt = SarvamSttHandler::new(config).with_billing(billing.clone()).into_processor();
let llm = OpenAILLMHandler::new(config).with_billing(billing.clone()).into_processor();
// ... attach to PipelineParams { billing_collector: Some(billing), .. }
```

→ Full guide — PostgreSQL schemas, cost queries, transcript capture, log-only mode: [doc/billing.md](doc/billing.md)

### Audio Capture — Session Recording

Records the conversation as **two synchronized tracks** — one for the user, one for the bot — so you can review a call, mix them into a single overlay, or feed them back into evaluation. Segments are linked to the billing transcript by turn id.

```rust
let (audio_cap, _drain) = SessionAudioCapture::new(session_id, storage, 64);
let audio_proc = AudioCaptureProcessor::new(audio_cap, user_turn_id, bot_turn_id);
// place after TTS, before transport.output()
```

Storage backends: `LocalAudioStorage` (WAV on disk) and `PostgresAudioMetaStorage` (metadata in Postgres, feature `db-postgres`).

→ [doc/audio-capture.md](doc/audio-capture.md)

### Testing Without a Server

`ChannelTransport` drives a full pipeline through plain `mpsc` channels — feed PCM in, assert on what comes out, no WebSocket server needed:

```rust
let transport = ChannelTransport::new("test", params, incoming_rx);
incoming_tx.send(ChannelMessage::Audio(pcm_bytes)).await?;
let result = outgoing_rx.recv().await;  // transcripts, TTS audio, events
```

Runnable version: [`examples/channel_pipeline.rs`](examples/channel_pipeline.rs) → full example: [doc/transport.md](doc/transport.md)

### Observability

`BaseObserver` gets a callback for every frame processed and every frame pushed, with processor name, direction, and timestamp — enough to build per-turn latency breakdowns (VAD stop → transcript → first LLM token → first TTS chunk) without touching pipeline code. Pass it to `task.run(clock, Some(observer))`. See the `LatencyObserver` in [`src/bin/websocket_server.rs`](src/bin/websocket_server.rs).

---

## Documentation

Every service and component has a dedicated guide in [doc/](doc/README.md) — exact config fields, environment variables, and feature flags:

| Area | Guides |
|---|---|
| Overview | [Architecture](architecture.md) · [Agents](agents.md) |
| Audio front-end | [Speech Enhancement](doc/audio-enhancement.md) · [VAD](doc/vad.md) · [SmartTurn](doc/turn-acid.md) |
| STT | [Sarvam](doc/stt-sarvam.md) · [Deepgram](doc/stt-deepgram.md) · [60db](doc/stt-60db.md) · [Gnani](doc/stt-gnani.md) |
| LLM | [OpenAI](doc/llm-openai.md) · [Sarvam](doc/llm-sarvam.md) |
| TTS | [Sarvam](doc/tts-sarvam.md) · [Deepgram](doc/tts-deepgram.md) · [Piper (local)](doc/tts-piper.md) |
| Transport | [WebSocket + Channel](doc/transport.md) · [WebRTC](doc/vaniwebrtc.md) · [Twilio serializer](doc/serializer-twilio.md) |
| Tools | [Postgres tool](doc/tools-postgres.md) |
| Observability | [Billing](doc/billing.md) · [Audio capture](doc/audio-capture.md) |

---

## For Pipecat Developers

If you know Pipecat, you already know rustvani. The mapping is 1:1:

| Pipecat (Python) | rustvani (Rust) |
|---|---|
| `FrameProcessor` | `FrameProcessor` |
| `Frame` subclasses | `Frame { inner: FrameInner }` enum |
| `Pipeline(processors)` | `PipelineTask::new(processors, params)` |
| `OpenAILLMService` | `OpenAILLMHandler` |
| `LLMUserResponseAggregator` | `LLMUserAggregator` |
| `LLMAssistantResponseAggregator` | `LLMAssistantAggregator` |
| `SileroVADAnalyzer` | `SileroVadNative` / `SileroVadOrt` |
| `SmartTurnAnalyzer` | `SmartTurnAnalyzer` |
| `FunctionCallHandler` | `FunctionRegistry` |
| `FlowManager` | `DharaManager` |
| `RTVIProcessor` | `RaviProcessor` |
| `TwilioFrameSerializer` | `TwilioFrameSerializer` |
| `BaseWorker` / agent bus | `BaseAgent` / `AgentBus` |
| `@transport.event_handler("on_client_connected")` | `task.add_on_pipeline_started(...)` |
| `isinstance(frame, VADUserStartedSpeakingFrame)` | `matches!(frame.inner, FrameInner::System(SystemFrame::VADUserStartedSpeaking { .. }))` |

The frame flow, interrupt semantics, aggregator logic, and pipeline nesting all work identically. If you've debugged a Pipecat bot, you can debug a rustvani bot.

---

## Project Status

rustvani is in active development. Core pipeline, frame system, and all listed services are functional and battle-tested in production for a Kerala government voice agent deployment.

**Working:**
- Full pipeline lifecycle (start, interruption, cancel, end)
- Silero VAD — native Rust + ONNX
- SmartTurn ML end-of-turn detection (pure Rust, zero runtime deps)
- Client + Server VAD coordination (Dioxus frontend integration)
- Speech enhancement chain — high-pass filter → **RNNoise or hush-vani** → AGC → soft limiter (pure Rust, on by default) + streaming resampling
- Sarvam STT / TTS / LLM
- Deepgram STT (nova-3) + Deepgram TTS (Aura-2)
- 60db STT (WebSocket streaming, 39 languages)
- Gnani STT (Vachana API, Indic languages)
- OpenAI-compatible LLM with function calling + re-invocation loop
- Piper TTS (local ONNX, zero network)
- Dhara conversation flow manager
- Agent swarm — bus, registry, runner, task routing, coordinator processor
- RAVI protocol
- Neon Postgres tool with pgvector
- WebSocket transport (axum) + **P2P WebRTC transport** + ChannelTransport (testing)
- **Twilio Media Streams serializer** with G.711 and REST auto-hangup
- Billing & usage tracking — session duration, LLM tokens, TTS chars, STT audio duration; PostgreSQL + log storage backends; non-blocking hot path
- Audio capture — synchronized user/bot WAV tracks with local + Postgres storage
- Available on [crates.io](https://crates.io/crates/rustvani)

**Planned:**
- Anthropic / Gemini LLM adapters (only the OpenAI wire format ships today)
- Whisper STT
- ElevenLabs / PlayHT TTS

---

## License

Rustvani is released under BSD-2-Clause. See [LICENSE](LICENSE).

Portions of this project are derived from [Pipecat](https://github.com/pipecat-ai/pipecat) by Daily and retain Pipecat's BSD-2-Clause license notice. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## Acknowledgements

rustvani wouldn't exist without [Pipecat](https://github.com/pipecat-ai/pipecat) by Daily. The architecture, frame taxonomy, aggregator patterns, and pipeline design are all derived from their excellent work.

Built with [Sarvam AI](https://www.sarvam.ai/) for Indian language voice — STT, TTS, and LLM services that actually work for Malayalam, Hindi, and 10+ Indian languages.
