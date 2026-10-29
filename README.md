[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Allenmylath/rustvani)
[![Crates.io](https://img.shields.io/crates/v/rustvani.svg)](https://crates.io/crates/rustvani)
[![docs.rs](https://img.shields.io/docsrs/rustvani)](https://docs.rs/rustvani)
[![License: BSD-2-Clause](https://img.shields.io/badge/license-BSD--2--Clause-blue.svg)](LICENSE)

<p align="center">
  <img src="assets/rustvanisupercolor.png" alt="rustvani hero" width="100%" />
</p>

# rustvani — वाणी

**A voice AI framework in Rust: real-time speech pipelines for voice agents, live captions, translation and dubbing.** A from-scratch port of [Pipecat](https://github.com/pipecat-ai/pipecat)'s frame-pipeline architecture, built for deployments where latency, memory and concurrency matter.

> *vānī* (वाणी) — voice, speech, language

Two pipeline shapes ship out of the box, assembled from the same processors:

```
Voice agent
  audio in → VAD → STT → user aggregator → LLM (+ tools) → assistant aggregator → TTS → audio out

Live captions, translation and dubbing
  audio in → VAD → realtime STT → SentenceSegmenter → TranslationService → DubbingService → your sink
             partial transcripts      sentences as        TranslationFrame        DubbedSpeechFrame
             as the words are said    soon as they end    per target language     (PCM, streamed)
```

---

## Table of contents

- [Install](#install)
- [Feature flags](#feature-flags)
- [Why rustvani](#why-rustvani)
- [Quick start: a voice agent](#quick-start-a-voice-agent)
- [Live captions, translation and dubbing](#live-captions-translation-and-dubbing)
- [Deploy](#deploy)
- [Architecture](#architecture)
- [Modules](#modules)
- [Features](#features)
- [Examples, binaries and tests](#examples-binaries-and-tests)
- [Documentation](#documentation)
- [For Pipecat developers](#for-pipecat-developers)
- [Project status](#project-status)
- [License](#license)

---

## Install

```toml
[dependencies]
rustvani = "0.4.0-dev.21"
```

`0.4.0-dev.*` is a **prerelease**, so Cargo needs the version spelled out: a plain `cargo add rustvani` resolves to the last stable release, `0.3.0`, which has none of the STT core, translation, dubbing, WebRTC, Twilio or agent-swarm work described here. Use:

```bash
cargo add rustvani@0.4.0-dev.21
```

Everything in this README describes `0.4.0-dev.21`. The [CHANGELOG](CHANGELOG.md) lists what each `dev.N` added.

Two common manifests:

```toml
# A Sarvam voice agent (Sarvam TTS is opt-in).
rustvani = { version = "0.4.0-dev.21", features = ["tts-sarvam"] }

# A captioning / dubbing service that wants a small dependency tree:
# no ONNX Runtime, no Postgres, no OpenAI client.
rustvani = { version = "0.4.0-dev.21", default-features = false,
             features = ["transport-websocket", "stt-sarvam", "translate-sarvam", "tts-sarvam"] }
```

Minimum supported Rust: a recent stable toolchain (edition 2021). `vaniwebrtc` additionally needs cmake and a C/C++ compiler; `tts-piper` needs `espeak-ng` at runtime.

---

## Feature flags

rustvani ships many providers, so each is behind a Cargo feature that pulls in only its dependency. **Several things people reach for first are opt-in**, so check this table before filing a "cannot find `SarvamTtsHandler`" issue.

**Enabled by default:** `vad-silero-ort`, `transport-websocket`, `serializer-twilio`, `stt-sarvam`, `stt-60db`, `stt-deepgram`, `llm-openai`, `llm-sarvam`, `translate-sarvam`, `tts-deepgram`, `dhara`, `db-postgres`.

| Feature | Default | Gates | Notes |
|---|:---:|---|---|
| `vad-silero-ort` | ✅ | `SileroVadOrt` | ONNX Runtime backend (8 kHz and 16 kHz). The pure-Rust `SileroVadNative` is **always** compiled and needs no feature. |
| `transport-websocket` | ✅ | `transport`, `ravi`, `serializers` modules; the `rustvani::axum` re-export | axum 0.7 + tungstenite. `ChannelTransport` lives here too. |
| `serializer-twilio` | ✅ | Twilio REST auto-hangup | The `TwilioFrameSerializer` itself builds under `transport-websocket`; only the hang-up call needs `reqwest`. |
| `stt-sarvam` | ✅ | `SarvamSttHandler`, `SarvamRealtimeSttHandler` | Streaming and realtime (partials) endpoints |
| `stt-deepgram` | ✅ | `DeepgramSttHandler` | |
| `stt-60db` | ✅ | `SixtyDbSttHandler` | |
| `llm-openai` | ✅ | `OpenAILLMHandler`, `FunctionRegistry` | Any OpenAI-compatible endpoint |
| `llm-sarvam` | ✅ | `SarvamLLMHandler` | |
| `translate-sarvam` | ✅ | `SarvamTranslateProvider` | `TranslationService` and the `TranslationProvider` trait need no feature |
| `tts-deepgram` | ✅ | `DeepgramTtsHandler` | Aura-2 |
| `dhara` | ✅ | `dhara` module | Implies `llm-openai` and `transport-websocket` |
| `db-postgres` | ✅ | `NeonPostgresTool`, `PostgresBillingStorage`, `PostgresAudioMetaStorage` | tokio-postgres + native-tls |
| **`tts-sarvam`** | ❌ | `SarvamTtsHandler`, `SarvamRestTts`, `SarvamStreamTts` | Bulbul v2/v3: the streaming agent handler, and the one-utterance synthesizers dubbing uses |
| **`tts-piper`** | ❌ | `PiperTtsHandler` | Local ONNX; runtime needs `espeak-ng` |
| **`stt-gnani`** | ❌ | `GnaniSttHandler` | Vachana API |
| **`vaniwebrtc`** | ❌ | `VaniWebRTCTransport` | Peer-to-peer WebRTC. Large dependency tree; needs cmake and a C compiler (MSVC on Windows) for `audiopus`/libopus. |

The shared STT core (`SttService`, `SttProvider`, `TurnGate`, `AudioFrontend`, `NoiseBackend`) is compiled whenever any `stt-*` feature is on. `SentenceSegmenter`, `TranslationService`, `DubbingService`, `TtsSynthesizer`, billing and audio capture are always compiled.

---

## Why rustvani

Pipecat's architecture is excellent: frame-based pipelines, clean processor abstractions, interrupt handling. rustvani keeps that mental model and replaces the runtime.

| | Pipecat (Python) | rustvani (Rust) |
|---|---|---|
| Runtime | asyncio + threads, GIL | Tokio, work-stealing, sessions truly parallel across cores |
| Frame dispatch | Dynamic `isinstance` chains | Enum match, exhaustive at compile time |
| Audio payloads | Copied per hop | `bytes::Bytes`, a clone is a reference-count bump |
| Model inference (VAD, end-of-turn) | Thread-pool executor | `spawn_blocking`, never on the async workers |
| Deployment | Docker image with a Python environment | One static binary, Silero weights embedded |
| Noise suppression | Points you at a paid SDK | Two neural denoisers built in, pure Rust |

This is not a wrapper or a binding. It is a ground-up Rust implementation that mirrors Pipecat's frames, aggregators and pipeline lifecycle, so you can reason about both codebases the same way.

### What rustvani has that Pipecat does not

**A shared STT core with a turn gate.** Every STT provider built on `SttService` gets the same program for free: audio forwarded only during VAD-attested turns (plus pre-roll), the closing transcript bundled onto the `VADUserStoppedSpeaking` frame so the LLM can never see the stop before the words, a per-turn duration ledger for billing, live partials, mid-turn finalisation for speakers who never pause, and a socket that reopens itself when the provider closes it. A provider contributes its wire format and nothing else. See [Speech-to-text](#speech-to-text).

**Live captions, translation and dubbing as pipeline stages.** `SentenceSegmenter` finalises sentences out of partial transcripts, `TranslationService` turns every final into one `TranslationFrame` per target language, and `DubbingService` speaks each translation and emits a `DubbedSpeechFrame` carrying the audio and the source transcript's timestamp, streamed from the first chunk. Targets can change while the pipeline runs. See [Live captions, translation and dubbing](#live-captions-translation-and-dubbing).

**A speech-enhancement front end, pure Rust.** High-pass filter, then RNNoise or a DeepFilterNet3-style denoiser, then automatic gain control and a soft limiter, in process, before audio reaches the STT. No Krisp SDK, no per-minute clean-up fees. See [Speech enhancement](#speech-enhancement-the-audio-front-end).

**Zero-dependency VAD and end-of-turn detection.** The native Silero backend and the SmartTurn end-of-turn model are pure Rust: no ONNX Runtime, no `.so` files. The Silero weights are compiled into the binary.

**Client and server VAD that agree.** A browser can run its own VAD and send start/stop events; one atomic toggle guarantees exactly one `VADUserStartedSpeaking` per utterance whichever side fires first.

**Dhara conversation flows from JSON.** Nodes, prompts, tools and transitions in `dhara.json`, handlers in Rust, validated at load. **An agent swarm** with a two-priority bus. **Peer-to-peer WebRTC** with no media server. **Billing** with a per-minute durable checkpoint and a crash sweeper.

**Production-tested.** Deployed for a Kerala government voice agent serving real users in Malayalam, Hindi and English, and as the speech engine behind a live multilingual captioning and dubbing product.

---

## Quick start: a voice agent

A complete voice agent server. This is [`examples/quickstart.rs`](examples/quickstart.rs) verbatim, so it is compiler-checked. It builds with **default features only**:

```bash
cargo build --example quickstart
SARVAM_API_KEY=… OPENAI_API_KEY=… DEEPGRAM_API_KEY=… cargo run --example quickstart
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

**Wire protocol of the raw WebSocket transport.** Client to server: binary frames of raw i16 little-endian PCM, 16 kHz mono (512-sample chunks are typical; [`tests/test.py`](tests/test.py) is a minimal client). Server to client: binary frames of raw i16 PCM at the TTS sample rate, and a text frame `{"type":"interruption"}` when the bot is cut off, so the client can flush its playback buffer. Text frames from the client are JSON: RAVI messages (`"label":"ravi"`), or the bare types `client_vad_start`, `client_vad_stop` and `client_interruption` described under [Client and server VAD](#client-and-server-vad-coordination).

> **axum version pinning.** `WebSocketTransport::run_socket` takes an `axum::extract::ws::WebSocket` by value. Build your router off the `rustvani::axum` re-export, as above, so your axum and rustvani's cannot drift into the confusing "expected `WebSocket`, found `WebSocket`" error.

More complete programs live in [`examples/`](examples/) and [`src/bin/`](src/bin/): a Twilio phone agent, a WebRTC server, Dhara flows backed by Postgres, and full billing plus recording wiring. See [Examples, binaries and tests](#examples-binaries-and-tests).

---

## Live captions, translation and dubbing

The same processors assemble into a captioning engine: transcripts of a speaker as they talk, each sentence translated into any number of languages, and each translation spoken aloud so a host can play it where the words were said. This needs `stt-sarvam`, `translate-sarvam` and `tts-sarvam`.

```rust
use std::sync::Arc;

use rustvani::{
    DubConfig, DubbingService, SarvamRealtimeSttConfig, SarvamRealtimeSttHandler, SarvamStreamTts,
    SarvamStreamTtsConfig, SarvamTranslateConfig, SarvamTranslateProvider, SegmenterConfig,
    SentenceSegmenter, TranslationService, TranslationTargets,
};

// The languages someone is watching. The handle is shared: change it at any time.
let targets = TranslationTargets::new(["en", "hi"]);

// Sarvam's realtime endpoint: partial transcripts while the speaker is still talking.
let stt = SarvamRealtimeSttHandler::new(SarvamRealtimeSttConfig {
    api_key: sarvam_key.clone(),
    language: Some("ml-IN".into()), // or None to detect it
    ..Default::default()
})
.into_processor();

// A sentence whose end shows up in a partial goes downstream as a final at once,
// instead of waiting ~2 s for the pause and the STT's own final.
let segmenter = SentenceSegmenter::new(SegmenterConfig::default()).into_processor();

// Every final transcript → one TranslationFrame per target, one ordered queue per language.
let translate = TranslationService::new(
    SarvamTranslateProvider::new(SarvamTranslateConfig { api_key: sarvam_key.clone(), ..Default::default() }),
    targets.clone(),
)
.into_processor();

// Every translation → one DubbedSpeechFrame, the audio streamed from its first chunk.
let dub = DubbingService::new(
    Arc::new(SarvamStreamTts::new(SarvamStreamTtsConfig { api_key: sarvam_key, ..Default::default() })),
    targets.clone(),
    DubConfig::default(),
)
.into_processor();

let task = PipelineTask::new(
    vec![transport.input(), stt, segmenter, translate, dub, your_sink, transport.output()],
    PipelineParams::default(),
);
```

What comes out, and where to catch it:

| Frame | Carries | Emitted by |
|---|---|---|
| `TranscriptionFrame` with `finalized: false` | The current hypothesis, with its language when detected | realtime STT, then rewritten by the segmenter to the unfinalised remainder |
| `TranscriptionFrame` with `finalized: true`, or the transcript bundled on `VADUserStoppedSpeaking` | A sentence or a turn's final text | the segmenter and the STT turn gate |
| `TranslationFrame` (`TranslationData`) | `text`, `target_language`, plus the source transcript's `source_text`, `source_language`, `timestamp` | `TranslationService` |
| `DubbedSpeechFrame` (`DubbedSpeechData`) | PCM (s16le), `sample_rate`, `voice`, `pace`, the source `timestamp`; with a streaming voice, the first chunk plus a `SpeechStream` for the rest | `DubbingService` |

`your_sink` is any `FrameHandler` that matches on these frames (a WebSocket fan-out, an SFU track writer). Browser clients can instead be driven by the [RAVI protocol](#ravi-protocol): `RaviObserver` sends `user-transcription` and `user-translation` messages for every transcript and translation.

Measured on All India Radio Malayalam news with the realtime endpoint: first text 0.6–1.1 s after speech starts and words on screen 1.3–1.8 s after they are spoken, at 3.9–5.2% character error. The classic streaming endpoint delivered the same text 6.4 s after, or 3.5 s with an 8 s `max_turn_ms`. A `PaceHint` lets the host fit each dubbed utterance into the time the original took.

→ [Translation](doc/translate-sarvam.md) · [Dubbing](doc/dubbing.md) · [STT abstraction](doc/stt-abstraction.md)

---

## Deploy

### Docker (single static binary)

```dockerfile
FROM rust:1-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y pkg-config libssl-dev curl && rm -rf /var/lib/apt/lists/*
COPY . .
RUN cargo build --release --bin your-bot

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/your-bot /usr/local/bin/
WORKDIR /app
CMD ["your-bot"]
```

No Python, no virtualenv, no `requirements.txt`. The repository's own [`Dockerfile`](Dockerfile) builds any `src/bin/` binary with `--build-arg BIN=… --build-arg FEATURES=…`.

Build-time and runtime caveats:

- **Model weights.** `SileroVadNative` weights are embedded into the binary when the crate is installed from crates.io, so a deployed binary never downloads anything for VAD. SmartTurn weights and the Silero ONNX model are fetched into the rustvani cache directory (`~/.rustvani/cache`, or `RUSTVANI_CACHE_DIR`) at build time by `build.rs` when the network allows, and otherwise on first use. For a fully offline image, copy `smart_turn_weights.bin.gz` there yourself or set `SmartTurnConfig::weights_path`.
- **`tts-piper`** needs `espeak-ng` in the *runtime* image and the Piper `.onnx` voices on disk.
- **`vaniwebrtc`** needs `cmake` and a C/C++ toolchain in the *builder* image (libopus via `audiopus`), and one forwardable UDP port (see [WebRTC](#webrtc-transport-peer-to-peer-no-sfu)).

### Environment variables

Only the keys for services you actually enable are required.

```bash
SARVAM_API_KEY=your_key     # Sarvam STT / realtime STT / TTS / LLM / Translate
DEEPGRAM_API_KEY=your_key   # Deepgram STT / TTS
SIXTYDB_API_KEY=your_key    # 60db STT
GNANI_API_KEY=your_key      # Gnani STT
OPENAI_API_KEY=your_key     # or any OpenAI-compatible endpoint
DATABASE_URL=postgres://…   # Postgres tool, billing storage, audio metadata
TWILIO_AUTH_TOKEN=…         # Twilio serializer auto-hangup
RUSTVANI_CACHE_DIR=…        # optional: where model weights are cached
RUST_LOG=info               # e.g. "rustvani=debug,info"
```

### Fly.io (scale-to-zero)

```toml
# fly.toml
[build]
  dockerfile = "Dockerfile"

[http_service]
  internal_port = 8080
  auto_stop_machines = "stop"
  auto_start_machines = true
  min_machines_running = 0
```

```bash
fly launch
fly secrets set SARVAM_API_KEY=… OPENAI_API_KEY=… DEEPGRAM_API_KEY=…
fly deploy
```

Zero idle cost when no users are connected.

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

VAD sits in Transport.Input. It fires VADUserStartedSpeaking /
VADUserStoppedSpeaking frames that drive the STT turn gate and the aggregators.
```

### Core concepts (1:1 with Pipecat)

**Frames** are typed messages: `Frame { id, sibling_id, inner: FrameInner }` with three categories. *System* frames carry lifecycle, VAD signals, audio input, DTMF and RAVI messages. *Control* frames mark the end of the pipeline and LLM/function-call boundaries. *Data* frames carry transcriptions, translations, dubbed speech, LLM text, output audio and tool results. `Frame::kind()` gives a flat `FrameKind` for filters, and `AudioRawData.audio` is a `bytes::Bytes`, so cloning an audio frame never copies the payload.

**FrameProcessor** is the universal building block. Every component, including the pipeline itself, is one. Each has two queues: an input queue where system frames take priority, and a process queue for data and control frames. Lifecycle frames such as `InterruptionFrame` therefore never wait behind a backlog of audio. An interruption drains both queues, keeping only the uninterruptible frames (`End`, `EndTask`, `StopTask`, `CancelTask`). You write a `FrameHandler` and wrap it with `FrameProcessor::new(name, handler, direct_mode)`.

**Pipeline** chains processors into a linked list with source and sink sentinels. A pipeline *is* a `FrameProcessor`, so pipelines nest.

**PipelineTask** is the lifecycle wrapper: setup, `StartFrame` injection, optional heartbeats and idle timeout, graceful shutdown, and a `push_sender()` for frames from outside (your transport). It exposes `add_on_pipeline_started`, `add_on_pipeline_finished`, `add_on_pipeline_error`, `add_on_idle_timeout`, frame-reached callbacks with per-direction `FrameKind` filters, and a `lifecycle_receiver()` watch channel. When `PipelineParams::billing_collector` is set it records `SessionStart` and `SessionEnd` itself.

**The turn contract.** Every provider built on the STT core guarantees that a turn's transcript reaches the aggregator *before* the `VADUserStoppedSpeaking` that closes it, because the closing transcript rides on that same frame. `LLMUserAggregator` relies on this: it has exactly one trigger path (the stop frame on an open turn), interims are forwarded but never aggregated, and a transcript arriving with no open turn is deferred to the next flush rather than answered on its own. The LLM handlers stage tool rounds and commit them at the round boundary, so a barge-in mid-round leaves no orphaned `tool_calls` in the context ([doc/turn-acid.md](doc/turn-acid.md)).

→ Deep dive: [architecture.md](architecture.md)

---

## Modules

```
src/
├── adapters/          LLM provider adapters (OpenAI wire format)
│   └── schemas/       Provider-agnostic FunctionSchema / ToolsSchema / ToolChoice
├── agents/            Multi-agent swarm: bus, registry, runner, task dispatch, coordinator
├── audio_capture/     Session recording (user + bot tracks → one mixed WAV) + storage
├── audio_process/     Speech enhancement + resampling
│   ├── agc/           High-pass filter, AGC, soft limiter (AudioEnhancer)
│   ├── noisefilter/   RNNoise (nnnoiseless)
│   ├── hushfilter/    DeepFilterNet3-style (hush-vani), made streaming
│   └── resamplers/    Streaming sample-rate conversion (rubato)
├── billing/           Usage events, SessionBilling collector, checkpointing
│   └── storage/       LogBillingStorage (JSON log lines), PostgresBillingStorage
├── clock/             BaseClock, SystemClock
├── context/           LLMContext: messages, tools, staged tool rounds, trimming
├── dhara/             Conversation flows: dhara.json loader + handler registry (and the legacy DharaManager)
├── error/             PipecatError + Result
├── frames/            Frame types, FrameProcessor, priority queues
├── metrics/           Metric hooks (placeholders today)
├── observer/          BaseObserver: per-frame tracing hooks
├── pipeline/          Pipeline assembly + PipelineTask lifecycle
├── processors/        LLMUserAggregator, LLMAssistantAggregator, SentenceSegmenter
├── ravi/              RAVI protocol: RaviProcessor (inbound), RaviObserver (outbound)
├── serializers/       Wire-protocol adapters: Twilio Media Streams, G.711
├── services/
│   ├── stt/core/      SttService, SttProvider, TurnGate, AudioFrontend: shared by every provider on it
│   ├── stt/           Sarvam (streaming + realtime), Deepgram, 60db, Gnani
│   ├── translate/     TranslationService, TranslationProvider, Sarvam Translate
│   ├── dub/           DubbingService, SpeechStream, PaceHint
│   ├── tts/           Deepgram, Sarvam Bulbul (streaming handler, REST and streaming synthesizers), Piper
│   │   └── synth.rs   TtsSynthesizer: one utterance in, its audio out
│   └── llm/           OpenAI + Sarvam chat completions, FunctionRegistry
├── tools/             BuiltinTool trait, Neon Postgres tool (pgvector, structured filters)
├── transport/         BaseTransport, WebSocketTransport, ChannelTransport, incoming text dispatch
│   └── vaniwebrtc/    Peer-to-peer WebRTC (Opus/SRTP, signaling, Opus fmtp munging)
├── turn/              SmartTurn end-of-turn model (pure Rust Whisper features + INT8 transformer)
├── utils/             Sentence splitter, Indian-numbering text preprocessor, model cache
└── vad/               Silero VAD (native + ONNX), VadInput rate conversion, state machine, VadProcessor
```

---

## Features

### Speech enhancement: the audio front end

STT accuracy lives or dies on input audio. Real users call from noisy streets on cheap phone microphones: too quiet, too loud, full of rumble and background noise. Providers on the STT core run every frame through a processing chain *before* it reaches the provider:

```
raw mic audio
   │
   ▼
┌──────────────────────┐   DC offset, rumble, handling noise below 90 Hz
│ High-pass filter     │   (2nd-order Butterworth)
└──────────────────────┘
   │
   ▼
┌──────────────────────┐   Neural noise suppression, pure Rust.
│ RNNoise or hush-vani │   Backend selected at runtime; resampling handled
└──────────────────────┘   transparently (16k ↔ 48k for RNNoise).
   │
   ▼
┌──────────────────────┐   Quiet speakers boosted (up to +30 dB), loud ones
│ AGC                  │   tamed, normalised to −20 dBFS. Fast attack (10 ms),
└──────────────────────┘   slow release (400 ms), gain held during silence so
   │                       the noise floor is never pumped up between words
   ▼
┌──────────────────────┐   Peaks compressed smoothly toward full scale;
│ Soft limiter         │   hard digital clipping is impossible by construction
└──────────────────────┘
   │
   ▼
clean, consistently levelled audio → STT
```

The chain is **pure Rust, in process, and on by default** for every provider on the STT core (today: Sarvam streaming and Sarvam realtime). It also resamples from whatever rate the transport delivers to the rate the provider wants.

```rust
use rustvani::{NoiseBackend, SarvamSttConfig, SarvamSttHandler};

let stt = SarvamSttHandler::new(SarvamSttConfig {
    api_key: std::env::var("SARVAM_API_KEY").unwrap(),
    noise_reduction: true,                    // denoiser on          (default: true)
    noise_backend:   NoiseBackend::Rnnoise,   // which one            (default: Rnnoise)
    agc:             true,                    // HPF + AGC + limiter  (default: true)
    ..SarvamSttConfig::default()
})
.into_processor();
```

#### Choosing a noise backend

Both backends implement the `StreamingDenoiser` trait (`filter` / `flush` / `reset`), so the front end swaps them behind one `Box<dyn>`.

| | `NoiseBackend::Rnnoise` *(default)* | `NoiseBackend::HushVani` |
|---|---|---|
| Model | RNNoise (`nnnoiseless`) | DeepFilterNet3-style (`hush-vani`) |
| Nature | True streaming, 10 ms frames | True streaming, 10 ms frames (`HushStream`) |
| Suppression | Good | Stronger, especially on non-stationary noise |
| Added latency | ~10 ms (one frame) | ~10 ms (160 samples held back per call) |
| Native rate | 48 kHz (resampled in/out) | 16 kHz (resampled in/out) |

**How hush-vani streams.** rustvani runs hush-vani's `HushStream`: one 10 ms frame at a time, with the model's GRUs and STFT normalisation carried from frame to frame, so the output matches a batch `enhance()` over the same audio. The 160-sample lag is dropped at the start of each utterance, so `filter` + `flush` give back exactly the input's length, lined up with it; `reset()` between utterances keeps the model's state, so each utterance does not pay the normalisation's 1 s settling again. An init failure logs and falls back to RNNoise rather than failing the pipeline.

`hush-vani` is a regular dependency; there is no feature flag, just set `noise_backend`. The pieces (`RNNoiseFilter`, `HushVaniFilter`, `AudioEnhancer`, `StreamResampler`) are also usable standalone, and the AGC is tunable via `AgcConfig` (target level, max gain, attack and release, noise gate, limiter knee). The adapted gain is kept across utterances, so the same speaker is not re-learned from scratch every sentence.

→ Full guide: [doc/audio-enhancement.md](doc/audio-enhancement.md)

### Voice activity detection

Two backends, same API:

```rust
// Pure Rust, no ONNX Runtime, 16 kHz model. Always available; weights embedded.
let vad = SileroVadNative::new(16_000)?;

// ONNX Runtime, 8 kHz and 16 kHz, the same model as Pipecat. Needs `vad-silero-ort`.
let vad = SileroVadOrt::new(16_000)?;
```

- A four-state machine ported from Pipecat: `Quiet → Starting → Speaking → Stopping → Quiet`, gated on confidence *and* a dBFS volume floor.
- `VadParams { confidence, start_secs, stop_secs, min_volume }`. The defaults (`0.5`, `0.2 s`, `0.8 s`, `0.2`) were measured on real speech: after a browser's audio processing and Opus, Silero's confidence stays high for runs of only ~0.36 s, so the old 0.4 s start window let 31–62% of a speaker's words never reach the STT. The new defaults pass 99–100% and open no turns on white noise down to −30 dBFS. Pass `VadParams` explicitly to keep other values.
- **Any input rate or channel count.** `VadInput` converts each frame from the rate and channels it carries to mono at the model's rate, so 48 kHz Opus decodes and stereo SFU taps work unchanged. Audio already in the model's format is borrowed, not copied.
- Inference runs on `spawn_blocking` and never stalls the Tokio workers. Every ready window in a frame is analysed, not just the first, so wide frames (a Twilio upsample is 960 samples) never leave the VAD behind.
- `VadProcessor` runs the same machine as a standalone processor for pipelines that have no transport.

Installed from crates.io, `SileroVadNative` has its weights **embedded at compile time**. Built from a source checkout without the bundled model it fetches them once into the cache directory. `SileroVadOrt` always fetches `silero.onnx` into the cache on first use.

→ [doc/vad.md](doc/vad.md)

### SmartTurn: ML end-of-turn detection

VAD tells you the user went quiet. It cannot tell you whether they *finished*. SmartTurn is a local end-of-turn model that defers the stop event on hesitation pauses ("my number is… uh… 98…") instead of letting the bot barge in mid-thought.

Entirely pure Rust: Whisper-style mel feature extraction with the mel filters embedded, and a hand-rolled INT8 transformer inference engine (about 17 MB in memory). **No ONNX Runtime, no Python, no `.so` files.** The gzipped weights are fetched once into the cache directory, or ship them yourself and point `weights_path` at the file for an offline deployment. Inference takes hundreds of milliseconds of CPU, so it runs on Tokio's blocking pool; if it ever panics, the turn ends and the input falls back to VAD-only.

```rust
use rustvani::turn::SmartTurnConfig;

let params = TransportParams {
    vad_analyzer: Some(vad),                        // required
    turn_config:  Some(SmartTurnConfig::default()), // None = VAD-only
    ..TransportParams::default()
};
```

`SmartTurnConfig` exposes `stop_secs` (3.0), `pre_speech_ms` (500), `max_duration_secs` (8), `precision` (F32/F64), `resampler_quality` and `weights_path`. `TurnMetrics` reports `is_complete`, `probability` and `e2e_processing_time_ms`.

→ [doc/turn-acid.md](doc/turn-acid.md) · [doc/turn-acid-phase2.md](doc/turn-acid-phase2.md)

### Client and server VAD coordination

A browser or native client (a Dioxus/WASM front end, for instance) can run its own lightweight VAD and tell the server when speech starts and stops, so the turn opens the moment the user speaks rather than after the server's confirmation window. A shared atomic toggle on the input transport guarantees exactly one `VADUserStartedSpeaking` per utterance, whichever side detects speech first: the first source to win `compare_exchange(false, true)` emits the event; the second is a no-op. No locks, no double triggers.

The client sends JSON text messages on the same connection it sends audio on (WebSocket text frame, or the WebRTC data channel):

```json
{"type": "client_vad_start", "timestamp": 1718000000.123}
{"type": "client_vad_stop",  "timestamp": 1718000001.456}
```

From Rust, push `Frame::client_vad_user_started_speaking(ts)` through `task.push_sender()`, or send `ChannelMessage::ClientVadStart(ts)` on a `ChannelTransport`. [`tests/vad_coordination_test.rs`](tests/vad_coordination_test.rs) pins the exactly-once behaviour; [`examples/vaniwebrtc_client.html`](examples/vaniwebrtc_client.html) is a browser client that does it.

### Speech-to-text

Every STT backend is the same program with a different wire format. The **STT core** (`services::stt::core`) owns the program:

- `SttService<P>` is the only `FrameHandler` in the subsystem. It owns the WebSocket tasks, the [audio front end](#speech-enhancement-the-audio-front-end), the turn gate and billing.
- `TurnGate` forwards audio only during local-VAD-attested turns plus a 500 ms pre-roll (so the provider's own server VAD has no inter-turn noise to hallucinate from), stashes `VADUserStoppedSpeaking` until the provider's final arrives and releases the two as one frame, times out after 1.2 s if no final comes, and keeps a per-turn duration ledger that attributes each transcript to its turn and feeds billing.
- `SttCoreConfig` holds the provider-independent knobs: `sample_rate`, `noise_reduction`, `noise_backend`, `agc`, `audio_gating`, `pre_roll_ms`, `stop_release_timeout_ms`, `interim_policy` (`Drop` or `Emit` partials as non-final `TranscriptionFrame`s) and `max_turn_ms`.
- **`max_turn_ms`**: once a turn has sent that much audio, the provider is asked to finalise what it has heard at the next gap between words (80 ms of quiet, so no word is cut in two), or 2 s later if none comes. The text arrives mid-turn and the turn goes on. For live captions of speakers who rarely pause: a newsreader's turns run 15–25 s. At 8 s, captions came three times sooner for about 1.7 points more character error.
- **Reconnect.** When the provider closes the socket or a send fails, the next send reopens it (at most every 3 s, not while the session is ending), so one server error does not silence the rest of the session.
- Repeated identical partials are pushed once.

A provider implements `SttProvider`: `handshake()`, `encode_audio()`, `finalize_msg()`, `parse()`, plus optional `turn_start_msg`, `mid_turn_finalize_msg`, `on_connected`, `keepalive` and `close_msg`. All methods take `&self`; a provider holds configuration, never session state. Adding a provider is a protocol exercise, not a concurrency one. Sarvam's is the worked example, unit-tested without a socket.

| Provider | Handler | On the STT core | Notes |
|---|---|---|---|
| **Sarvam streaming** | `SarvamSttHandler` | ✅ | `saaras:v3`: transcribe, translate, verbatim, transliterate, codemix modes; `unknown` auto-detects among Indian languages; `prompt` for biasing; `vad_tuning` for Sarvam's server VAD. One final per turn. |
| **Sarvam realtime** | `SarvamRealtimeSttHandler` | ✅ | `saaras:v3-realtime`: **partials** while the speaker talks, then a final. `stream_type` `fast` / `balanced` / `simulated`; `endpointing` `manual` (local VAD drives `speech_start` / `speech_end`, the default) or `vad` (Sarvam's). The handler for captions. |
| **Deepgram** | `DeepgramSttHandler` | ❌ | `nova-3`, the low-latency choice for English and telephony. Finals only; `Finalize` sent on VAD stop. |
| **60db** | `SixtyDbSttHandler` | ❌ | 39 languages, two-phase finals (fast dictation then LLM-refined), diarization; linear or µ-law. |
| **Gnani (Vachana)** | `GnaniSttHandler` | ❌ | Indic languages; feature `stt-gnani`. |

Providers not yet on the core send raw audio at their configured rate, without the front end, the turn gate or the transcript-before-stop guarantee; their closing transcript may arrive after the stop frame, in which case `LLMUserAggregator` defers it to the next turn. Moving them onto the core is planned.

→ Per-service guides: [STT abstraction](doc/stt-abstraction.md) · [Sarvam](doc/stt-sarvam.md) · [Deepgram](doc/stt-deepgram.md) · [60db](doc/stt-60db.md) · [Gnani](doc/stt-gnani.md)

### Sentence segmentation

`SentenceSegmenter` sits after a realtime STT with partials on. Anything that waits for finals (translation, dubbing, an LLM) otherwise waits for the pause and the STT's finalisation, about two seconds after the last word. The segmenter finalises earlier: once a partial shows a sentence end followed by a few characters of the next sentence (so the boundary is unlikely to be revised), the sentence goes downstream as a final `TranscriptionFrame` of its own, the partial carries on with the remainder, and the segment's real final (standalone, or bundled on `VADUserStoppedSpeaking`) is stripped of what was already finalised, or dropped if nothing is left.

```text
partial "Good morning everyone. This"              →  final "Good morning everyone."  +  partial "This"
partial "Good morning everyone. This is a test."   →  partial "This is a test."
final   "Good morning everyone. This is a test."   →  final "This is a test."
```

Boundaries use the abbreviation-aware splitter (`Dr.`, initials, acronyms, Devanagari danda, ellipses). A hypothesis the STT revises in front of the committed text is taken up from the longest run of words it still shares, so nothing is finalised twice. `SegmenterConfig { min_tail_chars: 3, min_sentence_chars: 6 }`.

### Translation

- **`TranslationService`** is a stage after the STT that translates every *final* transcript (`finalized` frames and the turn-closing transcript bundled on `VADUserStoppedSpeaking`) into each target language, as `TranslationFrame`s. Interims are never translated. Every other frame passes through unchanged.
- One ordered queue per target language, off the pipeline's tasks: a slow call never delays the transcript or another language, and a language's translations come out in the order their transcripts did. Targets are a `TranslationTargets` handle that can change while the pipeline runs. A transcript already in the target language is passed through as its own translation, so a target is never silent. Long text is split at sentence ends to the provider's limit.
- `with_default_source_language` sets the source for STTs that report none per phrase; otherwise the provider detects it.
- **`TranslationProvider`** (`name`, `model`, `supports`, `max_chars`, `translate`) fits any text-to-text backend.
- **Sarvam Translate** (`sarvam-translate:v1` by default, or `mayura:v1`): English and 22 Indian languages, BCP-47 tags (`hi`, `en-IN`, `ml_IN`) mapped to Sarvam's codes. Sarvam detects the source language only on `mayura:v1`, so a transcript of unknown language goes to `auto_detect_model`. One retry on 429 or 5xx. *(feature `translate-sarvam`)*
- Billing: `BillingEvent::TranslationUsage` per call, by source characters.

→ [doc/translate-sarvam.md](doc/translate-sarvam.md)

### Dubbing

- **`DubbingService`** sits after the translation stage. It speaks every final translation whose target is among its `TranslationTargets` with a `TtsSynthesizer` and emits a `DubbedSpeechFrame`: the utterance's PCM plus the source transcript's `timestamp` and `source_text`, so a host can play it where the words were said. One ordered queue per language. `DubConfig` maps a voice per language and sets the sample rate asked of the provider; a `PaceHint` closure decides each utterance's pace (clamped to the provider's range).
- **Streaming.** With a synthesizer whose `streams()` is true, the frame goes out as soon as the first chunk of audio exists, carrying it, and the rest follows on `DubbedSpeechData::stream` (a `SpeechStream`: `next().await`, `finish`, `fail`). A host can start playing a few hundred milliseconds after the text instead of after the whole utterance.
- **`TtsSynthesizer`** is the one-utterance shape, as opposed to the streaming TTS handlers that speak an LLM's answer token by token: `synthesize(text, lang, voice, pace, sample_rate) -> Pcm`, and `synthesize_streaming -> PcmChunks`.
- **Sarvam Bulbul over REST** (`SarvamRestTts`, `bulbul:v3` by default): the eleven Bulbul languages, up to 48 kHz, three attempts on a rate limit. **Sarvam Bulbul streaming** (`SarvamStreamTts`): one WebSocket per (language, voice) kept open and reused, utterances one after another on it, 24 kHz. *(feature `tts-sarvam`)*
- Billing: `BillingEvent::TtsUsage` per utterance, with the language.

→ [doc/dubbing.md](doc/dubbing.md)

### Large language models

- **OpenAI-compatible** chat completions with SSE streaming: OpenAI, Azure, self-hosted. `OpenAILLMConfig` covers temperature, top-p, penalties, seed, `max_completion_tokens`, `service_tier`, `max_tool_rounds`.
- **Sarvam LLM** (`sarvam-m`, `sarvam-30b`, `sarvam-105b`) with optional chain-of-thought via `reasoning_effort`.
- Full function calling with a re-invocation loop (model calls tool, rustvani executes it, re-invokes), capped by `max_tool_rounds`. Tool rounds are staged and committed atomically, so an interrupted round never leaves an orphaned `tool_calls` message.
- Automatic context-window trimming from a per-model token table, overridable with `context_window_tokens`. Whole user turns are dropped, never half a tool round.
- Billing from the provider's `usage` chunk when it sends one; an interrupted stream is billed from an estimate rather than lost.
- Provider adapters: implement `LLMAdapter` to add a wire format.

→ [OpenAI](doc/llm-openai.md) · [Sarvam](doc/llm-sarvam.md)

### Text-to-speech

- **Deepgram Aura-2**: WebSocket streaming; interruption via `Clear` without reconnecting. *(default feature)*
- **Sarvam Bulbul** (`bulbul:v2` default, `v3-beta`, `v3`): WebSocket streaming with 25+ voices; pace, pitch, loudness and temperature where the model supports them. *(feature `tts-sarvam`)*
- **Piper**: fully local ONNX inference, zero network calls. espeak-ng phonemisation, three quality levels, a model shareable across pipelines via `Arc<Mutex<PiperModel>>`. US English voices. *(feature `tts-piper`)*
- Sentence-aware text buffering with abbreviation detection (`Mr.`, `Dr.`, `IPC.`), so the TTS gets whole sentences as the LLM streams.
- Indian numbering preprocessing for TTS (`100000` → `1,00,000`).
- TTS characters are billed per flush confirmation, including text already submitted when a barge-in clears the buffer.

→ [Sarvam](doc/tts-sarvam.md) · [Deepgram](doc/tts-deepgram.md) · [Piper (local)](doc/tts-piper.md)

### RAVI protocol

RAVI (**R**eal-time **A**udio **V**ideo **I**nterface) is rustvani's client protocol, the counterpart of Pipecat's RTVI: JSON messages with `"label": "ravi"` on the same WebSocket or data channel as the audio.

```rust
use rustvani::ravi::{RaviObserverParams, RaviParams, RaviProcessor};

let ravi = RaviProcessor::new(RaviParams { context: Some(context.clone()), ..Default::default() });
let observer = RaviProcessor::create_observer(&ravi, RaviObserverParams::default());

// Pipeline: [transport.input(), ravi, stt, …, transport.output()]
// task.run(system_clock(), Some(Arc::new(observer)))
```

`RaviProcessor` handles inbound messages: `client-ready` (answered with `bot-ready`), `disconnect-bot`, `send-text` (injects a typed user turn into the context and triggers the LLM) and custom `client-message`s, which become `RaviClientMessage` frames for your own processors. `RaviObserver` watches the pipeline and sends `user-started-speaking`, `user-transcription` (with `final` and the detected `language`), `user-translation`, `bot-llm-started`, `bot-llm-text`, `bot-tts-*`, `bot-started-speaking` and `function-call-*` events, each switchable in `RaviObserverParams`. Dhara handlers can push arbitrary `server-message`s with `DharaContext::push_ravi_message`.

### Telephony: Twilio Media Streams

Point a Twilio phone number at your rustvani server and you have a phone agent. The `FrameSerializer` layer sits between `WebSocketTransport` and the provider: outgoing frames become provider messages, incoming provider messages become frames. The rest of the pipeline never learns the caller is on a phone.

```rust
use rustvani::{TwilioFrameSerializer, TwilioInputParams, TwilioStart};

// `start` is the parsed Twilio `start` handshake, the first text message on the socket.
let serializer = TwilioFrameSerializer::from_start(
    start,
    twilio_auth_token,   // Option<String>: None disables REST hang-up
    TwilioInputParams { auto_hang_up: true, ..TwilioInputParams::default() },
)?;
transport.set_serializer(Box::new(serializer));   // before run_socket
```

- G.711 µ-law codec (`serializers::g711`, bit-exact with Python's `audioop`) with transparent 8 kHz ↔ pipeline-rate resampling
- Barge-in maps to Twilio's `clear` event, no reconnect; `dtmf` events become `InputDTMFFrame`s
- `auto_hang_up` ends the call over the Twilio REST API on `EndFrame` or `CancelFrame` *(feature `serializer-twilio`, on by default)*
- Set `audio_out_10ms_chunks: 2` to match Twilio's 20 ms media cadence

Working server: [`src/bin/twilio_coordinator_server.rs`](src/bin/twilio_coordinator_server.rs). → [doc/serializer-twilio.md](doc/serializer-twilio.md)

### WebRTC transport (peer-to-peer, no SFU)

`vaniwebrtc` carries audio over a real peer-to-peer WebRTC connection: Opus over RTP/SRTP with **no SFU or media server in the path**. Signaling (SDP offer/answer plus trickle ICE) runs over a WebSocket; control messages (interruption, RAVI, client VAD) ride a WebRTC data channel.

```rust
use rustvani::transport::{VaniWebRTCParams, VaniWebRTCTransport};

let transport = VaniWebRTCTransport::new("webrtc", VaniWebRTCParams {
    transport: TransportParams { /* … same params as any transport … */ },
    ..Default::default()
});
// transport.input(), transport.output(), transport.run(signaling_socket, push_tx)
```

- `TurnServer` configures TURN with credentials; `nat_1to1_ips` advertises a public IPv4 behind an edge.
- `build_shared_udp_mux(bind_addr)` binds one UDP socket at startup so every session shares one port, which is what lets a single container behind Fly's edge hold many calls.
- The answer SDP is munged to force high-bitrate, full-band Opus from the browser, so a 48 kHz denoiser (the `Denoiser48k` hook) is not starved of high frequencies.
- Outbound Opus packets are paced at 20 ms, so a barge-in can drop the unplayed tail instead of letting a whole utterance blast into the client's jitter buffer.

Opt-in (`vaniwebrtc`): pulls a large dependency tree and needs cmake plus a C compiler for libopus. Server: [`src/bin/vaniwebrtc_server.rs`](src/bin/vaniwebrtc_server.rs) · Browser client: [`examples/vaniwebrtc_client.html`](examples/vaniwebrtc_client.html) · → [doc/vaniwebrtc.md](doc/vaniwebrtc.md)

### Function calling and tools

Two halves: the schema the model sees goes on the context, and the handler that runs goes in a `FunctionRegistry`.

```rust
use rustvani::adapters::schemas::{FunctionSchema, ToolsSchema};
use rustvani::context::shared_context_with_tools;
use rustvani::services::llm::function_registry::ToolCallOutput;
use rustvani::{FunctionRegistry, OpenAILLMConfig, OpenAILLMHandler};
use serde_json::{json, Value};

// 1. What the model sees.
let get_weather = FunctionSchema::new("get_weather", "Current weather for a city")
    .with_parameters(json!({
        "type": "object",
        "properties": { "city": { "type": "string" } },
        "required": ["city"]
    }));
let context = shared_context_with_tools(
    Some("You are a weather assistant.".into()),
    ToolsSchema::new(vec![get_weather]),
    None, // tool_choice: provider default ("auto")
);

// 2. What runs when the model calls it.
let mut registry = FunctionRegistry::new();

// Simple: the string goes straight into the LLM context.
registry.register("get_weather", |args: String| async move {
    let city = serde_json::from_str::<Value>(&args)
        .ok()
        .and_then(|v| v["city"].as_str().map(str::to_owned))
        .unwrap_or_else(|| "unknown".into());
    format!("Weather in {city}: 28°C, partly cloudy")
});

// Data: a summary for the LLM, the full payload as a FunctionCallRawResultFrame for UI or logging.
registry.register_data("search_cases", |args: String| async move {
    let rows = vec![json!({ "id": 1, "query": args })]; // your database call
    ToolCallOutput::with_data(format!("Found {} cases", rows.len()), json!(rows))
});

let llm = OpenAILLMHandler::with_registry(config, registry).into_processor();
```

Built-in **Neon Postgres tool** (`db-postgres`): schema introspection cached at `StartFrame`, parameterised queries, pgvector similarity search, structured filters that the tool turns into SQL. The LLM never writes raw SQL. It registers `pg_schema`, `pg_query`, `pg_refine` and `pg_vector_search`:

```rust
let pg = Arc::new(NeonPostgresTool::from_env());   // reads DATABASE_URL, no connection yet
let mut llm = OpenAILLMHandler::new(config);
llm.add_tool(pg);                                  // handlers registered now, connects on StartFrame
let tools = ToolsSchema::new(llm.collect_tool_schemas()); // put this on the context
```

Any struct implementing `BuiltinTool` gets the same lifecycle: `on_start` with a cancellation token, `on_stop`, `on_cancel`.

→ [doc/tools-postgres.md](doc/tools-postgres.md)

### Dhara: conversation flows

> *dhara* (ധാര) — flow, stream

A conversation as a graph of nodes. Each node owns its system prompt (or override), task messages, context strategy (`append` or `reset`) and the functions the model may call; each function declares which node a given handler status leads to. The graph lives in `dhara.json`, the handlers in Rust, and the loader validates the two against each other: unknown functions, unreachable nodes, dangling transitions and missing handlers are errors at load, not surprises mid-call.

```rust
use rustvani::dhara::{Dhara, DharaFunctionRegistry, HandlerResult};

// Handlers get the LLM's JSON arguments and a DharaContext (your state, the pipeline, RAVI).
let mut handlers = DharaFunctionRegistry::new();
handlers.register("place_order", |args: String, ctx| async move {
    let order = ctx.state::<std::sync::Mutex<OrderState>>().unwrap();
    // … validate, persist …
    ctx.push_ravi_message(serde_json::json!({ "type": "order_placed" })).await;
    HandlerResult::ok(r#"{"status":"placed"}"#)          // transitions["default"]
    // HandlerResult::with_status(result, "out_of_stock")  // transitions["out_of_stock"]
});

let dhara = Dhara::load("dhara/pizza_order")?;             // reads and validates dhara.json
let built = dhara.build(&handlers, state, conn_id)?;        // per connection

let llm = OpenAILLMHandler::with_shared_registry(config, built.llm_registry);
llm.set_transition_hook(built.hook);
// built.context → the aggregators and RaviParams
// after PipelineTask::new(): built.dhara_ctx.set_push_sender(task.push_sender());
```

On a transition the hook applies the new node's prompt and task messages, swaps the LLM's tool set and registry, and re-invokes the model. [`dhara/pizza_order/`](dhara/pizza_order/) is a complete flow (greeting → menu → confirm → farewell) run by [`src/bin/pizza_voice_server_v2.rs`](src/bin/pizza_voice_server_v2.rs). The code-first `DharaManager` / `NodeConfig` / `TransitionResult` API from earlier releases still works.

### Agent swarm: multi-agent coordination

For workloads a single pipeline should not own: a voice agent that hands research to a background worker, or several specialists behind one caller. Each agent owns its own `PipelineTask`, they communicate over an `AgentBus`, and an `AgentRunner` orchestrates them. No global state.

```rust
use rustvani::agents::{AgentRunner, BaseAgent, CoordinatorProcessor, LocalAgentBus, TaskContext};
```

- **`LocalAgentBus`**: two-priority fan-out. System messages (End, Cancel, Activate, urgent replies, registry) are never dropped and always overtake queued data; data messages are dropped and counted rather than blocking when a subscriber is full. Control never drops, data never blocks. Messages fan out as one `Arc`, and `send()` takes no lock.
- **`BaseAgent`**: the task router. Register job handlers with `on_task(name, handler)`; each runs in its own Tokio task with a `TaskRequestCtx` for `complete`, `stream_start`, `stream_data` and `stream_end`. End cascades to child agents and waits for them; cancellation of a running job always sends the requester a terminal `Cancelled`.
- **`TaskContext::dispatch`** is ready-gated: it waits (watch-based, no polling) for the target to appear in the registry instead of dropping the request. `dispatch_fenced` stamps a turn epoch so a late reply for a superseded turn is quarantined (`AgenticCoordinator`).
- **`CoordinatorProcessor`**: a frame processor that sits in the LLM's slot and answers from a closure that can `cx.call(agent, task, payload)` peers over the bus, with or without a swarm behind it. **`BusOutputEdge`** republishes a pipeline's frames to peer agents, so one agent's output becomes another's input.

→ Full guide: [agents.md](agents.md) · demos: [`examples/bridged_agents.rs`](examples/bridged_agents.rs), [`src/bin/agent_swarm_mixed.rs`](src/bin/agent_swarm_mixed.rs), [`src/bin/agent_swarm_coordinator/`](src/bin/agent_swarm_coordinator/)

### Billing and usage tracking

Non-blocking billing that captures what you need to cost and invoice a session, all linked by `session_id` and written to PostgreSQL or structured JSON logs.

| Signal | Source | Accuracy |
|---|---|---|
| Session duration | Pipeline start and finish hooks | Exact |
| LLM input and output tokens | OpenAI `stream_options.include_usage` | Exact; estimated (and flagged) when a stream is interrupted or the provider sends no usage |
| TTS characters synthesised | Per flush confirmation (Deepgram, Sarvam); per utterance with the language (dubbing) | Exact |
| STT audio duration | Provider-reported per transcript, or the turn gate's ledger | Exact / computed |
| Translation characters | Per call, source text length | Exact |
| Conversation transcript | User and assistant turns, with `interrupted` for cut-off replies | |

`record()` is one send onto an unbounded channel drained by a background task: it never blocks and never drops, so billing overhead is invisible to audio latency. The drain task writes a **durable checkpoint about once a minute** (an absolute snapshot plus the ledger events since the last one, each with a stable id so a retry never double-inserts) and a final one at session end. A crash therefore costs at most one checkpoint interval, and `billing_sweeper` settles sessions whose process died mid-call.

```rust
let (billing, drain) = SessionBilling::new(session_id, storage, 256);

let stt = SarvamSttHandler::new(stt_cfg).with_billing(billing.clone()).into_processor();
let llm = OpenAILLMHandler::new(llm_cfg).with_billing(billing.clone()).into_processor();
let user_agg = LLMUserAggregator::with_billing(context.clone(), billing.clone(), user_turn_id.clone());
// … PipelineParams { billing_collector: Some(billing.clone()), billing_metadata, .. }
// At the end: drop every clone, then `drain.await` flushes the final checkpoint.
```

Storage backends: `LogBillingStorage` (JSON log lines, no dependencies) and `PostgresBillingStorage` (`db-postgres`; `run_migrations` creates `billing_sessions` and `billing_events`).

→ Full guide, schemas, cost queries, transcript capture: [doc/billing.md](doc/billing.md)

### Audio capture: session recording

Records the conversation as two time-aligned tracks, user and bot, each padded with silence while the other speaks (except during a real barge-in), then mixes them into one `recording.wav` per session at 16 kHz mono: the conversation as you would have heard it in the room. Turn ids are shared with the aggregators so transcript entries line up with the recording.

```rust
let (audio_cap, _drain) = SessionAudioCapture::new(session_id, Arc::new(LocalAudioStorage::new("recordings")), 64);
let audio_proc = AudioCaptureProcessor::new(audio_cap, user_turn_id, bot_turn_id);
// place after TTS, before transport.output()
```

Storage: `LocalAudioStorage` writes `{base_dir}/{session_id}/recording.wav`; `PostgresAudioMetaStorage` (`db-postgres`) stores segment metadata. `SessionBilling::new_with_dir` puts `transcript.json` in the same folder.

→ [doc/audio-capture.md](doc/audio-capture.md)

### Testing without a server

`ChannelTransport` drives a full pipeline through plain `mpsc` channels: feed PCM in, assert on what comes out, no WebSocket server needed.

```rust
let (incoming_tx, incoming_rx) = mpsc::channel(64);
let (outgoing_tx, mut outgoing_rx) = mpsc::channel(64);

let transport = ChannelTransport::new("test", params, incoming_rx);
let task = PipelineTask::new(vec![transport.input(), /* … */, transport.output()], PipelineParams::default());
let push_tx = task.push_sender();
tokio::spawn(async move { transport.run(push_tx, outgoing_tx).await });

incoming_tx.send(ChannelMessage::Audio(pcm_bytes)).await?;                         // at the configured rate
incoming_tx.send(ChannelMessage::AudioFrame { audio, sample_rate: 48_000, channels: 2 }).await?; // any format
let out = outgoing_rx.recv().await;  // ChannelMessage::Audio / Text (RAVI) / Interruption
```

Runnable version: [`examples/channel_pipeline.rs`](examples/channel_pipeline.rs) · [doc/transport.md](doc/transport.md)

```bash
cargo test                                   # unit + integration tests, default features
cargo test --lib --no-default-features \
  --features transport-websocket,stt-sarvam,translate-sarvam,tts-sarvam   # a trimmed library build
SARVAM_API_KEY=… cargo test --features tts-sarvam sarvam_ -- --ignored --nocapture   # live provider tests
```

The live tests (`sarvam_translate_live`, `sarvam_rest_tts_live`, `sarvam_stream_tts_live`) are `#[ignore]`d and hit the real API. The binaries in `src/bin/` need the default feature set, so test the library with `--lib` when trimming features.

### Observability

`BaseObserver` gets a callback for every frame processed and every frame pushed, with processor name, direction and timestamp: enough to build per-turn latency breakdowns (VAD stop → transcript → first LLM token → first TTS chunk) without touching pipeline code. Pass it to `task.run(clock, Some(observer))`. See the `LatencyObserver` in [`src/bin/websocket_server.rs`](src/bin/websocket_server.rs). `PipelineTask` also exposes lifecycle callbacks, an idle timeout with optional auto-cancel, heartbeats, and a `watch` channel of `PipelineLifecycle`. Everything logs through `log`; set `RUST_LOG=rustvani=debug` to see raw provider messages.

---

## Examples, binaries and tests

| | What it shows | Features beyond default |
|---|---|---|
| [`examples/quickstart.rs`](examples/quickstart.rs) | The README voice agent, verbatim | |
| [`examples/channel_pipeline.rs`](examples/channel_pipeline.rs) | A full pipeline driven from a WAV file through `ChannelTransport` | |
| [`examples/echo_pipeline.rs`](examples/echo_pipeline.rs) | Fake STT/LLM/TTS processors: the frame lifecycle with no network | |
| [`examples/bridged_agents.rs`](examples/bridged_agents.rs) | Two bridged agents: dispatch, a streamed reply, cancellation | |
| [`examples/gnani_pipeline.rs`](examples/gnani_pipeline.rs), [`gnani_stt_test.rs`](examples/gnani_stt_test.rs) | Gnani STT end to end and standalone | `stt-gnani` |
| [`examples/sixtydb_stt.rs`](examples/sixtydb_stt.rs) | 60db STT from a WAV file | |
| [`examples/vaniwebrtc_client.html`](examples/vaniwebrtc_client.html) | Browser client for the WebRTC server, with client-side VAD | |
| [`src/bin/websocket_server.rs`](src/bin/websocket_server.rs) | Sarvam STT → OpenAI → Sarvam TTS over WebSocket, with a latency observer | `tts-sarvam` |
| [`src/bin/websocket_server_piper.rs`](src/bin/websocket_server_piper.rs) | The same with local Piper TTS | `tts-piper` |
| [`src/bin/ravi_server.rs`](src/bin/ravi_server.rs) | RAVI protocol plus a `get_weather` tool | `tts-sarvam` |
| [`src/bin/vaniwebrtc_server.rs`](src/bin/vaniwebrtc_server.rs) | Peer-to-peer WebRTC voice agent | `vaniwebrtc`, `tts-sarvam` |
| [`src/bin/twilio_coordinator_server.rs`](src/bin/twilio_coordinator_server.rs) | A phone agent on Twilio Media Streams with a `CoordinatorProcessor` brain | |
| [`src/bin/pizza_voice_server_v2.rs`](src/bin/pizza_voice_server_v2.rs), [`pizza_voice_server.rs`](src/bin/pizza_voice_server.rs) | Dhara ordering flow backed by Neon Postgres (JSON API, and the legacy API) | |
| [`src/bin/interview_voice_server.rs`](src/bin/interview_voice_server.rs) | A structured interview flow | |
| [`src/bin/agent_text_demo.rs`](src/bin/agent_text_demo.rs), [`agent_swarm_mixed.rs`](src/bin/agent_swarm_mixed.rs), [`agent_swarm_coordinator/`](src/bin/agent_swarm_coordinator/) | Agent swarms: two LLM agents, plain-function agents, an inline coordinator | |
| [`src/bin/full_pipeline_test.rs`](src/bin/full_pipeline_test.rs) | Billing, transcript and recording wiring end to end | |
| [`src/bin/billing_sweeper.rs`](src/bin/billing_sweeper.rs), [`db_verify.rs`](src/bin/db_verify.rs) | Settle crashed sessions; check what reached Postgres | |
| [`tests/`](tests/) | `ChannelTransport` round trip, client/server VAD exactly-once, VAD at 48 kHz stereo input | |

Run a binary with `cargo run --bin <name> --features <…>`; each file's header lists its environment variables.

---

## Documentation

Every service and component has a guide in [doc/](doc/README.md) with exact config fields, environment variables and feature flags:

| Area | Guides |
|---|---|
| Overview | [Architecture](architecture.md) · [Agents](agents.md) · [Turn-level ACID](doc/turn-acid.md) · [Phase 2](doc/turn-acid-phase2.md) |
| Audio front end | [Speech enhancement](doc/audio-enhancement.md) · [VAD](doc/vad.md) |
| STT | [STT abstraction](doc/stt-abstraction.md) · [Sarvam](doc/stt-sarvam.md) · [Deepgram](doc/stt-deepgram.md) · [60db](doc/stt-60db.md) · [Gnani](doc/stt-gnani.md) |
| Translation and dubbing | [Sarvam Translate](doc/translate-sarvam.md) · [DubbingService and Sarvam Bulbul](doc/dubbing.md) |
| LLM | [OpenAI](doc/llm-openai.md) · [Sarvam](doc/llm-sarvam.md) |
| TTS | [Sarvam](doc/tts-sarvam.md) · [Deepgram](doc/tts-deepgram.md) · [Piper (local)](doc/tts-piper.md) |
| Transport | [WebSocket and Channel](doc/transport.md) · [WebRTC](doc/vaniwebrtc.md) · [Twilio serializer](doc/serializer-twilio.md) |
| Tools | [Postgres tool](doc/tools-postgres.md) |
| Observability | [Billing](doc/billing.md) · [Audio capture](doc/audio-capture.md) |
| Changes | [CHANGELOG](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) |

---

## For Pipecat developers

If you know Pipecat, you already know rustvani. The mapping is close to 1:1:

| Pipecat (Python) | rustvani (Rust) |
|---|---|
| `FrameProcessor` | `FrameProcessor` + your `FrameHandler` |
| `Frame` subclasses | `Frame { inner: FrameInner }` enum |
| `Pipeline(processors)` / `PipelineTask` | `PipelineTask::new(processors, params)` |
| `OpenAILLMService` | `OpenAILLMHandler` |
| `LLMUserResponseAggregator` | `LLMUserAggregator` |
| `LLMAssistantResponseAggregator` | `LLMAssistantAggregator` |
| `SileroVADAnalyzer` | `SileroVadNative` / `SileroVadOrt` |
| `SmartTurnAnalyzer` | `SmartTurnAnalyzer` |
| `FunctionCallHandler` / `FunctionSchema` / `ToolsSchema` | `FunctionRegistry` / `FunctionSchema` / `ToolsSchema` |
| `FlowManager` (pipecat-flows) | `Dhara` + `DharaFunctionRegistry` |
| `RTVIProcessor` / `RTVIObserver` | `RaviProcessor` / `RaviObserver` |
| `TwilioFrameSerializer` | `TwilioFrameSerializer` |
| `AudioBufferProcessor` | `AudioCaptureProcessor` |
| `BaseWorker` / agent bus | `BaseAgent` / `AgentBus` |
| `@transport.event_handler("on_client_connected")` | `task.add_on_pipeline_started(...)` |
| `isinstance(frame, VADUserStartedSpeakingFrame)` | `matches!(frame.inner, FrameInner::System(SystemFrame::VADUserStartedSpeaking { .. }))` |
| *(no equivalent)* | STT core with `TurnGate`, `SentenceSegmenter`, `TranslationService`, `DubbingService`, the built-in denoisers |

The frame flow, interrupt semantics, aggregator logic and pipeline nesting work identically. If you have debugged a Pipecat bot, you can debug a rustvani bot.

---

## Project status

rustvani is in active development on the `0.4.0-dev` line. The pipeline core, the frame system and every service listed above are functional and in production.

**Working:**
- Full pipeline lifecycle: start, interruption, stop, cancel, end; idle timeout; heartbeats
- Silero VAD, native Rust and ONNX, from any input rate or channel count; SmartTurn end-of-turn detection, pure Rust
- Client and server VAD coordination
- Speech enhancement chain (high-pass → RNNoise or hush-vani → AGC → soft limiter) and streaming resampling
- STT core: turn gate, pre-roll, transcript-before-stop ordering, live partials, `max_turn_ms`, duration ledger, reconnect
- Sarvam STT (streaming and realtime), Deepgram STT, 60db STT, Gnani STT
- Sentence segmentation from partial transcripts
- Translation stage with Sarvam Translate; dubbing stage with Sarvam Bulbul over REST and streaming WebSocket
- OpenAI-compatible LLM with function calling and a re-invocation loop; Sarvam LLM
- Deepgram Aura-2, Sarvam Bulbul and local Piper TTS
- RAVI protocol; WebSocket, peer-to-peer WebRTC and channel transports; Twilio Media Streams with G.711
- Dhara flows from JSON; agent swarm with bus, registry, runner, task routing, fenced dispatch
- Neon Postgres tool with pgvector; billing with checkpoints and a crash sweeper; session recording
- Published on [crates.io](https://crates.io/crates/rustvani)

**Known limits:**
- Deepgram, 60db and Gnani are not yet on the STT core: no front end, no turn gate, no partials
- `SileroVadNative` is 16 kHz only (use `SileroVadOrt` for an 8 kHz model); Piper ships US English voices
- The Sarvam LLM handler streams text but does not run the tool loop; use the OpenAI-compatible handler for tools
- The `metrics` module is a placeholder: TTFB and token metrics flow through billing and observers instead

**Planned:**
- Deepgram, 60db and Gnani onto the STT core
- Anthropic and Gemini LLM adapters (only the OpenAI wire format ships today)
- Whisper STT; ElevenLabs and PlayHT TTS
- A full-band (48 kHz) denoiser behind the WebRTC `Denoiser48k` hook

Contributions are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

rustvani is released under BSD-2-Clause. See [LICENSE](LICENSE).

Portions of this project are derived from [Pipecat](https://github.com/pipecat-ai/pipecat) by Daily and retain Pipecat's BSD-2-Clause license notice. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## Acknowledgements

rustvani would not exist without [Pipecat](https://github.com/pipecat-ai/pipecat) by Daily. The architecture, frame taxonomy, aggregator patterns and pipeline design are derived from their work.

Built with [Sarvam AI](https://www.sarvam.ai/) for Indian-language voice: STT, translation, TTS and LLM services that work for Malayalam, Hindi and the other Indian languages.
