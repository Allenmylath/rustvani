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
         coordinated VAD            end-to-end
```

---

## Install

```toml
[dependencies]
rustvani = "0.1.6"
```

```bash
cargo add rustvani
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

**Client + Server VAD coordination.** rustvani is designed for deep Dioxus frontend integration. The browser client runs its own lightweight VAD and sends `ClientVADUserStartedSpeaking` events directly into the server pipeline. A toggle-switch CAS gate ensures exactly one `VADUserStartedSpeaking` is emitted regardless of which side fires first — no double-triggers, no race conditions. Pipecat has no equivalent.

**SmartTurn end-of-turn prediction.** A local ONNX model predicts whether the user has finished speaking before emitting a stop event. This eliminates false stops on hesitation pauses without adding network round-trips.

**Dhara conversation flow engine.** Node-based state machine where each node owns its own system prompt, tool set, and context strategy. Handlers return `Stay` or `Transition { next_node }` — full multi-turn flow control without orchestration boilerplate.

**Zero-dependency VAD.** The native Silero backend is pure Rust — no ONNX Runtime, no dynamic libraries, no `.so` files to bundle. One binary, everything included.

**Production-tested.** Deployed for a Kerala government voice agent serving real users across Malayalam, Hindi, and English.

---

## Quick Start

```rust
use rustvani::*;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    // 1. Shared conversation context
    let context = shared_context(Some("You are a helpful voice assistant.".into()));

    // 2. VAD — pure Rust, zero external deps
    let vad = SileroVadNative::new(16_000).expect("VAD load failed");

    // 3. Services
    let stt = SarvamSttHandler::new(SarvamSttConfig {
        api_key: std::env::var("SARVAM_API_KEY").unwrap(),
        ..Default::default()
    }).into_processor();

    let llm = OpenAILLMHandler::new(OpenAILLMConfig {
        api_key:  std::env::var("OPENAI_API_KEY").unwrap(),
        model:    "gpt-4.1".into(),
        context:  context.clone(),
        ..Default::default()
    }).into_processor();

    let tts = SarvamTtsHandler::new(SarvamTtsConfig {
        api_key: std::env::var("SARVAM_API_KEY").unwrap(),
        model:   "bulbul:v2".into(),
        ..Default::default()
    }).unwrap().into_processor();

    // 4. Transport (WebSocket via axum)
    let transport = BaseTransport::new(TransportParams {
        audio_in_enabled:     true,
        audio_in_sample_rate: Some(16_000),
        vad_analyzer:         Some(Arc::new(vad)),
        audio_out_enabled:    true,
        ..Default::default()
    });

    // 5. Aggregators bridge VAD ↔ LLM
    let user_agg      = LLMUserAggregator::new(context.clone()).into_processor();
    let assistant_agg = LLMAssistantAggregator::new(context.clone()).into_processor();

    // 6. Assemble and run
    let task = PipelineTask::new(
        vec![transport.input(), stt, user_agg, llm, assistant_agg, tts, transport.output()],
        PipelineParams { allow_interruptions: true, ..Default::default() },
    );

    task.run(system_clock(), None).await.unwrap();
}
```

---

## Deploy in 5 Minutes

### Docker (single static binary)

```dockerfile
FROM rust:1.82-slim AS builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
# Only needed for Piper TTS (local). Remove if using Sarvam TTS.
RUN apt-get update && apt-get install -y ca-certificates espeak-ng \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/your-bot /usr/local/bin/
WORKDIR /app
CMD ["your-bot"]
```

No Python, no virtualenv, no `requirements.txt`. The image is ~50 MB total.

### Environment variables

```bash
SARVAM_API_KEY=your_key
OPENAI_API_KEY=your_key   # or any OpenAI-compatible endpoint
DATABASE_URL=postgres://…  # if using the Postgres built-in tool
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
fly secrets set SARVAM_API_KEY=… OPENAI_API_KEY=…
fly deploy
```

Your voice agent is live. Zero idle cost when no users are connected.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  PipelineTask                                                    │
│                                                                  │
│  [TaskSource] → Transport.Input → STT → UserAgg →              │
│                 LLM → AssistantAgg → TTS → Transport.Output →   │
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

---

## Modules

```
src/
├── adapters/          LLM provider adapters (OpenAI wire format)
│   └── schemas/       Provider-agnostic tool/function schemas
├── audio_process/     Noise suppression (RNNoise) + resampling (rubato)
├── context/           Shared LLMContext (messages, tools, tool_choice)
├── dhara/             Conversation flow engine (node-based state machine)
├── frames/            Frame types, FrameProcessor, priority queues
├── pipeline/          Pipeline assembly + PipelineTask lifecycle
├── processors/        LLM user/assistant aggregators
├── ravi/              RAVI protocol (real-time audio/video interface)
├── services/
│   ├── llm/           OpenAI + Sarvam LLM (SSE streaming, function calling)
│   ├── stt/           Sarvam STT (WebSocket streaming)
│   └── tts/           Sarvam TTS (WebSocket) + Piper TTS (local ONNX)
├── tools/             Built-in tools (Neon Postgres with pgvector)
├── transport/         WebSocket transport (axum) + base I/O + ChannelTransport
├── utils/             Sentence splitter, text preprocessor
└── vad/               Silero VAD (native Rust + ONNX) + state machine
```

---

## Features

### Voice Activity Detection

Two backends, same API:

```rust
// Pure Rust — zero ONNX Runtime dependency, 16kHz only
let vad = SileroVadNative::new(16_000)?;

// ONNX Runtime — 8kHz + 16kHz, same model as Pipecat
let vad = SileroVadOrt::new(VadBackend::Silero16k)?;
```

- 4-state machine: `Quiet → Starting → Speaking → Stopping → Quiet`
- Configurable confidence threshold, start/stop durations, minimum volume
- Volume calculation using dBFS approximation of EBU R128
- Inference runs on `spawn_blocking` — never stalls the Tokio executor
- **SmartTurn**: optional local ONNX end-of-turn model defers stop events on hesitation pauses

### Client + Server VAD Coordination (Dioxus Integration)

rustvani's flagship differentiator: the browser/Dioxus client runs its own lightweight VAD and pushes events directly into the server pipeline. A shared atomic toggle ensures exactly one `VADUserStartedSpeaking` is emitted per utterance regardless of which side detects speech first.

```rust
// Called from your WebSocket handler when the Dioxus client reports speech
transport.push_client_vad_started(&processor, timestamp).await;
transport.push_client_vad_stopped(&processor, timestamp).await;
```

The coordination rule: `emitted_speaking` is an `AtomicBool` shared between client and server paths. The first source to win `compare_exchange(false, true)` emits the event; the second is a no-op. This eliminates double-triggers with zero locking overhead.

### Speech-to-Text
- **Sarvam AI** streaming WebSocket STT (`saaras:v3`)
- Supports transcription, translation, verbatim, transliteration, and codemix modes
- Multi-language: `ml-IN`, `hi-IN`, `en-IN`, auto-detect (`unknown`)
- Integrated **RNNoise** noise suppression (pure Rust via `nnnoiseless`)
- Transparent resampling if source rate ≠ 48 kHz (via `rubato`)

### Large Language Models
- **OpenAI-compatible** API with SSE streaming
- **Sarvam LLM** (`sarvam-m`, `sarvam-30b`) with optional CoT thinking mode
- Full function calling with re-invocation loop (model calls tool → execute → re-invoke)
- Configurable max tool rounds to prevent infinite loops
- Provider adapter system — add new providers by implementing `LLMAdapter`

### Text-to-Speech
- **Sarvam Bulbul** TTS (v2, v3-beta, v3) — WebSocket streaming with 25+ voices
- **Piper TTS** — fully local ONNX inference, zero network calls
  - espeak-ng phonemization → Piper ONNX → chunked PCM streaming
  - Multiple quality levels (Low/Medium/High)
  - Shared model across pipeline instances via `Arc<Mutex<PiperModel>>`
- Sentence-aware text buffering with abbreviation detection (Mr., Dr., IPC., etc.)
- Indian numbering system preprocessing for TTS (10000 → "ten thousand")

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

### Piper TTS (Local, Zero Network Calls)

```rust
let tts = PiperTtsHandler::new(PiperTtsConfig {
    quality:   PiperQuality::Medium,  // ~60 MB, 150–300ms/sentence
    model_dir: "./piper-models".into(),
    ..Default::default()
})?.into_processor();

// Share one model across multiple concurrent sessions
let shared = tts_handler.shared_model();
let tts2   = PiperTtsHandler::with_shared_model(config, shared).into_processor();
```

Requires `espeak-ng` for phonemization (`apt install espeak-ng`).

### Audio Processing

```rust
// RNNoise noise suppression — pure Rust, auto-resamples 16k ↔ 48k
let mut nf = RNNoiseFilter::new(16_000);
let clean  = nf.filter(&noisy_pcm_i16);
let tail   = nf.flush();  // drain at end of utterance
nf.reset();               // clean slate for next utterance
```

---

## Testing with ChannelTransport

`ChannelTransport` lets you drive a full pipeline in tests without a WebSocket server:

```rust
use rustvani::transport::{ChannelTransport, ChannelMessage, TransportParams};
use rustvani::pipeline::{PipelineTask, PipelineParams};
use rustvani::clock::system_clock;
use tokio::sync::mpsc;

#[tokio::test]
async fn test_my_pipeline() {
    let (incoming_tx, incoming_rx) = mpsc::channel::<ChannelMessage>(10);
    let (outgoing_tx, mut outgoing_rx) = mpsc::channel::<ChannelMessage>(10);

    let transport = ChannelTransport::new("test", TransportParams {
        audio_in_enabled: true,
        audio_in_sample_rate: Some(16_000),
        ..Default::default()
    }, incoming_rx);

    let task = PipelineTask::new(
        vec![transport.input(), /* your processors */, transport.output()],
        PipelineParams::default(),
    );
    let push_tx = task.push_sender();

    tokio::spawn(async move { task.run(system_clock(), None).await.ok(); });
    tokio::spawn(async move { transport.run(push_tx, outgoing_tx).await; });

    incoming_tx.send(ChannelMessage::Audio(pcm_bytes)).await.unwrap();
    let result = outgoing_rx.recv().await.unwrap();
}
```

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
| `FunctionCallHandler` | `FunctionRegistry` |
| `FlowManager` | `DharaManager` |
| `RTVIProcessor` | `RaviProcessor` |
| `@transport.event_handler("on_client_connected")` | `task.add_on_pipeline_started(...)` |
| `isinstance(frame, VADUserStartedSpeakingFrame)` | `matches!(frame.inner, FrameInner::System(SystemFrame::VADUserStartedSpeaking { .. }))` |

The frame flow, interrupt semantics, aggregator logic, and pipeline nesting all work identically. If you've debugged a Pipecat bot, you can debug a rustvani bot.

---

## Project Status

rustvani is in active development. Core pipeline, frame system, and all listed services are functional and battle-tested in production for a Kerala government voice agent deployment.

**Working:**
- Full pipeline lifecycle (start, interruption, cancel, end)
- Silero VAD — native Rust + ONNX — with SmartTurn end-of-turn prediction
- Client + Server VAD coordination (Dioxus frontend integration)
- Sarvam STT / TTS / LLM
- OpenAI-compatible LLM with function calling + re-invocation loop
- Piper TTS (local ONNX, zero network)
- Dhara conversation flow manager
- RAVI protocol
- Neon Postgres tool with pgvector
- WebSocket transport (axum) + ChannelTransport (testing)
- RNNoise noise suppression + audio resampling
- Available on [crates.io](https://crates.io/crates/rustvani)

**Planned:**
- Anthropic / Gemini LLM adapters
- WebRTC transport
- Deepgram / Whisper STT
- ElevenLabs / PlayHT TTS
- Metrics and observability

---

## License

Rustvani is released under BSD-2-Clause. See [LICENSE](LICENSE).

Portions of this project are derived from [Pipecat](https://github.com/pipecat-ai/pipecat) by Daily and retain Pipecat's BSD-2-Clause license notice. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## Acknowledgements

rustvani wouldn't exist without [Pipecat](https://github.com/pipecat-ai/pipecat) by Daily. The architecture, frame taxonomy, aggregator patterns, and pipeline design are all derived from their excellent work.

Built with [Sarvam AI](https://www.sarvam.ai/) for Indian language voice — STT, TTS, and LLM services that actually work for Malayalam, Hindi, and 10+ Indian languages.
