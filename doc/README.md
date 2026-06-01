# rustvani Service Documentation

This folder contains detailed usage documentation for every service and component in the rustvani voice pipeline framework.

## Services

### Speech-to-Text (STT)
| Service | File | Protocol | Best For |
|---|---|---|---|
| [Sarvam STT](stt-sarvam.md) | `src/services/stt/sarvam.rs` | WebSocket | Indian languages, auto-detect |
| [60db STT](stt-60db.md) | `src/services/stt/sixtydb.rs` | WebSocket | 39 languages, two-phase finals, diarization |
| [Gnani STT](stt-gnani.md) | `src/services/stt/gnani.rs` | WebSocket | Indic languages (Vachana API) |

### Text-to-Speech (TTS)
| Service | File | Protocol | Best For |
|---|---|---|---|
| [Sarvam TTS](tts-sarvam.md) | `src/services/tts/sarvam.rs` | WebSocket | Indian languages, 25+ voices |
| [Deepgram TTS](tts-deepgram.md) | `src/services/tts/deepgram.rs` | WebSocket | Aura voices, low latency |
| [Piper TTS](tts-piper.md) | `src/services/tts/piper.rs` | Local ONNX | Zero network, fully local |

### Large Language Models (LLM)
| Service | File | Protocol | Best For |
|---|---|---|---|
| [OpenAI LLM](llm-openai.md) | `src/services/llm/openai.rs` | SSE HTTP | Function calling, tool loops, any OpenAI-compatible endpoint |
| [Sarvam LLM](llm-sarvam.md) | `src/services/llm/sarvam.rs` | SSE HTTP | Indian language LLM, CoT thinking mode |

## Infrastructure

| Component | Documentation | Purpose |
|---|---|---|
| [VAD](vad.md) | `src/vad/` | Voice Activity Detection (Silero native + ONNX) |
| [Transport](transport.md) | `src/transport/` | WebSocket I/O, ChannelTransport for testing |
| [Postgres Tool](tools-postgres.md) | `src/tools/postgres/` | Built-in LLM tool for Neon Postgres + pgvector |

## Quick Pipeline Assembly

All services follow the same pattern:

```rust
use rustvani::*;

let stt = SomeSttHandler::new(config).into_processor();
let llm = SomeLlmHandler::new(config).into_processor();
let tts = SomeTtsHandler::new(config).unwrap().into_processor();

let task = PipelineTask::new(
    vec![transport.input(), stt, user_agg, llm, assistant_agg, tts, transport.output()],
    PipelineParams { allow_interruptions: true, ..Default::default() },
);
```

See individual service pages for exact config fields, feature flags, and environment variables.
