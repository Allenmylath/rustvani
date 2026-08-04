# rustvani Service Documentation

This folder contains detailed usage documentation for every service and component in the rustvani voice pipeline framework.

Broader design docs live at the repository root: [architecture.md](../architecture.md) (frames, processors, pipeline internals) and [agents.md](../agents.md) (multi-agent bus, registry, runner).

## Services

### Speech-to-Text (STT)
| Service | File | Feature | Protocol | Best For |
|---|---|---|---|---|
| [Sarvam STT](stt-sarvam.md) | `src/services/stt/sarvam.rs` | `stt-sarvam` ✅ | WebSocket | Indian languages, auto-detect |
| [Deepgram STT](stt-deepgram.md) | `src/services/stt/deepgram.rs` | `stt-deepgram` ✅ | WebSocket | English, telephony, low latency |
| [60db STT](stt-60db.md) | `src/services/stt/sixtydb.rs` | `stt-60db` ✅ | WebSocket | 39 languages, two-phase finals, diarization |
| [Gnani STT](stt-gnani.md) | `src/services/stt/gnani.rs` | `stt-gnani` ❌ | WebSocket | Indic languages (Vachana API) |

### Text-to-Speech (TTS)
| Service | File | Feature | Protocol | Best For |
|---|---|---|---|---|
| [Deepgram TTS](tts-deepgram.md) | `src/services/tts/deepgram.rs` | `tts-deepgram` ✅ | WebSocket | Aura-2 voices, low latency |
| [Sarvam TTS](tts-sarvam.md) | `src/services/tts/sarvam.rs` | `tts-sarvam` ❌ | WebSocket | Indian languages, 25+ voices |
| [Piper TTS](tts-piper.md) | `src/services/tts/piper.rs` | `tts-piper` ❌ | Local ONNX | Zero network, fully local |

### Large Language Models (LLM)
| Service | File | Feature | Protocol | Best For |
|---|---|---|---|---|
| [OpenAI LLM](llm-openai.md) | `src/services/llm/openai.rs` | `llm-openai` ✅ | SSE HTTP | Function calling, tool loops, any OpenAI-compatible endpoint |
| [Sarvam LLM](llm-sarvam.md) | `src/services/llm/sarvam.rs` | `stt-sarvam` ✅ | SSE HTTP | Indian language LLM, CoT thinking mode |

✅ = on by default · ❌ = opt-in. See the [feature flags table](../README.md#feature-flags) for the full list.

## Audio Front-End

| Component | Documentation | Purpose |
|---|---|---|
| [Speech Enhancement](audio-enhancement.md) | `src/audio_process/` | HPF → RNNoise **or** hush-vani → AGC → soft limiter; clean audio for STT |
| [VAD](vad.md) | `src/vad/` | Voice Activity Detection (Silero native + ONNX) |
| [SmartTurn](turn-acid.md) | `src/turn/` | ML end-of-turn detection — defers stops on hesitation pauses ([phase 2](turn-acid-phase2.md)) |

## Transport

| Component | Documentation | Feature | Purpose |
|---|---|---|---|
| [WebSocket + Channel](transport.md) | `src/transport/` | `transport-websocket` ✅ | WebSocket I/O, ChannelTransport for testing |
| [WebRTC](vaniwebrtc.md) | `src/transport/vaniwebrtc/` | `vaniwebrtc` ❌ | P2P Opus/SRTP, no SFU |
| [Twilio serializer](serializer-twilio.md) | `src/serializers/` | `serializer-twilio` ✅ | Twilio Media Streams, G.711, phone agents |

## Tools

| Component | Documentation | Purpose |
|---|---|---|
| [Postgres Tool](tools-postgres.md) | `src/tools/postgres/` | Built-in LLM tool for Neon Postgres + pgvector |

## Observability

| Component | Documentation | Purpose |
|---|---|---|
| [Billing](billing.md) | `src/billing/` | Per-session usage (tokens, TTS chars, STT duration) + conversation transcript |
| [Audio Capture](audio-capture.md) | `src/audio_capture/` | Per-turn WAV recordings for user and bot, linked to transcript entries |

## Quick Pipeline Assembly

All services follow the same pattern:

```rust
use rustvani::*;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

// Shared turn-id cells (needed if using transcript + audio capture together)
let active_user_turn_id: Arc<Mutex<Option<Uuid>>> = Arc::new(Mutex::new(None));
let active_bot_turn_id:  Arc<Mutex<Option<Uuid>>> = Arc::new(Mutex::new(None));

let (billing, _) = SessionBilling::new(session_id, billing_storage, 256);
let (audio_cap, _) = SessionAudioCapture::new(session_id, audio_storage, 64);

let stt          = SomeSttHandler::new(stt_config).into_processor();
let user_agg     = LLMUserAggregator::with_billing(context.clone(), billing.clone(), active_user_turn_id.clone());
let llm          = SomeLlmHandler::new(llm_config).into_processor();
let assistant_agg = LLMAssistantAggregator::with_billing(context.clone(), billing.clone(), active_bot_turn_id.clone());
let tts          = SomeTtsHandler::new(tts_config).unwrap().into_processor();
let audio_proc   = AudioCaptureProcessor::new(audio_cap, active_user_turn_id, active_bot_turn_id);

let task = PipelineTask::new(
    vec![
        transport.input(),
        stt, user_agg, llm, assistant_agg, tts,
        audio_proc,           // after TTS, before output transport
        transport.output(),
    ],
    PipelineParams {
        allow_interruptions: true,
        billing_collector: Some(billing),
        billing_metadata: [("user_id".into(), "u_42".into())].into_iter().collect(),
        ..Default::default()
    },
);
```

Note that the aggregators' `new` / `with_billing` constructors already return a `FrameProcessor` — unlike the service handlers, they take no `.into_processor()`.

For a complete, compilable program see [`examples/quickstart.rs`](../examples/quickstart.rs) (`cargo build --example quickstart`).

See individual service pages for exact config fields, feature flags, and environment variables.
