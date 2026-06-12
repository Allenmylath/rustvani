# Voice Activity Detection (VAD)

**Files:** `src/vad/silero_native.rs`, `src/vad/silero_ort.rs`  
**Feature:** `vad-silero-ort` (enabled by default) — gates **only the ONNX (ort) backend** and its `ort`/C-C++ dependency, not Silero VAD itself. The pure-Rust `SileroVadNative` backend is always available and needs no feature.

Two backends for Silero VAD with the same API. Detects speech start/stop and drives the pipeline via `VADUserStartedSpeaking` / `VADUserStoppedSpeaking` frames.

## Backends

| Backend | Struct | Sample Rates | ONNX Runtime | Memory |
|---|---|---|---|---|
| Native | `SileroVadNative` | 16 kHz only | No | ~5 MB |
| ONNX Runtime | `SileroVadOrt` | 8 kHz + 16 kHz | Yes | Larger |

## Usage

### Native (Zero Dependencies)

```rust
use rustvani::vad::{SileroVadNative, VadBackend, create_vad};

let vad = SileroVadNative::new(16_000).expect("VAD load failed");

// Or use the factory:
let vad = create_vad(VadBackend::Native, 16_000).unwrap();
```

### ONNX Runtime

```rust
use rustvani::vad::{SileroVadOrt, VadBackend, create_vad};

let vad = SileroVadOrt::new(16_000).expect("VAD load failed");

// Or use the factory:
let vad = create_vad(VadBackend::Ort, 16_000).unwrap();
```

### In Transport

```rust
use rustvani::transport::TransportParams;

let transport = BaseTransport::new(TransportParams {
    audio_in_enabled: true,
    audio_in_sample_rate: Some(16_000),
    vad_analyzer: Some(Arc::new(vad)),
    ..Default::default()
});
```

## Configuration

`VadParams` (used by both backends):

| Field | Default | Description |
|---|---|---|
| `confidence` | `0.5` | Speech probability threshold |
| `start_secs` | `0.2` | Duration of speech before `StartedSpeaking` |
| `stop_secs` | `0.8` | Duration of silence before `StoppedSpeaking` |
| `min_volume` | `-45.0` | dFS minimum volume gate |

```rust
use rustvani::vad::VadParams;

let params = VadParams {
    confidence: 0.6,
    start_secs: 0.15,
    stop_secs: 0.5,
    ..Default::default()
};
```

## State Machine

```
Quiet → Starting → Speaking → Stopping → Quiet
```

## SmartTurn (End-of-Turn Prediction)

An optional local ONNX model defers `VADUserStoppedSpeaking` on hesitation pauses. Enable via the turn engine:

```rust
use rustvani::turn::engine::TurnEngine;
```

See `src/turn/` for details.

## Client + Server VAD Coordination

When using the Dioxus frontend integration, the browser runs its own lightweight VAD and pushes `ClientVADUserStartedSpeaking` events into the server pipeline. A CAS toggle ensures exactly one `VADUserStartedSpeaking` is emitted per utterance regardless of which side fires first.

```rust
// Called from your WebSocket handler
transport.push_client_vad_started(&processor, timestamp).await;
transport.push_client_vad_stopped(&processor, timestamp).await;
```

## Cargo Feature

Enabled by default. To disable:

```toml
[dependencies]
rustvani = { version = "0.2", default-features = false, features = ["transport-websocket", "stt-sarvam", "stt-60db", "llm-openai", "tts-deepgram", "tts-sarvam", "dhara"] }
```
