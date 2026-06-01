# Piper TTS

**File:** `src/services/tts/piper.rs`  
**Feature:** Always available (no feature gate)  
**Protocol:** Local ONNX inference — zero network calls

Fully on-device text-to-speech via ONNX Runtime. Requires `espeak-ng` for phonemization. Ideal for offline deployments or when network latency must be eliminated.

## Pipeline Position

```
llm → assistant_agg → PiperTtsHandler → transport.output()
```

## Usage

```rust
use rustvani::services::tts::piper::{PiperTtsConfig, PiperTtsHandler, PiperQuality};

let tts = PiperTtsHandler::new(PiperTtsConfig {
    quality: PiperQuality::Medium,
    model_dir: "./piper-models".into(),
    ..Default::default()
}).unwrap().into_processor();
```

## Share Model Across Sessions

```rust
let shared = tts.shared_model();
let tts2 = PiperTtsHandler::with_shared_model(config, shared).into_processor();
```

## Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `quality` | `PiperQuality` | `Medium` | `Low` (~15 MB), `Medium` (~60 MB), `High` (~65 MB) |
| `model_path` | `Option<PathBuf>` | `None` | Explicit `.onnx` path |
| `config_path` | `Option<PathBuf>` | `None` | Explicit `.onnx.json` path |
| `model_dir` | `PathBuf` | `./piper-models` | Directory containing model files |
| `speaker_id` | `Option<i64>` | `None` | Multi-speaker model ID |
| `length_scale` | `Option<f32>` | `None` | Speed: `<1.0` faster, `>1.0` slower |
| `noise_scale` | `Option<f32>` | `None` | Phoneme variation |
| `noise_w` | `Option<f32>` | `None` | Phoneme width variation |
| `num_threads` | `Option<usize>` | `None` | ONNX intra-op threads (default: quality preset) |
| `min_buffer_size` | `usize` | `50` | Min chars before sentence split |
| `max_chunk_length` | `usize` | `150` | Max chars per chunk |

## Quality Defaults

| Quality | Default Model | ONNX Threads |
|---|---|---|
| `Low` | `en_US-lessac-low` | 1 |
| `Medium` | `en_US-lessac-medium` | 2 |
| `High` | `en_US-lessac-high` | 2 |

## Frames

**Consumed:**
- `StartFrame` → logs readiness
- `LLMFullResponseStart` → begins buffering
- `LLMText` → buffers, sentence-splits, synthesizes
- `LLMFullResponseEnd` → flushes remaining text
- `Interruption` → clears buffer (no reconnect — local inference)
- `EndFrame` / `CancelFrame` → passthrough

**Produced:**
- `OutputAudioRaw` (downstream) as 16-bit PCM, chunked into 20 ms frames

## Timing Logs

```
[123.456] [tts:piper] phonemize  12.3ms  (42 chars → 62 IPA chars)
[123.457] [tts:piper] inference  156.2ms
[123.458] [tts:piper] first_chunk  640 bytes
```

## System Dependency

Install `espeak-ng` before running:

```bash
# Debian / Ubuntu
apt-get install espeak-ng

# Fedora
dnf install espeak-ng
```

## Model Files

Download Piper models and place them in `model_dir`:

```
piper-models/
├── en_US-lessac-medium.onnx
└── en_US-lessac-medium.onnx.json
```

Models available at: https://github.com/rhasspy/piper/releases
