# Transport

**Files:** `src/transport/`  
**Feature:** `transport-websocket` (enabled by default)

Audio I/O layer. Two implementations: WebSocket (production) and ChannelTransport (testing).

## WebSocket Transport

**File:** `src/transport/websocket/axum_websocket_transport.rs`

Axum-based WebSocket transport for browser/Dioxus clients.

```rust
use rustvani::transport::{BaseTransport, TransportParams};

let transport = BaseTransport::new(TransportParams {
    audio_in_enabled: true,
    audio_in_sample_rate: Some(16_000),
    vad_analyzer: Some(Arc::new(vad)),
    audio_out_enabled: true,
    ..Default::default()
});

let task = PipelineTask::new(
    vec![transport.input(), stt, user_agg, llm, assistant_agg, tts, transport.output()],
    PipelineParams { allow_interruptions: true, ..Default::default() },
);
```

### TransportParams

| Field | Type | Default | Description |
|---|---|---|---|
| `audio_in_enabled` | `bool` | `false` | Accept incoming audio |
| `audio_in_sample_rate` | `Option<u32>` | `None` | Expected input rate |
| `vad_analyzer` | `Option<Arc<dyn VadAnalyzer>>` | `None` | VAD to run on input |
| `audio_out_enabled` | `bool` | `false` | Send audio downstream |
| `...` | | | See `src/transport/params.rs` for full list |

## ChannelTransport

**File:** `src/transport/channel.rs`

In-memory channel transport for unit/integration tests. No WebSocket server needed.

```rust
use rustvani::transport::{ChannelTransport, ChannelMessage, TransportParams};
use tokio::sync::mpsc;

let (incoming_tx, incoming_rx) = mpsc::channel::<ChannelMessage>(10);
let (outgoing_tx, mut outgoing_rx) = mpsc::channel::<ChannelMessage>(10);

let transport = ChannelTransport::new("test", TransportParams {
    audio_in_enabled: true,
    audio_in_sample_rate: Some(16_000),
    ..Default::default()
}, incoming_rx);

let task = PipelineTask::new(
    vec![transport.input(), stt, user_agg, llm, assistant_agg, tts, transport.output()],
    PipelineParams::default(),
);
let push_tx = task.push_sender();

tokio::spawn(async move { task.run(system_clock(), None).await.ok(); });
tokio::spawn(async move { transport.run(push_tx, outgoing_tx).await; });

incoming_tx.send(ChannelMessage::Audio(pcm_bytes)).await.unwrap();
let result = outgoing_rx.recv().await.unwrap();
```

## RAVI Protocol

**File:** `src/ravi/`

Real-time Audio/Video Interface protocol processor. Equivalent to Pipecat's RTVI.

```rust
use rustvani::ravi::RaviProcessor;
```

## Cargo Feature

Enabled by default. To disable:

```toml
[dependencies]
rustvani = { version = "0.2", default-features = false, features = ["vad-silero", "stt-sarvam", "stt-60db", "llm-openai", "tts-deepgram", "tts-sarvam", "dhara"] }
```
