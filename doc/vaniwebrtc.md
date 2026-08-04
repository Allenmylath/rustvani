# WebRTC Transport (`vaniwebrtc`)

**Files:** `src/transport/vaniwebrtc/`  
**Feature:** `vaniwebrtc` (**opt-in**)  
**Media:** Opus over RTP/SRTP, peer-to-peer — **no SFU or media server in the path**  
**Signaling:** SDP offer/answer + trickle ICE as JSON over a WebSocket

The WebSocket transport is simple but pays a latency and bandwidth tax: raw PCM over TCP, with head-of-line blocking when the network stutters. `vaniwebrtc` carries audio the way browsers actually want to — Opus over SRTP, straight between the browser and your server process.

## Build prerequisites

This feature pulls a large dependency tree (`webrtc-rs`) and compiles libopus through `audiopus`, so the build host needs **cmake and a C/C++ compiler** (MSVC on Windows). This is why it is not a default feature.

```toml
[dependencies]
rustvani = { version = "0.4.0-dev.9", features = ["vaniwebrtc"] }
```

## Usage

The shape mirrors `WebSocketTransport` exactly — `input()`, `output()`, and a `run` that drives the connection:

```rust
use rustvani::transport::{TransportParams, VaniWebRTCParams, VaniWebRTCTransport};

let transport = VaniWebRTCTransport::new(
    "webrtc",
    VaniWebRTCParams {
        transport: TransportParams {
            audio_in_enabled:  true,
            audio_out_enabled: true,
            vad_analyzer:      Some(vad),
            ..TransportParams::default()
        },
        ..VaniWebRTCParams::default()
    },
);

let task = PipelineTask::new(vec![transport.input(), stt, /* … */, transport.output()], params);
let push_tx = task.push_sender();

tokio::join!(
    async { task.run(system_clock(), None).await.ok(); },
    transport.run(socket, push_tx),   // `socket` is the *signaling* WebSocket
);
```

The `socket` handed to `run` carries **signaling only**. Media never touches it — once ICE completes, audio flows P2P over SRTP, and control messages ride a data channel the client opens.

## Configuration — `VaniWebRTCParams`

| Field | Type | Default | Description |
|---|---|---|---|
| `transport` | `TransportParams` | audio in+out enabled, 16 kHz mono | Same role as `WebSocketParams.transport` |
| `ice_servers` | `Vec<String>` | `["stun:stun.l.google.com:19302"]` | STUN urls. Must **not** carry credentials. |
| `turn_servers` | `Vec<TurnServer>` | `[]` | TURN with long-term credentials. Kept separate because `webrtc-rs` rejects a credential-less TURN url. |
| `nat_1to1_ips` | `Vec<String>` | `[]` | Public IPv4(s) advertised as Host candidates |
| `udp_mux` | `Option<Arc<dyn UDPMux>>` | `None` | Share one bound UDP port across all connections |
| `opus_max_avg_bitrate` | `u32` | `510_000` | Forced on the browser's encoder via answer-SDP `fmtp` |
| `opus_fullband` | `bool` | `true` | Request `maxplaybackrate=48000` |
| `opus_dtx` | `bool` | `false` | Discontinuous transmission — off keeps frames steady |
| `denoiser_factory` | `Option<DenoiserFactory>` | `None` | Per-connection 48 kHz inbound denoiser hook |

### Why the Opus settings are so aggressive

The defaults deliberately force high-bitrate, full-band Opus out of the browser. A full-band denoiser (DeepFilterNet-class) needs the high-frequency content that a typical 24 kbps narrowband Opus stream throws away — starve it and the enhancement stage has nothing to work with. `munge_answer_sdp` rewrites the Opus `a=fmtp` line in the answer to apply these; it is idempotent and leaves the SDP untouched if no Opus codec is present.

`Denoiser48k` / `denoiser_factory` is the hook for that stage. It is a factory rather than a shared instance because a denoiser holds per-connection state. `None` (the default) is a transparent pass-through.

## Deploying behind a NAT / on Fly.io

Two settings matter, and both are easy to miss:

**1. Share one UDP port.** By default each connection grabs ephemeral UDP ports, which is fine on a LAN but not forwardable behind an edge that only routes a known port. Build the mux **once at startup** and clone the `Arc` into every connection — rebuilding it per connection re-binds the same port and fails with "address in use":

```rust
use rustvani::transport::build_shared_udp_mux;

// Once, in main():
let mux = build_shared_udp_mux("fly-global-services:3478").await?;   // or "0.0.0.0:3478"

// Per connection:
VaniWebRTCParams { udp_mux: Some(mux.clone()), ..Default::default() }
```

**2. Advertise a reachable Host candidate.** Set `nat_1to1_ips` to the platform's dedicated public IPv4 so an IPv4-only browser has something to pair with. With that in place, STUN alone is usually enough and you can leave `turn_servers` empty.

## Signaling protocol

JSON `SignalMsg` values over the WebSocket:

| Variant | Direction | Payload |
|---|---|---|
| `Offer` | client → server | `{ "sdp": "…" }` |
| `Answer` | server → client | `{ "sdp": "…" }` |
| `Ice` | either | `{ "candidate": "…", "sdpMid": …, "sdpMLineIndex": … }` |
| `Bye` | either | graceful close |

## Running the reference server

```bash
cargo run --bin vaniwebrtc_server --features vaniwebrtc,tts-sarvam
```

The server uses Sarvam TTS, so it needs `tts-sarvam` in addition to `vaniwebrtc`. Signaling is at `ws://0.0.0.0:8080/rtc`. Open [`examples/vaniwebrtc_client.html`](../examples/vaniwebrtc_client.html) in a browser to talk to it.

Required environment: `SARVAM_API_KEY`, `OPENAI_API_KEY`.

## See Also

- [Transport](transport.md) — `WebSocketTransport`, `ChannelTransport`
- [Speech Enhancement](audio-enhancement.md) — the 16 kHz STT-path chain (distinct from the 48 kHz `Denoiser48k` hook)
- [VAD](vad.md)
