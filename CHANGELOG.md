# Changelog

## 0.4.0-dev (unreleased)

Prerelease line. Install with an explicit version — `rustvani = "0.4.0-dev.9"` —
since Cargo will not resolve a prerelease from a plain version requirement.

### Telephony — Twilio Media Streams (new)

- New `crate::serializers` module: a `FrameSerializer` trait that sits between
  `WebSocketTransport` and a telephony provider, encoding outgoing frames into
  provider messages and decoding incoming ones back into frames.
- `TwilioFrameSerializer` handles Twilio's Media Streams JSON end to end:
  inbound `media` events (base64 µ-law @ 8 kHz) are resampled up to the pipeline
  rate; outbound TTS audio is resampled down and re-encoded; barge-in emits
  Twilio's `clear` event; `dtmf` events become `InputDTMFFrame`s.
- `TwilioStart::parse` extracts `stream_sid` / `call_sid` / `account_sid` from
  the handshake; `TwilioFrameSerializer::from_start` builds a serializer from it.
- `serializers::g711` — standalone µ-law/A-law codec with optional in-line
  resampling (`pcm_to_ulaw`, `ulaw_to_pcm`).
- `WebSocketTransport::set_serializer` installs one; call before `run_socket`.
- New feature `serializer-twilio` (default) gates only the REST auto-hang-up
  path, which pulls `reqwest`. The serializer itself builds under
  `transport-websocket`.
- New binary `twilio_coordinator_server` — a complete phone agent on default
  features.

### Transport — peer-to-peer WebRTC (new)

- New opt-in feature `vaniwebrtc` and `crate::transport::vaniwebrtc` module:
  Opus over RTP/SRTP carried peer-to-peer with **no SFU or media server** in the
  path. Signaling (SDP offer/answer + trickle ICE) rides a WebSocket; control
  messages ride a WebRTC data channel.
- `VaniWebRTCTransport::new(name, params)` mirrors `WebSocketTransport` —
  `input()`, `output()`, and `run(socket, push_tx)`.
- `VaniWebRTCParams`: `ice_servers`, `turn_servers` (`TurnServer` with long-term
  credentials), `nat_1to1_ips`, `udp_mux`, Opus tuning
  (`opus_max_avg_bitrate`, `opus_fullband`, `opus_dtx`), and a
  `denoiser_factory` hook for a 48 kHz inbound `Denoiser48k`.
- `build_shared_udp_mux(bind_addr)` binds one UDP socket at startup so every
  connection shares a single port — required behind edges that forward only a
  known port (Fly.io). Call once and clone the `Arc`.
- `munge_answer_sdp` rewrites the Opus `a=fmtp` line to force high-bitrate,
  full-band Opus, so a full-band denoise stage isn't starved of high frequencies.
- Adds optional `webrtc` and `audiopus` dependencies; building this feature
  needs cmake and a C/C++ toolchain (MSVC on Windows).
- New binary `vaniwebrtc_server` (`required-features = ["vaniwebrtc"]`) and a
  browser client at `examples/vaniwebrtc_client.html`.

### Audio — selectable noise-suppression backend

- New `StreamingDenoiser` trait (`filter` / `flush` / `reset`) in
  `crate::audio_process`, implemented by both denoisers so the STT path can
  select one at runtime behind a single `Box<dyn>`.
- New `audio_process::hushfilter::HushVaniFilter` — DeepFilterNet3-style
  suppression via the `hush-vani` crate. `hush-vani` exposes only a batch
  `enhance()` whose GRUs reset per call, so the filter wraps it in a sliding
  window: each call re-runs `enhance` over 200 ms of lookback context plus the
  new audio to re-prime the GRUs, discards the context region of the output, and
  holds back the trailing 160 samples (the documented algorithmic lag) until
  `flush()`. Non-16 kHz input is resampled transparently.
- New `SarvamSttConfig::noise_backend: NoiseBackend` — `Rnnoise` (default,
  true streaming) or `HushVani` (opt-in, stronger suppression). A `HushVani`
  init failure logs and falls back to RNNoise rather than failing the pipeline.
- Adds `hush-vani` as a regular (non-optional) dependency — no feature flag.

### Agents

- New `CoordinatorProcessor` (`agents::coordinator_processor`): a bus-connected
  `FrameProcessor` that can sit in the LLM slot and answer from local logic,
  with or without an agent swarm behind it. Exposes `CoordinatorCtx`,
  `CoordinatorFn`, `DEFAULT_CALL_TIMEOUT`.
- `AgenticCoordinator` / `FenceOutcome` for turn-level epoch fencing.

### VAD

- Fixed: the analyzer now drains **every** ready window per audio frame instead
  of one, so speech transitions are not delayed when frames carry more than one
  window's worth of samples.

### Documentation

- README rewritten against the current API. The Quick Start is now
  `examples/quickstart.rs` verbatim and builds on default features; several
  snippets in the previous README did not compile (`OpenAILLMConfig` has no
  `context` field, `BaseTransport::new` takes a name, the aggregators' `new`
  already returns a `FrameProcessor`, and `SarvamTtsHandler` needs the
  non-default `tts-sarvam` feature).
- New: a feature-flags table, plus `doc/stt-deepgram.md`,
  `doc/serializer-twilio.md`, and `doc/vaniwebrtc.md`.
- `doc/audio-enhancement.md` documents the two noise backends and the
  hush-vani sliding-window design.

## 0.3.0 — 2026-06-11

Rust-native rewrite of the agents module internals plus Pipecat-aligned
worker semantics (job routing, cancellation cleanup, ready-gating, frame
bridging via pipeline edges). **Breaking release** — all changes below are
in `rustvani::agents` unless noted.

### Bus (`agents::bus`)

- **Breaking:** `BusSubscriber::on_bus_message` now takes `Arc<BusMessage>`
  instead of `BusMessage`. Messages fan out as a single `Arc` — no deep
  clones per subscriber.
- **Breaking:** `BusMessage` gained a `seq: u64` field, stamped by the bus
  in `send()` before fan-out (total-order debugging across agents).
  Construct messages with the new `BusMessage::new(source, target, payload)`
  instead of a struct literal.
- `LocalAgentBus` internals rewritten: per-subscriber dispatch tasks built
  on an unbounded system channel + bounded data channel (default capacity
  256, configurable via `LocalAgentBus::with_capacity`), drained with a
  `biased tokio::select!`. System messages (End/Cancel/Activate/urgent task
  replies/registry) are always delivered before queued data messages and
  are never dropped; data messages are dropped (counted + rate-limited
  warn) when a subscriber's channel is full — control never drops, data
  never blocks.
- `subscribe()` now rejects duplicate names with an error (they previously
  coexisted silently).
- `unsubscribe()`/`stop()` let in-flight handlers finish (2 s grace, then
  abort) and deliver pending system messages before exiting.
- `send()` after `stop()` is a no-op. `send()` itself is lock-free
  (`arc-swap` subscriber snapshot).
- New: `LocalAgentBus::dropped_count(name)`, `DEFAULT_DATA_CAPACITY`.

### BaseAgent (`agents::base`)

- **Breaking:** `BaseAgent::new`'s `bridged` parameter changed from `bool`
  to `Option<Vec<String>>`: `None` = not bridged, `Some(vec![])` = accept
  bridged frames from any source, `Some(names)` = accept only from the
  listed peers. The `Agent::bridged() -> bool` accessor is unchanged
  (`true` when `Some`).
- Task routing implemented (Pipecat `BaseWorker` parity): register job
  handlers with `BaseAgent::on_task(name, handler)` /
  `on_task_default(handler)`. Handlers receive a `TaskRequestCtx` (new
  type) with `complete` / `stream_start` / `stream_data` / `stream_end`
  helpers; each handler runs in its own tokio task. Requests with no
  matching handler fail fast with a terminal `Failed` response.
- `TaskCancel` aborts the running handler and always sends the requester a
  terminal `Cancelled` response.
- Lifecycle cleanup: on `end()`/`cancel()`/`run()` return, all in-flight
  jobs are cancelled with terminal responses and all pending dispatched
  handles are failed (`TaskContext::fail_all_pending`) — a requester is
  never left hanging on a dead agent.
- `end()`/`cancel()` are now single-shot (repeat calls are no-ops) and
  cascade to child agents (see Runner below).
- New: `BaseAgent::bridged_pipeline(...)` convenience constructor,
  `with_output_edge`, `task_ctx()`, `pipeline()`, `active_flag()` accessors.
  New exported types: `TaskHandler`, `TaskRequestCtx`.
- Internals: bus/registry now `OnceCell` (set once in `setup()`, which now
  errors if called twice).

### TaskContext (`agents::task`)

- **Breaking:** `TaskContext::new(bus)` is now
  `TaskContext::new(bus, registry)`.
- `dispatch()` now ready-gates: if the target is not in the registry it
  waits (watch-based, no polling) up to 10 s
  (`DEFAULT_READY_TIMEOUT`) and errors on timeout instead of silently
  dropping the request. `dispatch_with(...)` exposes the timeout
  (`None` = send immediately without gating).
- New: `fail_all_pending(reason)`, `stream_start`, `stream_end`.
  `UpdateHandler` is now exported.
- `TaskHandle::await_completion` now skips non-terminal updates (stream
  chunks/progress) instead of erroring on them.

### Bus edges (`agents::edges` — new module)

- `BusOutputEdge`: a tail-of-pipeline `FrameProcessor` that forwards every
  frame through unchanged and publishes non-excluded frames to configured
  peer agents (empty = broadcast) as `BusPayload::Frame`. Lifecycle,
  task-control, processor-control, error, and heartbeat frames are never
  published. Publishing is gated by the owning agent's activation flag
  (inactive agents also skip injecting bridged input) — Pipecat-style
  handoff between several brains and one transport pipeline.

### Registry (`agents::registry`)

- New: `children_of(name)`, `mark_finished(name)`, `wait_finished(name)`
  (per-agent finished signal used by the End cascade).

### Runner (`agents::runner`)

- Shutdown now cascades through the agent tree: `End` goes to root agents
  only; an agent receiving `End` first forwards it to its children and
  waits (5 s per child, runner's 10 s join timeout as backstop) for them to
  finish before pushing its own `EndFrame`. `Cancel` propagates without
  waiting. The runner validates parent references at `run()` time.

### Frames (`crate::frames`)

- **Breaking (type change):** `AudioRawData.audio` is now `bytes::Bytes`
  instead of `Vec<u8>` — cloning an audio frame no longer copies the
  payload. `AudioRawData::new`, `Frame::input_audio`, `Frame::output_audio`
  accept `impl Into<Bytes>`, so existing `Vec<u8>` callers compile
  unchanged. Code that needs `Vec<u8>` back uses `.to_vec()` (explicit
  copy).

### Dependencies

- Added `arc-swap` (lock-free subscriber snapshot) and `bytes`
  (cheap-clone audio payloads). `tokio-util` (already a dependency) now
  also provides `CancellationToken`.

### Misc

- New example: `examples/bridged_agents.rs` — two bridged agents
  demonstrating dispatch, streaming reply, and cancel.
- Fixed stale assertions in the sixtydb STT config test and a
  non-exhaustive match in `examples/channel_pipeline.rs`.
