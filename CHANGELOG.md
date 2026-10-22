# Changelog

## 0.4.0-dev (unreleased)

Prerelease line. Install with an explicit version — `rustvani = "0.4.0-dev.15"` —
since Cargo will not resolve a prerelease from a plain version requirement.

### Dubbing: translations spoken aloud (0.4.0-dev.15)

- **`TtsSynthesizer` trait** (`services::tts::synth`, always compiled): one utterance
  in, its whole audio (`Pcm`) out, with a voice and a pace per call. The shape
  dubbing needs, beside the streaming TTS handlers that speak an LLM's answer.
- **`SarvamRestTts` / `SarvamRestTtsConfig`** (feature `tts-sarvam`, which now
  also pulls `reqwest`): Bulbul over `POST /text-to-speech`, `bulbul:v3` by
  default, 48 kHz WAV decoded to PCM; the eleven languages Bulbul speaks.
- **`DubbingService`** (`services::dub`, always compiled): after the
  `TranslationService`, speaks every final `TranslationFrame` whose target is in
  its `TranslationTargets` and emits a new **`DubbedSpeechFrame`**
  (`DataFrame::DubbedSpeech`, `DubbedSpeechData`: the PCM plus the source
  transcript's `timestamp` and `source_text`, the voice and the pace). One
  ordered queue per language; a `PaceHint` from the host sets each utterance's
  pace.
- **`BillingEvent::TtsUsage` gains `language: Option<String>`** (struct literals
  need the new field; it serializes only when set).

### Translation as a pipeline stage (0.4.0-dev.14)

- **New `TranslationService`** (`services::translate`, always compiled): sits after
  the STT and translates every *final* transcript — `TranscriptionFrame`s with
  `finalized`, and the turn-closing transcript bundled on `VADUserStoppedSpeaking`
  — into each target language, as a new **`TranslationFrame`**
  (`DataFrame::Translation`, `TranslationData`: `text`, `target_language`, and the
  source transcript's `source_text`, `source_language`, `user_id`, `timestamp`).
  Interims are not translated. One ordered queue per target language runs off the
  pipeline's tasks, so a slow call never delays the transcript or another language.
  Targets are a **`TranslationTargets`** handle that can change while the pipeline
  runs; a transcript already in the target language is passed through as its own
  translation. Long text is split at sentence ends to the provider's limit.
- **`TranslationProvider` trait** (`name`, `model`, `supports`, `max_chars`,
  `translate`) for any text-to-text backend.
- **`SarvamTranslateProvider` / `SarvamTranslateConfig`** (feature
  `translate-sarvam`, default on): `POST /translate`, `sarvam-translate:v1` by
  default or `mayura:v1`; BCP-47 tags (`hi`, `en-IN`) mapped to Sarvam's `xx-IN`
  codes. Sarvam detects the source language only on `mayura:v1`, so a transcript
  of unknown language goes to `auto_detect_model` (default `mayura:v1`). One retry
  on 429/5xx.
- **RAVI `user-translation`** message (`models::msg_user_translation`), sent by
  `RaviObserver` for every `TranslationFrame`;
  `RaviObserverParams::user_translation_enabled` (default `true`). Struct-literal
  constructions of `RaviObserverParams` need the new field.
- **`BillingEvent::TranslationUsage`** (`provider`, `model`, `source_language`,
  `target_language`, `char_count`), summed into `SessionSummary::translation_chars`
  / `translation_calls`; the Postgres schema adds those columns and
  `billing_events.source_language` / `target_language` (migrated with
  `ADD COLUMN IF NOT EXISTS`). Exhaustive matches on `BillingEvent`, `DataFrame`
  and `FrameKind` gain a variant.

### Sarvam realtime STT: live partial transcripts (0.4.0-dev.13)

- **New `SarvamRealtimeSttHandler` / `SarvamRealtimeSttConfig`** (feature
  `stt-sarvam`) for Sarvam's realtime endpoint
  (`/speech-to-text-realtime/ws`, `saaras:v3-realtime`): interim transcripts
  while the user is still speaking, then a final. `stream_type` `fast`
  (default), `balanced` or `simulated`. `endpointing` defaults to `manual`:
  local VAD drives the turns (`speech_start` / `speech_end`), as with every
  other provider; `vad` hands endpointing to Sarvam's server VAD
  (`silence_duration_ms`, `threshold`, `min_speech_duration_ms`), which
  ignores `flush`. Language `unknown` (the default) detects it.
  On All India Radio Malayalam news: first text 0.6–1.1 s after speech
  starts, words on screen ~1.3–1.8 s after they are spoken (the classic
  streaming endpoint: 6.4 s, or 3.5 s with an 8 s turn limit), at 3.9–5.2%
  character error.
- **`SttProvider::turn_start_msg`** (default `None`): sent when local VAD opens
  a turn, before the pre-roll.
- **`SttProvider::mid_turn_finalize_msg`** (default: `finalize_msg`) for
  `max_turn_ms` cuts; the driver sends `turn_start_msg` right after it. For
  Sarvam realtime in manual mode a mid-turn `flush` lost audio; it now ends the
  segment with `speech_end` and opens the next with `speech_start`.
- **Repeated identical partials are pushed once.** Sarvam realtime repeats the
  same hypothesis many times a second.

### Live transcription: VAD defaults, a turn limit, detected language (0.4.0-dev.12)

Measured on real speech: All India Radio Malayalam news, and browser-microphone
speech received over WebRTC/Opus.

- **VAD defaults changed** to confidence 0.5, min_volume 0.2, start_secs 0.2,
  stop_secs 0.8 (were 0.7, 0.3, 0.4, 0.4). A turn opens only after
  `start_secs` of unbroken confident speech; after a browser's audio
  processing and Opus, Silero's confidence stays above 0.7 for runs of only
  ~0.36 s (median), so at 0.4 s turns often never opened: 38–69% of a
  speaker's speech reached the STT, with deaf spells of up to 7 s. The new
  defaults pass 99–100%, open no turns on white noise down to -30 dBFS, and
  lowered Sarvam's character error rate on the news from 3.9/2.6% to
  3.0/1.3%. Pass `VadParams` explicitly to keep the old behaviour.
- **New `SttCoreConfig::max_turn_ms` / `SarvamSttConfig::max_turn_ms`**
  (default `None`, unchanged behaviour): once a turn has sent that much audio,
  the provider is asked to finalise what it has heard (its `finalize_msg`,
  Sarvam's flush) at the next gap between words (80 ms of quiet), or 2 s
  later if none comes; the text arrives as a mid-turn transcript and the turn
  goes on. For live captions of speakers who rarely pause: a newsreader's
  turns run 15–25 s, and a transcript otherwise waits for the turn to end.
  At 8 s, captions came 3× sooner for about 1.7 points more character error.
- **RAVI `user-transcription` carries the language** (`data.language`) when
  the STT reports one. `ravi::models::msg_user_transcription` takes a
  `language: Option<&str>` argument.
- **SmartTurn inference runs on Tokio's blocking pool**
  (`SmartTurnAnalyzer::analyze_end_of_turn_blocking`). It takes ~0.8 s of CPU
  per turn end and ran inline on the input transport's audio task (and inside
  `VadProcessor`'s state lock), holding a runtime worker that long and
  stalling other tasks. If the inference panics the turn ends and the input
  falls back to VAD-only.

### VAD — follows the sample rate of incoming audio (0.4.0-dev.11)

VAD and SmartTurn read every input frame as if it were at
`TransportParams::audio_in_sample_rate`, so 48 kHz audio (an Opus decode, an
SFU tap) or stereo was misread and VAD misfired.

- New `vad::VadInput`: converts each frame from the rate and channel count it
  carries to mono at the model's rate. Audio already in that format is
  borrowed, not copied; the resampler is rebuilt when the source rate changes.
  Resampling adds 10 ms of VAD latency at 48 kHz (60 ms at 8 kHz), none at
  16 kHz.
- New `VadAnalyzer::sample_rate()` (default method; `SileroVadNative` reports
  16 kHz, `SileroVadOrt` the rate it was built with). The input transport and
  `VadProcessor` run the VAD state machine at that rate and feed VAD and
  SmartTurn the converted audio.
- New `ChannelMessage::AudioFrame { audio, sample_rate, channels }` for audio
  tagged with its own format. `ChannelMessage::Audio(bytes)` still uses the
  configured rate. Exhaustive matches on `ChannelMessage` need a new arm.
- `SttService` downmixes multi-channel frames before its front-end.
- New `audio_process::downmix_to_mono`.
- Passthrough frames keep their original format, so STT resamples once, to
  the provider's rate. Deepgram, Gnani and 60db are not on the STT core yet
  and still assume their configured rate.

### STT — reusable provider abstraction (new)

Every STT backend was the same program with a different wire format, and each
re-derived ~250–350 lines of plumbing: the WebSocket handshake, the send and
receive tasks, `WsSink`/`WsStream` aliases, byte helpers, the frame-dispatch
match, and a billing block whose only difference was a string literal. Worse,
the turn machinery existed in exactly one of them.

- New `services::stt::core` module:
  - `SttProvider` — the base trait a service implements. It answers four
    questions and owns no state: `handshake()`, `encode_audio()`,
    `finalize_msg()`, `parse()`. All methods take `&self`.
  - `SttService<P>` — generic over `SttProvider`, and now the **only**
    `FrameHandler` in the STT subsystem. Owns the WebSocket session, the turn
    gate, the audio front-end and billing. `SttCoreConfig` carries the
    provider-independent settings.
  - `TurnGate` — moved out of `services/stt/sarvam.rs`, unchanged, with its
    tests. Audio gating, pre-roll ring, stashed `VADUserStoppedSpeaking`,
    generation-guarded release timeout, and the duration ledger for turn
    attribution and billing. Every provider built on the core now gets the
    transcript-before-stop ordering `LLMUserAggregator` depends on.
  - `AudioFrontend` — resample → high-pass → denoise → AGC + soft limiter, as
    one reusable struct composing the existing `StreamResampler`,
    `StreamingDenoiser` and `AudioEnhancer`.
  - `SttEvent` / `Outgoing` / `Handshake` / `AudioSpec` / `WsMessage` vocabulary
    types; shared `ws` and `util` helpers.
- `NoiseBackend` moved to `core::frontend` — it was never Sarvam-specific. It is
  re-exported from `services::stt::sarvam`, `services::stt`, `services` and the
  crate root, so every existing path still resolves.
- **New: input audio is resampled to the provider's rate.** Sarvam and Deepgram
  previously ignored `AudioRawData.sample_rate` entirely, so a transport/config
  mismatch mis-transcribed silently.
- Interim transcripts are first-class: `SttEvent::Partial` plus
  `SttCoreConfig::interim_policy` (`InterimPolicy::Drop`, the default, or
  `Emit` for live captions).
- `SarvamSttHandler` rebuilt on the core — 1508 → 682 lines, with the protocol
  now unit-testable without a socket (6 → 23 tests). Public API is unchanged.
- New `SarvamSttConfig::prompt` (recognition biasing) and
  `SarvamSttConfig::vad_tuning: SarvamVadTuning` — the ten fine-grained
  `saaras:v3` server-VAD parameters, emitted only when set and only for v3.
- Deepgram, Gnani and 60db are unchanged and still on the old path.

### Fixed — interim transcripts polluted LLM context

`LLMUserAggregator` appended the text of *any* `TranscriptionFrame` and never
read `TranscriptionData::finalized`. A provider streaming partials therefore
concatenated every prefix of the sentence into the user turn
("he hello hello there"). 60db is the one provider that propagates the real
`is_final` flag, so this was live. Non-finalized transcripts are now forwarded
downstream — caption UIs still see them — but never aggregated.

### Fixed — `stt-sarvam` declared the wrong dependency

`stt-sarvam` was declared as `["dep:reqwest"]`, but Sarvam STT is a WebSocket
service and never used `reqwest`; it only built because the default feature set
drags in `transport-websocket`. `cargo build --no-default-features --features
stt-sarvam` failed. It is now `["dep:tokio-tungstenite"]`.

**Breaking under non-default feature sets:** Sarvam's LLM was gated on
`stt-sarvam` purely because that feature happened to carry `reqwest`. It now has
its own `llm-sarvam` feature (default-on). A build using
`--no-default-features --features stt-sarvam` together with `SarvamLLMHandler`
must add `llm-sarvam`. Default-features builds are unaffected.

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
  (`NoiseBackend` later moved to `services::stt::core::frontend`, where it
  applies to every provider; the old paths still resolve.)
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
