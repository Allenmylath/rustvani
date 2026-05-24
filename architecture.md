# Rustvani Architecture Reference

Complete reference for Frames, FrameProcessor, Pipeline, PipelineTask, Processors,
Services, Context, Tools, and Dhara. For the multi-agent layer see `agents.md`.

---

## Table of Contents

1. [Frames](#1-frames)
2. [FrameProcessor](#2-frameprocessor)
3. [Pipeline](#3-pipeline)
4. [PipelineTask](#4-pipelinetask)
5. [LLMContext](#5-llmcontext)
6. [Processors](#6-processors)
7. [Services — LLM](#7-services--llm)
8. [Services — STT / TTS](#8-services--stt--tts)
9. [Built-in Tools](#9-built-in-tools)
10. [Dhara — Conversation Flows](#10-dhara--conversation-flows)

---

## 1. Frames

Every unit of work in rustvani is a `Frame`. Frames flow through a chain of
`FrameProcessor`s in one direction at a time.

### 1.1 Frame structure

```rust
pub struct Frame {
    pub id:         u64,           // globally unique, monotonically increasing
    pub sibling_id: Option<u64>,   // set by broadcast_frame() to pair DS/US copies
    pub inner:      FrameInner,
}

pub enum FrameInner {
    System(SystemFrame),   // lifecycle, speaking signals, audio input, control
    Control(ControlFrame), // pipeline lifecycle + LLM/function boundaries
    Data(DataFrame),       // content — audio out, text, transcriptions, tool results
}
```

`frame.name()` — human-readable string (e.g. `"StartFrame"`)
`frame.kind()` — flat `FrameKind` enum for filtering/matching
`frame.is_system()` — true for `SystemFrame` variants
`frame.is_uninterruptible()` — true for `EndFrame`, `EndTaskFrame`, `StopTaskFrame`,
`CancelTaskFrame` (these survive an interruption queue drain)

### 1.2 FrameDirection

```rust
pub enum FrameDirection {
    Downstream, // source → sink (transport → STT → LLM → TTS → transport)
    Upstream,   // sink → source (errors, speaking signals bubble back up)
}
```

### 1.3 SystemFrame variants

| Frame | Direction | Purpose |
|---|---|---|
| `Start(StartFrameData)` | DS | Initialise all processors; sets allow_interruptions, metrics flags |
| `Cancel { reason }` | DS | Hard abort — trips CancellationToken, discards queue |
| `Error(ErrorFrameData)` | US | Non/fatal error from any processor |
| `Interruption` | both (broadcast) | User spoke mid-bot-turn — drains process queues |
| `Stop { reason }` | DS | Graceful stop — keeps transport connections alive (hand-off) |
| `EndTask / CancelTask / StopTask / InterruptionTask` | US | Task-control signals intercepted by TaskSource |
| `BotSpeaking / UserSpeaking` | US | Reset idle timer |
| `BotStartedSpeaking / BotStoppedSpeaking` | US | TTS started/finished audio |
| `UserStartedSpeaking { emulated }` | US | VAD confirmed user speaking |
| `UserStoppedSpeaking { emulated }` | US | VAD confirmed user stopped |
| `VADUserStartedSpeaking { start_secs, timestamp }` | US | Raw VAD edge |
| `VADUserStoppedSpeaking { stop_secs, timestamp }` | US | Raw VAD edge |
| `InputAudioRaw(AudioRawData)` | DS | Raw PCM from transport |
| `PauseProcessor { name }` | any | Pause a named processor's data queue |
| `PauseProcessorUrgent { name }` | any | Pause system queue too |
| `ResumeProcessor { name }` | any | Resume data queue |
| `ResumeProcessorUrgent { name }` | any | Resume system queue |
| `Heartbeat(f64)` | DS | Periodic health probe (unix timestamp) |
| `RaviClientMessage { msg_id, msg_type, data }` | DS | Client → server protocol |
| `RaviServerMessage { payload }` | DS | Server → client broadcast |
| `RaviServerResponse { client_msg_id, payload }` | DS | Server → client reply |

### 1.4 ControlFrame variants

| Frame | Purpose |
|---|---|
| `End { reason }` | Graceful pipeline shutdown — processors flush and close |
| `LLMFullResponseStart` | LLM began generating |
| `LLMFullResponseEnd` | LLM finished generating |
| `FunctionCallStart` | Model returned tool calls — execution beginning |
| `FunctionCallEnd` | All tool calls for this turn executed |

### 1.5 DataFrame variants

| Frame | Payload | Purpose |
|---|---|---|
| `Data(DataFrameData)` | `Vec<u8>` + metadata | Generic binary data |
| `OutputAudioRaw(AudioRawData)` | PCM bytes, rate, channels | Audio to play |
| `Transcription(TranscriptionData)` | text, user_id, timestamp, language, finalized | STT result |
| `LLMText(String)` | streamed token | LLM response text chunk |
| `LLMContextFrame(Arc<Mutex<LLMContext>>)` | shared context | Trigger LLM inference |
| `FunctionCallInProgress(FunctionCallData)` | id, name, arguments | Model requested tool |
| `FunctionCallResult(FunctionCallResultData)` | id, name, result (summary) | Tool result for LLM |
| `FunctionCallRawResult(FunctionCallRawResultData)` | id, name, raw_data (Value) | Full tool output (LLM never sees this) |

### 1.6 Key payload types

```rust
pub struct AudioRawData {
    pub audio: Vec<u8>,
    pub sample_rate: u32,
    pub num_channels: u16,
    pub num_frames: usize,
    pub transport_source: Option<String>,
}

pub struct StartFrameData {
    pub allow_interruptions: bool,
    pub enable_metrics: bool,
    pub enable_usage_metrics: bool,
    pub report_only_initial_ttfb: bool,
    pub metadata: HashMap<String, String>,
}

pub struct TranscriptionData {
    pub text: String,
    pub user_id: String,
    pub timestamp: String,
    pub language: Option<String>,
    pub finalized: bool,
}

pub struct FunctionCallData {
    pub id: String,           // "call_abc123"
    pub function_name: String,
    pub arguments: String,    // raw JSON string
}
```

### 1.7 Frame constructors (common)

```rust
Frame::start(StartFrameData { allow_interruptions: true, ..Default::default() })
Frame::end()
Frame::end_with("reason")
Frame::cancel()
Frame::cancel_with("reason")
Frame::stop()
Frame::interruption()
Frame::error("msg", fatal, None)

Frame::input_audio(audio_bytes, 16000, 1)
Frame::output_audio(audio_bytes, 24000, 1)

Frame::llm_text("hello ".to_string())
Frame::llm_context(arc_mutex_context)
Frame::llm_full_response_start()
Frame::llm_full_response_end()

Frame::transcription(TranscriptionData::new("hello", "user1", "2024-01-01"))

Frame::function_call_start()
Frame::function_call_end()
Frame::function_call_in_progress(FunctionCallData { id, function_name, arguments })
Frame::function_call_result(FunctionCallResultData { id, function_name, result })
Frame::function_call_raw_result(FunctionCallRawResultData { id, function_name, raw_data })

Frame::heartbeat(unix_timestamp)
Frame::bot_started_speaking()
Frame::bot_stopped_speaking()
Frame::user_started_speaking()
Frame::user_stopped_speaking()
Frame::vad_user_started_speaking(start_secs, timestamp)
Frame::vad_user_stopped_speaking(stop_secs, timestamp)
```

---

## 2. FrameProcessor

`FrameProcessor` is the universal unit of computation. Every component — LLM handler,
VAD, aggregator, transport, Pipeline itself — is a `FrameProcessor`.

```rust
pub struct FrameProcessor(Arc<Inner>);  // cheap to clone — just an Arc bump
```

### 2.1 Construction

```rust
let processor = FrameProcessor::new(
    "MyProcessor",              // name — appears in logs and metrics
    Box::new(MyHandler {}),     // FrameHandler implementation
    false,                      // enable_direct_mode
);
```

**Direct mode** (`enable_direct_mode: true`): skips the async input/process task loops
and processes frames inline. Used for `PipelineSource`, `PipelineSink`,
`TaskSource`, `TaskSink` — infrastructure nodes that must not add latency.
Regular user processors should use `false`.

### 2.2 FrameHandler trait

```rust
#[async_trait]
pub trait FrameHandler: Send + Sync {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,  // self reference for push_frame
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()>;

    fn can_generate_metrics(&self) -> bool { false }
}
```

A minimal pass-through handler:

```rust
#[async_trait]
impl FrameHandler for MyHandler {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        // Inspect frame:
        match &frame.inner {
            FrameInner::Data(DataFrame::LLMText(text)) => {
                println!("LLM said: {}", text);
                processor.push_frame(frame, direction).await?;
            }
            _ => {
                // Pass everything else through
                processor.push_frame(frame, direction).await?;
            }
        }
        Ok(())
    }
}
```

### 2.3 Two-queue architecture

Each processor has two internal queues processed by two background tokio tasks:

```
External call → queue_frame()
                     │
                     ▼
              input_task (always running)
                     │
          ┌──────────┴────────────┐
       is_system?              is_data?
          │                       │
          ▼                       ▼
    process inline          process_queue
    (system priority)            │
                                 ▼
                          process_task (started on StartFrame)
```

- **System frames** (`is_system() == true`) are processed inline by `input_task`
  and never enter `process_queue` — they are never starved
- **Data + Control frames** go through `process_queue` and are processed by
  `process_task` one at a time

On `InterruptionFrame`, the current `process_task` is aborted and replaced — the
backlog is drained except for uninterruptible frames (`EndFrame`, task-control frames).

### 2.4 Linking

```rust
// Connect two processors in sequence: a → b
processor_a.link(&processor_b);

// pipeline.link() does this automatically for all processors in the chain
```

### 2.5 Pushing frames

```rust
// From inside on_process_frame:
processor.push_frame(frame, FrameDirection::Downstream).await?;
processor.push_frame(frame, FrameDirection::Upstream).await?;

// Broadcast in both directions (assigns paired sibling_ids):
processor.broadcast_frame(Frame::interruption()).await?;
processor.broadcast_interruption().await?;  // broadcast + drain queue

// Push an error upstream:
processor.push_error("something went wrong", false).await?;
processor.push_error("fatal error", true).await?;
```

### 2.6 Event hooks

Register sync callbacks on any processor — useful for logging, metrics, debugging:

```rust
processor.on_before_process_frame(|frame| {
    println!("about to process: {}", frame.name());
});

processor.on_after_process_frame(|frame| {
    println!("finished processing: {}", frame.name());
});

processor.on_before_push_frame(|frame| { /* before forwarding */ });
processor.on_after_push_frame(|frame| { /* after forwarding */ });

processor.on_error(|err| {
    eprintln!("error from {}: {}", err.processor_name.as_deref().unwrap_or("?"), err.error);
});
```

### 2.7 Pause / resume

```rust
// Pause data processing (system frames still flow):
processor.pause_processing_frames().await;
processor.resume_processing_frames().await;

// Pause everything including system frames:
processor.pause_processing_system_frames().await;
processor.resume_processing_system_frames().await;

// Via frame (targets by name — goes through normal frame routing):
processor.push_frame(Frame::pause_processor("MyProcessor"), FrameDirection::Downstream).await?;
processor.push_frame(Frame::resume_processor("MyProcessor"), FrameDirection::Downstream).await?;
```

### 2.8 Lifecycle

```rust
// Called by PipelineTask — propagates recursively to all sub-processors:
processor.setup(FrameProcessorSetup { clock, observer }).await?;
processor.cleanup().await?;
```

`setup()` starts the `input_task`. `process_task` is started when `StartFrame`
flows through. `cleanup()` aborts both tasks.

---

## 3. Pipeline

`Pipeline` chains processors into a directed graph and returns a single
`FrameProcessor`. It IS a `FrameProcessor` — it can be nested.

### 3.1 Construction

```rust
let pipeline: FrameProcessor = Pipeline::new(vec![
    stt_processor,
    llm_user_aggregator,
    llm_handler,
    llm_assistant_aggregator,
    tts_processor,
]);
```

### 3.2 Internal topology

```
Pipeline (outer FrameProcessor)
  │
  │  Downstream ──► PipelineSource ──► p1 ──► p2 ──► ... ──► PipelineSink
  │  Upstream   ◄── PipelineSource ◄── p1 ◄── p2 ◄── ... ◄── PipelineSink
```

- **PipelineSource** (direct mode) — downstream: pass through. Upstream: frame
  escaped past the entry; forwarded to outer pipeline's upstream neighbour
- **PipelineSink** (direct mode) — upstream: pass through. Downstream: frame
  escaped past the exit; forwarded to outer pipeline's downstream neighbour
- **PipelineHandler** — routes incoming downstream frames to Source, incoming
  upstream frames to Sink

### 3.3 Nesting

Because Pipeline returns a `FrameProcessor`, pipelines nest recursively:

```rust
let inner = Pipeline::new(vec![llm_user_agg, llm_handler, llm_assistant_agg]);
let outer = Pipeline::new(vec![stt, inner, tts]);
```

Frames escaping `inner` propagate to `outer`'s neighbours automatically.

### 3.4 setup / cleanup propagation

`pipeline.setup()` recursively calls `setup()` on all sub-processors
(Source + user processors + Sink). No manual wiring needed.

---

## 4. PipelineTask

`PipelineTask` wraps a `Pipeline` with:
- External frame injection (mpsc channel)
- `StartFrame` emission on run
- Lifecycle callbacks (started, finished, error, frame boundary, idle timeout)
- Optional heartbeat frames
- Optional idle timeout with auto-cancel

### 4.1 Construction

```rust
let params = PipelineParams {
    allow_interruptions: true,
    enable_metrics: false,
    enable_usage_metrics: false,
    report_only_initial_ttfb: false,
    enable_heartbeats: false,
    heartbeat_seconds: 1.0,
    idle_timeout: Some(Duration::from_secs(30)),
    cancel_on_idle_timeout: true,
    idle_timeout_frames: {
        let mut s = HashSet::new();
        s.insert(FrameKind::BotSpeaking);
        s.insert(FrameKind::UserSpeaking);
        s
    },
};

let task = PipelineTask::new(processors, params);
```

### 4.2 Internal topology

```
push_sender() ──► mpsc ──► [TaskSource] ──► user processors ──► [TaskSink]
                                ↑                                     │
                          intercepts US:                        intercepts DS:
                          EndTask→EndFrame                    StartFrame→on_started
                          CancelTask→CancelFrame              EndFrame→on_finished
                          StopTask→StopFrame                  StopFrame→on_finished
                          InterruptionTask→broadcast          CancelFrame→on_finished
                          idle_timeout_frames→reset timer     ErrorFrame→on_error
```

**TaskSource** and **TaskSink** are direct-mode processors, so they process frames
inline without adding queue latency.

### 4.3 Callbacks

Register all callbacks **before** calling `run()`:

```rust
task.add_on_pipeline_started(|frame| Box::pin(async move {
    println!("pipeline started: {:?}", frame.name());
}));

task.add_on_pipeline_finished(|frame, reason| Box::pin(async move {
    println!("pipeline finished: {:?}", reason);
}));

task.add_on_pipeline_error(|err| Box::pin(async move {
    eprintln!("error: {} fatal={}", err.error, err.fatal);
}));

task.add_on_frame_reached_upstream(|frame| Box::pin(async move {
    // fires only for frames in the upstream_filter
}));

task.add_on_frame_reached_downstream(|frame| Box::pin(async move {
    // fires only for frames in the downstream_filter
}));

task.add_on_idle_timeout(|| Box::pin(async move {
    println!("no activity for idle_timeout duration");
}));

// Set which frame kinds trigger on_frame_reached_upstream:
task.set_upstream_filter(HashSet::from([FrameKind::Transcription]));
task.set_downstream_filter(HashSet::from([FrameKind::OutputAudioRaw]));
```

### 4.4 FinishReason

```rust
pub enum FinishReason {
    End,                   // EndFrame received — graceful shutdown
    Stop,                  // StopFrame — connections kept alive (hand-off)
    Cancel(Option<String>) // CancelFrame or idle timeout
}
```

### 4.5 Running

```rust
// Get sender BEFORE run() — the receiver is taken inside run()
let tx = task.push_sender();

// Inject frames from your transport
tokio::spawn(async move {
    tx.send((Frame::input_audio(bytes, 16000, 1), FrameDirection::Downstream)).await.ok();
});

// Blocks until Finished
task.run(clock, observer).await?;
```

### 4.6 Lifecycle watch

```rust
let mut rx = task.lifecycle_receiver(); // clone before run()
tokio::spawn(async move {
    while rx.changed().await.is_ok() {
        match &*rx.borrow() {
            PipelineLifecycle::NotStarted => {}
            PipelineLifecycle::Running => println!("pipeline is running"),
            PipelineLifecycle::Finished(r) => {
                println!("finished: {:?}", r);
                break;
            }
        }
    }
});
```

### 4.7 Heartbeats

```rust
PipelineParams {
    enable_heartbeats: true,
    heartbeat_seconds: 1.0, // HeartbeatFrame pushed every second
    ..Default::default()
}
```

---

## 5. LLMContext

Shared conversation state owned jointly by the two aggregators and read by the
LLM handler. Wrapped in `Arc<Mutex<LLMContext>>` for safe concurrent access.

```rust
pub struct LLMContext {
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Option<ToolsSchema>,
    pub tool_choice: Option<ToolChoice>,
}

pub enum Message {
    System { content: String },
    User { content: String },
    Assistant {
        content: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
    },
    ToolResult {
        tool_call_id: String,
        content: String,
    },
}
```

### 5.1 Construction

```rust
// Minimal
let ctx = shared_context(Some("You are a helpful assistant.".to_string()));

// With tools
let ctx = shared_context_with_tools(Some(system_prompt), tools_schema, None);
```

### 5.2 Mutation API

```rust
let mut ctx = context.lock().unwrap();

ctx.add_user_message("hello");
ctx.add_assistant_message("hi there");
ctx.add_assistant_tool_calls(None, vec![tool_call]);
ctx.add_tool_result("call_abc123", r#"{"result": "done"}"#);
ctx.push_message(Message::System { content: "...".to_string() });
```

### 5.3 Token budget management

```rust
// Rough estimate of current token usage (chars ÷ 4)
let estimated = ctx.estimate_tokens();

// Trim oldest conversation groups to fit context window
// Reserves 20% headroom for model reply
// Never orphans ToolResult from its paired Assistant tool_calls
ctx.trim_to_context_budget(128_000);
```

### 5.4 API serialisation

```rust
// Called by OpenAILLMHandler before each inference call
let messages: Vec<Message> = ctx.to_api_messages();
// Prepends system prompt as first Message::System
```

---

## 6. Processors

Standard processors that live in `src/processors/`.

### 6.1 LLMUserAggregator

Collects `TranscriptionFrame`s during a user turn, then pushes
`LLMContextFrame` downstream to trigger inference.

```rust
let agg = LLMUserAggregator::new(context.clone());
```

**Trigger conditions:**
- `VADUserStoppedSpeaking` arrives and aggregation is non-empty (normal fast path)
- `TranscriptionFrame` arrives while `user_speaking == false` (transcript arrived
  after VAD stop — race condition fix)

**Interruption:** on `VADUserStartedSpeaking`, if `allow_interruptions` is set,
broadcasts `InterruptionFrame` in both directions so bot speech stops immediately.

**TranscriptionFrames are consumed** — not forwarded downstream.

### 6.2 LLMAssistantAggregator

Collects `LLMTextFrame`s between `LLMFullResponseStart` and `LLMFullResponseEnd`,
saves the completed assistant message to shared `LLMContext`.

```rust
let agg = LLMAssistantAggregator::new(context.clone());
```

**LLMTextFrames are passed through** — TTS needs each chunk for streaming synthesis.

**On interruption:** partial aggregation is discarded. The model "never said" the
interrupted portion — it is not added to context.

### 6.3 Typical pipeline order

```
Transport
  ↓ InputAudioRaw
VAD / STT
  ↓ Transcription + speaking signals
LLMUserAggregator
  ↓ LLMContextFrame
OpenAILLMHandler
  ↓ LLMText + FunctionCall* frames
LLMAssistantAggregator
  ↓ LLMText (passed through)
TTS
  ↓ OutputAudioRaw
Transport
```

---

## 7. Services — LLM

### 7.1 OpenAILLMConfig

```rust
pub struct OpenAILLMConfig {
    pub api_key: String,
    pub model: String,                         // default: "gpt-4.1"
    pub base_url: String,                      // default: OpenAI API
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<i64>,
    pub max_completion_tokens: Option<u32>,
    pub service_tier: Option<String>,
    pub max_tool_rounds: usize,                // default: 5
    pub context_window_tokens: Option<usize>, // None → hardcoded table
}
```

`config.resolve_context_window_tokens()` — returns the effective limit:
explicit override → hardcoded table → `None` (no trimming).

Hardcoded defaults: `gpt-4.1` → 1M, `gpt-4o` → 128k, `claude-opus-4/sonnet-4` → 1M,
`claude-3*` → 200k, `gemini-2/1.5` → 1M.

### 7.2 OpenAILLMHandler — construction variants

```rust
// No tools
let handler = OpenAILLMHandler::new(config);

// Pre-built owned registry
let handler = OpenAILLMHandler::with_registry(config, registry);

// Shared registry (used with Dhara — swapped on node transitions)
let handler = OpenAILLMHandler::with_shared_registry(config, arc_mutex_registry);
```

### 7.3 Adding tools

```rust
let pg = Arc::new(NeonPostgresTool::from_env());
handler.add_tool(pg);

// Collect schemas for ToolsSchema construction:
let schemas = handler.collect_tool_schemas();
```

### 7.4 Transition hook (Dhara integration)

```rust
handler.set_transition_hook(my_hook);
// or get the slot to pass to DharaManager:
let slot = handler.transition_hook_slot();
```

### 7.5 Inference loop

Called when `LLMContextFrame` arrives:

```
1. trim context to budget (if context_window_tokens is known)
2. run_stream() → SSE streaming call to chat completions API
3. if InferenceOutcome::Text → push LLMFullResponseEnd, done
4. if InferenceOutcome::ToolCalls(calls):
   a. append Assistant {tool_calls} to context
   b. push FunctionCallStart
   c. for each call: push FunctionCallInProgress, execute handler,
      push FunctionCallRawResult (if data handler), push FunctionCallResult,
      append ToolResult to context
   d. push FunctionCallEnd
   e. fire transition hook (Dhara may swap context/registry here)
   f. increment round, goto 1 (up to max_tool_rounds)
```

### 7.6 FunctionRegistry

```rust
pub enum RegistryHandler {
    Simple(Arc<dyn Fn(String) -> BoxFuture<'static, String> + Send + Sync>),
    Data(Arc<dyn Fn(String) -> BoxFuture<'static, ToolCallOutput> + Send + Sync>),
}

pub struct ToolCallOutput {
    pub summary: String,       // → LLMContext (model sees this)
    pub full_data: Option<Value>, // → FunctionCallRawResultFrame (model never sees)
}
```

- **Simple handler**: returns a string summary directly to LLM context
- **Data handler**: returns summary + full raw data; raw data goes to
  `FunctionCallRawResultFrame` for downstream consumers (loggers, UI)

---

## 8. Services — STT / TTS

Both STT and TTS services implement `FrameHandler` and are placed in the
processor chain like any other processor.

### 8.1 STT (Speech-to-Text)

- Receives `InputAudioRawFrame` downstream
- Pushes `TranscriptionFrame` upstream when recognition completes
- Pushes `UserStartedSpeaking` / `UserStoppedSpeaking` upstream on VAD events

### 8.2 TTS (Text-to-Speech)

- Receives `LLMTextFrame` downstream (streaming tokens)
- Optionally buffers into sentences before synthesising (see `SentenceSplitter`)
- Pushes `OutputAudioRawFrame` downstream
- Pushes `BotStartedSpeaking` / `BotStoppedSpeaking` upstream

### 8.3 Providers

| Service | Providers |
|---|---|
| STT | Sarvam (`src/services/stt/sarvam.rs`) |
| TTS | Sarvam (`src/services/tts/sarvam.rs`), Deepgram (`deepgram.rs`), Piper (`piper.rs`) |
| LLM | OpenAI-compatible (`src/services/llm/openai.rs`), Sarvam (`sarvam.rs`) |

---

## 9. Built-in Tools

### 9.1 BuiltinTool trait

```rust
#[async_trait]
pub trait BuiltinTool: Send + Sync {
    fn name(&self) -> &str;
    fn is_cacheable(&self) -> bool { false }

    async fn on_start(&self, cancel: CancellationToken) -> Result<()> { Ok(()) }
    async fn on_stop(&self) -> Result<()> { Ok(()) }
    async fn on_cancel(&self) -> Result<()> { self.on_stop().await }

    fn tool_schemas(&self) -> Vec<FunctionSchema>;
    fn register_all(&self, registry: &mut FunctionRegistry);
}
```

### 9.2 Lifecycle (managed by OpenAILLMHandler)

```
add_tool(tool)              → register_all() called immediately
                              handlers capture Arc<OnceCell<...>> refs

StartFrame arrives          → on_start(child_cancel_token) for cacheable tools
                              connects, introspects, populates caches

Tool calls during inference → handlers fire (registry lookup, execute)

EndFrame arrives            → on_stop() for all tools (flush, return connections)

CancelFrame arrives         → cancel_token.cancel() → background tasks exit
                            → on_cancel() for each tool (abort in-flight, then on_stop)
```

### 9.3 ToolLifecycleState

```rust
pub enum ToolLifecycleState {
    Created,    // constructed, on_start not called
    Started,    // on_start completed
    Stopped,    // on_stop completed
    Cancelled,  // on_cancel was called
}
```

### 9.4 Cacheable vs non-cacheable

- **Cacheable** (`is_cacheable() → true`): needs async init. Example: Postgres
  (connect + schema introspection on StartFrame). `on_start()` must be implemented.
- **Non-cacheable** (`is_cacheable() → false`): ready immediately. Default `on_start()`
  is a no-op. Example: a calculator, datetime tool.

### 9.5 Cancellation pattern (background tasks)

```rust
async fn on_start(&self, cancel: CancellationToken) -> Result<()> {
    self.cancel_token.set(cancel.clone()).ok();

    tokio::spawn(async move {
        tokio::select! {
            _ = cancel.cancelled() => {
                log::info!("background task cancelled");
            }
            result = do_work() => {
                // handle result
            }
        }
    });
    Ok(())
}
```

### 9.6 Postgres tool (built-in)

Registered functions: `pg_schema`, `pg_query`, `pg_refine`, `pg_vector_search`.

- `pg_schema` — returns cached schema (populated on_start)
- `pg_query` — parameterised query, result cached
- `pg_refine` — safe SQL generation from natural language
- `pg_vector_search` — similarity search with scoping

---

## 10. Dhara — Conversation Flows

Dhara (ധാര, "stream") manages multi-step conversation flows: state machines where
each node has its own system prompt, tool set, and context strategy. Transitions
happen when a tool handler returns a node name.

### 10.1 Flow definition — dhara.json

```json
{
  "id": "interview",
  "initial_node": "greeting",
  "nodes": {
    "greeting": {
      "role_messages": [
        { "role": "system", "content": "You are a friendly interviewer. Greet the candidate." }
      ],
      "task_messages": [],
      "functions": ["begin_interview"],
      "context_strategy": "reset"
    },
    "questioning": {
      "role_messages": [
        { "role": "system", "content": "Ask technical questions." }
      ],
      "task_messages": [],
      "functions": ["ask_next_question", "end_interview"],
      "context_strategy": "keep"
    }
  },
  "functions": {
    "begin_interview": {
      "description": "Start the interview",
      "parameters": { "type": "object", "properties": {} },
      "transitions": {
        "success": "questioning"
      }
    },
    "end_interview": {
      "description": "End the interview session",
      "parameters": { "type": "object", "properties": {} },
      "transitions": {}
    }
  }
}
```

**Context strategies:**
- `"reset"` — clear message history on node entry (fresh start)
- `"keep"` — accumulate all history across transitions
- `"task"` — keep only task_messages on entry

### 10.2 Validation (at load time)

Dhara validates the flow graph at construction:
- `initial_node` exists in `nodes`
- Every function referenced by a node exists in `functions`
- Every transition target is a valid node
- All nodes are reachable from `initial_node` (warns on orphans)
- Task message roles are valid

Validation errors are returned as `DharaError::ValidationErrors(Vec<String>)`.

### 10.3 Loading

```rust
// From filesystem (reads dhara/<dir>/dhara.json)
let dhara = Dhara::load("dhara/interview")?;

// From embedded string (zero-file-I/O at runtime)
let dhara = Dhara::from_json(include_str!("../../dhara/interview/dhara.json"))?;
```

### 10.4 Building runtime pieces

```rust
// Register your handler implementations
let mut handlers = DharaFunctionRegistry::new();
handlers.register("begin_interview", |args, ctx| async move {
    // ctx: DharaContext — access to push_sender, custom state, conn_id
    ctx.transition("questioning")   // returns HandlerResult::Transition
});
handlers.register("end_interview", |args, ctx| async move {
    HandlerResult::Ok("interview complete".to_string())
});

// Build per-connection runtime (call once per session)
let built: DharaBuild = dhara.build(&handlers, my_shared_state, conn_id)?;

// built.context  — Arc<Mutex<LLMContext>> for aggregators
// built.registry — Arc<Mutex<FunctionRegistry>> for LLM handler
// built.hook     — TransitionHook for LLM handler
// built.dhara_ctx — DharaContext for handlers (push frames, store state)
```

### 10.5 Wiring into pipeline

```rust
let mut llm = OpenAILLMHandler::with_shared_registry(llm_config, built.llm_registry);
llm.set_transition_hook(built.hook);

let user_agg = LLMUserAggregator::new(built.context.clone());
let asst_agg = LLMAssistantAggregator::new(built.context.clone());

let task = PipelineTask::new(
    vec![transport_in, stt, user_agg, llm.into_processor(), asst_agg, tts, transport_out],
    PipelineParams { allow_interruptions: true, ..Default::default() },
);

// Give DharaContext the push sender so handlers can inject frames
built.dhara_ctx.set_push_sender(task.push_sender());

task.run(clock, None).await?;
```

### 10.6 DharaContext (in handlers)

```rust
// Trigger a node transition
ctx.transition("next_node")

// Return a simple string result (no transition)
HandlerResult::Ok("done".to_string())

// Inject a frame into the pipeline from a handler
ctx.push_frame(Frame::end(), FrameDirection::Downstream).await;

// Access custom shared state
let state = ctx.state::<MyState>();

// Connection ID (for multi-session routing)
let id = ctx.conn_id();
```

### 10.7 Transition mechanism (what happens under the hood)

```
Tool handler returns HandlerResult::Transition("next_node")
  ↓
Transition stored in DharaContext
  ↓
All tool calls in this batch complete
  ↓
TransitionHook fires (set on OpenAILLMHandler)
  ↓
DharaContext applies pending transition:
  - New system prompt loaded (role_messages)
  - Context strategy applied (reset/keep/task)
  - FunctionRegistry rebuilt with next node's functions
  ↓
Inference re-invoked with new context + tools
```

### 10.8 Legacy API (DharaManager)

The legacy imperative API (`DharaManager`) is kept for backward compatibility.
Prefer the declarative JSON API (`Dhara::load()` + `DharaFunctionRegistry`) for
new flows.

---

## Frame flow — end-to-end voice turn

```
User speaks
  transport.push_frame(InputAudioRaw, DS)
    → VAD: VADUserStartedSpeaking (US)
    → STT: accumulates audio
  User stops speaking
    → VAD: VADUserStoppedSpeaking (US)
    → STT: Transcription (US)

LLMUserAggregator (upstream)
  receives VADUserStoppedSpeaking → sets user_speaking=false
  receives Transcription → adds to aggregation
  flushes: adds User message to LLMContext
  pushes LLMContextFrame (DS)

OpenAILLMHandler
  receives LLMContextFrame
  trims context to budget (if known)
  calls OpenAI SSE stream → LLMText chunks (DS), LLMFullResponseStart/End (DS)
  if tool calls: executes, pushes FunctionCall* frames, re-invokes

LLMAssistantAggregator
  collects LLMText between Start/End
  saves complete assistant message to LLMContext
  passes LLMText downstream (TTS needs each chunk)

TTS
  receives LLMText → synthesises → OutputAudioRaw (DS)
  pushes BotStartedSpeaking (US), BotStoppedSpeaking (US)

Transport
  receives OutputAudioRaw → sends to client
```
