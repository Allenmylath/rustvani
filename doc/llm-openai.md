# OpenAI LLM

**File:** `src/services/llm/openai.rs`  
**Feature:** `llm-openai` (enabled by default)  
**Protocol:** SSE HTTP (`POST /chat/completions`)

OpenAI-compatible chat completions with SSE streaming, full function calling, re-invocation loop, and Dhara transition hooks. Works with any OpenAI-compatible endpoint (OpenAI, Azure, self-hosted, etc.).

## Pipeline Position

```
LLMUserAggregator → OpenAILLMHandler → LLMAssistantAggregator
```

## Usage

### Basic

```rust
use rustvani::services::llm::openai::{OpenAILLMConfig, OpenAILLMHandler};

let llm = OpenAILLMHandler::new(OpenAILLMConfig {
    api_key: std::env::var("OPENAI_API_KEY").unwrap(),
    model: "gpt-4.1".to_string(),
    ..Default::default()
}).into_processor();
```

### With Function Registry

```rust
use rustvani::services::llm::function_registry::FunctionRegistry;

let mut registry = FunctionRegistry::new();
registry.register("get_weather", |args: String| async move {
    format!("Weather: 28°C")
});

let llm = OpenAILLMHandler::with_registry(config, registry).into_processor();
```

### With Built-in Postgres Tool

```rust
use rustvani::tools::postgres::NeonPostgresTool;

let mut llm = OpenAILLMHandler::new(config);
llm.add_tool(Arc::new(NeonPostgresTool::from_env()));
let processor = llm.into_processor();
```

## Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `api_key` | `String` | — | API key (header: `Authorization: Bearer …`) |
| `model` | `String` | `"gpt-4.1"` | Model identifier |
| `base_url` | `String` | `"https://api.openai.com/v1"` | Endpoint base |
| `temperature` | `Option<f32>` | `None` | Sampling temperature |
| `top_p` | `Option<f32>` | `None` | Nucleus sampling |
| `frequency_penalty` | `Option<f32>` | `None` | Frequency penalty |
| `presence_penalty` | `Option<f32>` | `None` | Presence penalty |
| `seed` | `Option<i64>` | `None` | Deterministic seed |
| `max_completion_tokens` | `Option<u32>` | `None` | Max output tokens |
| `service_tier` | `Option<String>` | `None` | OpenAI service tier |
| `max_tool_rounds` | `usize` | `5` | Max recursive tool loops |
| `context_window_tokens` | `Option<usize>` | `None` | Override context limit |

## Context Window Defaults

| Model Family | Tokens |
|---|---|
| `gpt-4.1*` | 1,047,576 |
| `gpt-4o*` / `gpt-4-turbo*` | 128,000 |
| `gpt-3.5*` | 16,385 |
| `claude-opus-4*` / `claude-sonnet-4*` | 1,048,576 |
| `claude-3*` | 200,000 |
| `gemini-2*` / `gemini-1.5*` | 1,048,576 |

## Frames

**Consumed:**
- `StartFrame` → initializes cacheable tools (`on_start`)
- `LLMContextFrame` → triggers inference
- `EndFrame` → graceful shutdown (`on_stop`)
- `CancelFrame` → cancels in-flight work (`on_cancel`)

**Produced:**
- `LLMFullResponseStart` → before first token
- `LLMText` → per SSE content chunk
- `LLMFullResponseEnd` → after stream complete
- `FunctionCallStart` / `FunctionCallInProgress` / `FunctionCallResult` / `FunctionCallRawResult` / `FunctionCallEnd` → tool lifecycle
- `ErrorFrame` → on HTTP or stream failure

## Tool Calling Flow

1. Model requests tool calls → `FunctionCallStart`
2. Each tool executes → `FunctionCallInProgress`
3. Raw data (for UI) → `FunctionCallRawResult`
4. Summary (for model) → `FunctionCallResult`
5. All done → `FunctionCallEnd`
6. Context updated → model re-invoked (up to `max_tool_rounds`)

## Dhara Integration

```rust
let mut dhara = DharaManager::new(context.clone(), registry.clone());
dhara.register_node("greeting", greeting_node, vec![...]);

let mut llm = OpenAILLMHandler::new(config);
llm.set_transition_hook(dhara.create_transition_hook());
```

## Environment Variables

```bash
OPENAI_API_KEY=your_key
DATABASE_URL=postgres://…  # if using NeonPostgresTool
```

## Cargo Feature

Enabled by default. To disable:

```toml
[dependencies]
rustvani = { version = "0.2", default-features = false, features = ["vad-silero-ort", "transport-websocket", "stt-sarvam", "stt-60db", "tts-deepgram", "tts-sarvam", "dhara"] }
```
