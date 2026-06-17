# Sarvam LLM

**File:** `src/services/llm/sarvam.rs`  
**Feature:** `stt-sarvam` (shares feature gate with Sarvam STT; enabled by default)  
**Protocol:** SSE HTTP (`POST https://api.sarvam.ai/v1/chat/completions`)

Sarvam AI's OpenAI-compatible chat completions endpoint. Supports Indian language LLMs and an optional CoT thinking mode.

## Pipeline Position

```
LLMUserAggregator → SarvamLLMHandler → LLMAssistantAggregator
```

## Usage

```rust
use rustvani::services::llm::sarvam::{SarvamLLMConfig, SarvamLLMHandler};

let llm = SarvamLLMHandler::new(SarvamLLMConfig {
    api_key: std::env::var("SARVAM_API_KEY").unwrap(),
    model: "sarvam-30b".to_string(),
    temperature: Some(0.2),
    reasoning_effort: None,  // Set to Some("low") to enable CoT thinking
    ..Default::default()
}).into_processor();
```

## Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `api_key` | `String` | — | Sarvam API subscription key (header: `api-subscription-key`) |
| `model` | `String` | `"sarvam-30b"` | `sarvam-m`, `sarvam-30b`, `sarvam-105b` |
| `base_url` | `String` | `"https://api.sarvam.ai/v1"` | Endpoint base |
| `temperature` | `Option<f32>` | `Some(0.2)` | Sampling temperature |
| `reasoning_effort` | `Option<String>` | `None` | Any value enables CoT thinking (`low`/`medium`/`high`). `None` = fast non-think mode |

## Frames

**Consumed:**
- `LLMContextFrame` → triggers inference

**Produced:**
- `LLMFullResponseStart` → before first token
- `LLMText` → per SSE content chunk
- `LLMFullResponseEnd` → after stream complete or error
- `ErrorFrame` → on HTTP or stream failure

## Tool Calling

Sarvam LLM supports tool definitions via the shared OpenAI adapter. Pass tools in the `LLMContext` the same way as OpenAI:

```rust
use rustvani::context::LLMContext;

let mut ctx = shared_context(Some("You are a helpful assistant.".into()));
ctx.tools = Some(tools_schema);
ctx.tool_choice = Some(tool_choice);
```

## Environment Variables

```bash
SARVAM_API_KEY=your_key
```

## Cargo Feature

Enabled by default (shares `stt-sarvam` feature gate). To disable Sarvam services:

```toml
[dependencies]
rustvani = { version = "0.2", default-features = false, features = ["vad-silero-ort", "transport-websocket", "stt-60db", "llm-openai", "tts-deepgram", "tts-sarvam", "dhara"] }
```
