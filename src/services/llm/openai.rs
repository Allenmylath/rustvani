//! OpenAI LLM service (chat completions, SSE streaming, function calling).
//!
//! Supports:
//!   - Plain text inference
//!   - Tool/function calling with SSE delta accumulation
//!   - Re-invocation loop (model calls tool → execute → re-invoke)
//!   - Dhara transition hooks for conversation flow management
//!   - Built-in tool lifecycle (on_start / on_stop / on_cancel)
//!
//! Pipeline position:
//!   LLMUserAggregator → OpenAILLMHandler → LLMAssistantAggregator
//!
//! Lifecycle:
//!   StartFrame  → initialise cacheable tools (pg connects, caches schema)
//!   frames flow → inference + tool execution
//!   EndFrame    → graceful shutdown (flush caches, return connections)
//!   CancelFrame → cancel token fires, then on_cancel → on_stop

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use async_trait::async_trait;
use futures::StreamExt;
use log;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::adapters::base::LLMAdapter;
use crate::adapters::openai::OpenAILLMAdapter;
use crate::context::{LLMContext, ToolCall};
use crate::error::{PipecatError, Result};
use crate::frames::{
    ControlFrame, DataFrame, Frame, FrameDirection, FunctionCallData,
    FunctionCallRawResultData, FunctionCallResultData, FrameHandler, FrameInner,
    FrameProcessor, SystemFrame,
};
use crate::tools::BuiltinTool;

use super::function_registry::{FunctionRegistry, RegistryHandler};

/// Hook called after all tool calls in a batch complete, before re-invoking
/// inference. Used by DharaManager to apply node transitions.
pub type TransitionHook = Arc<dyn Fn(&Arc<Mutex<LLMContext>>) + Send + Sync>;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct OpenAILLMConfig {
    pub api_key: String,
    pub model: String,
    pub base_url: String,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<i64>,
    pub max_completion_tokens: Option<u32>,
    pub service_tier: Option<String>,
    /// Maximum number of recursive tool call rounds. Prevents infinite loops.
    pub max_tool_rounds: usize,
    /// Context window size for this model in tokens. `None` falls back to
    /// the hardcoded table in `resolve_context_window_tokens()`. Set
    /// explicitly to override for custom or self-hosted models.
    pub context_window_tokens: Option<usize>,
}

/// Hardcoded context windows for common models. Returns `None` for unknowns —
/// callers treat `None` as "no trimming".
fn default_context_tokens(model: &str) -> Option<usize> {
    let m = model.to_lowercase();
    let m = m.split('/').next_back().unwrap_or(&m);
    if m.starts_with("gpt-4.1") { return Some(1_047_576); }
    if m.starts_with("gpt-4o") { return Some(128_000); }
    if m.starts_with("gpt-4-turbo") { return Some(128_000); }
    if m.starts_with("gpt-3.5") { return Some(16_385); }
    if m.starts_with("claude-opus-4") || m.starts_with("claude-sonnet-4") { return Some(1_048_576); }
    if m.starts_with("claude-3") { return Some(200_000); }
    if m.starts_with("gemini-2") || m.starts_with("gemini-1.5") { return Some(1_048_576); }
    None
}

impl OpenAILLMConfig {
    /// Returns the effective context window token limit for this config,
    /// preferring the explicit override then falling back to the model table.
    pub fn resolve_context_window_tokens(&self) -> Option<usize> {
        self.context_window_tokens.or_else(|| default_context_tokens(&self.model))
    }
}

impl Default for OpenAILLMConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: "gpt-4.1".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            max_completion_tokens: None,
            service_tier: None,
            max_tool_rounds: 5,
            context_window_tokens: None,
        }
    }
}

// ---------------------------------------------------------------------------
// OpenAI API wire types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Value>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
}

#[derive(Deserialize)]
struct ChatChunk {
    choices: Vec<ChunkChoice>,
    #[allow(dead_code)]
    usage: Option<Value>,
}

#[derive(Deserialize)]
struct ChunkChoice {
    delta: ChunkDelta,
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ChunkDelta {
    content: Option<String>,
    #[allow(dead_code)]
    role: Option<String>,
    tool_calls: Option<Vec<ChunkToolCall>>,
}

#[derive(Deserialize)]
struct ChunkToolCall {
    index: u32,
    id: Option<String>,
    function: Option<ChunkToolCallFunction>,
}

#[derive(Deserialize)]
struct ChunkToolCallFunction {
    name: Option<String>,
    arguments: Option<String>,
}

// ---------------------------------------------------------------------------
// Tool call accumulator
// ---------------------------------------------------------------------------

struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl PartialToolCall {
    fn into_tool_call(self) -> ToolCall {
        ToolCall {
            id: self.id,
            function_name: self.name,
            arguments: self.arguments,
        }
    }
}

enum InferenceOutcome {
    Text,
    ToolCalls(Vec<ToolCall>),
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

pub struct OpenAILLMHandler {
    config: OpenAILLMConfig,
    client: Client,
    adapter: OpenAILLMAdapter,
    registry: Arc<Mutex<FunctionRegistry>>,
    /// Optional hook called after tool calls complete, before re-invoking.
    /// Used by DharaManager for node transitions.
    transition_hook: Arc<RwLock<Option<TransitionHook>>>,
    /// Built-in tools attached to this handler.
    tools: Vec<Arc<dyn BuiltinTool>>,
    /// Cancellation token — cancelled on CancelFrame, cascades to all tools.
    cancel_token: CancellationToken,
}

impl OpenAILLMHandler {
    /// Create a handler with an empty, owned registry (no tools).
    pub fn new(config: OpenAILLMConfig) -> Self {
        Self {
            config,
            client: Client::new(),
            adapter: OpenAILLMAdapter::new(),
            registry: Arc::new(Mutex::new(FunctionRegistry::new())),
            transition_hook: Arc::new(RwLock::new(None)),
            tools: Vec::new(),
            cancel_token: CancellationToken::new(),
        }
    }

    /// Create a handler with a pre-built registry (simple tool calling, no dhara).
    pub fn with_registry(config: OpenAILLMConfig, registry: FunctionRegistry) -> Self {
        Self {
            config,
            client: Client::new(),
            adapter: OpenAILLMAdapter::new(),
            registry: Arc::new(Mutex::new(registry)),
            transition_hook: Arc::new(RwLock::new(None)),
            tools: Vec::new(),
            cancel_token: CancellationToken::new(),
        }
    }

    /// Create a handler with a shared registry (used by DharaManager).
    ///
    /// The `Arc<Mutex<FunctionRegistry>>` is shared with DharaManager,
    /// which swaps its contents on node transitions.
    pub fn with_shared_registry(
        config: OpenAILLMConfig,
        registry: Arc<Mutex<FunctionRegistry>>,
    ) -> Self {
        Self {
            config,
            client: Client::new(),
            adapter: OpenAILLMAdapter::new(),
            registry,
            transition_hook: Arc::new(RwLock::new(None)),
            tools: Vec::new(),
            cancel_token: CancellationToken::new(),
        }
    }

    
    pub fn set_transition_hook(&self, hook: TransitionHook) {
    *self.transition_hook.write().unwrap() = Some(hook);
    }

    pub fn transition_hook_slot(&self) -> Arc<RwLock<Option<TransitionHook>>> {
        self.transition_hook.clone()
    }

    /// Attach a built-in tool.
    ///
    /// Registers the tool's handlers into the shared registry immediately.
    /// Handlers capture `Arc<OnceCell<...>>` refs — the actual resources
    /// (connections, caches) are populated later in `on_start()` when
    /// `StartFrame` flows through.
    ///
    /// # Example
    /// ```rust,ignore
    /// let pg = Arc::new(NeonPostgresTool::from_env());
    /// handler.add_tool(pg);
    /// ```
    pub fn add_tool(&mut self, tool: Arc<dyn BuiltinTool>) {
        log::info!("OpenAILLM: attaching tool '{}'", tool.name());
        tool.register_all(&mut self.registry.lock().unwrap());
        self.tools.push(tool);
    }

    /// Get the tool schemas from all attached tools.
    ///
    /// Convenience for building `ToolsSchema` at pipeline setup:
    /// ```rust,ignore
    /// let schemas = handler.collect_tool_schemas();
    /// let tools = ToolsSchema::new(schemas);
    /// ```
    pub fn collect_tool_schemas(&self) -> Vec<crate::adapters::schemas::FunctionSchema> {
        self.tools.iter().flat_map(|t| t.tool_schemas()).collect()
    }

    pub fn into_processor(self) -> FrameProcessor {
        FrameProcessor::new("OpenAILLM", Box::new(self), false)
    }

    // -----------------------------------------------------------------------
    // Lifecycle helpers
    // -----------------------------------------------------------------------

    /// Initialise all cacheable tools. Called on StartFrame.
    async fn start_tools(&self) {
        for tool in &self.tools {
            if tool.is_cacheable() {
                let child = self.cancel_token.child_token();
                log::info!("OpenAILLM: starting tool '{}'...", tool.name());
                if let Err(e) = tool.on_start(child).await {
                    log::error!(
                        "OpenAILLM: tool '{}' failed to start: {}",
                        tool.name(), e
                    );
                }
            }
        }
    }

    /// Gracefully stop all tools. Called on EndFrame.
    async fn stop_tools(&self) {
        for tool in &self.tools {
            log::debug!("OpenAILLM: stopping tool '{}'...", tool.name());
            if let Err(e) = tool.on_stop().await {
                log::error!(
                    "OpenAILLM: tool '{}' failed to stop: {}",
                    tool.name(), e
                );
            }
        }
    }

    /// Cancel all tools. Called on CancelFrame.
    async fn cancel_tools(&self) {
        // 1. Trip the cancellation token — background tasks exit via select!
        self.cancel_token.cancel();

        // 2. Give each tool a chance to do tool-specific cancellation
        //    (e.g. cancel in-flight postgres queries), then on_stop()
        for tool in &self.tools {
            log::debug!("OpenAILLM: cancelling tool '{}'...", tool.name());
            if let Err(e) = tool.on_cancel().await {
                log::error!(
                    "OpenAILLM: tool '{}' cancel failed: {}",
                    tool.name(), e
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // SSE streaming
    // -----------------------------------------------------------------------

    /// Run a single SSE stream.
    async fn run_stream(
        &self,
        context: &Arc<Mutex<LLMContext>>,
        processor: &FrameProcessor,
    ) -> Result<InferenceOutcome> {
        let (api_messages, tools, tool_choice) = {
            let ctx = context.lock().unwrap();
            let messages = ctx.to_api_messages();
            let converted = self.adapter.convert_messages(&messages);
            let tools = ctx.tools.as_ref().map(|t| self.adapter.to_provider_tools_format(t));
            let tool_choice = ctx.tool_choice.as_ref().map(|tc| self.adapter.to_provider_tool_choice(tc));
            (converted, tools, tool_choice)
        };

        let url = format!("{}/chat/completions", self.config.base_url);
        log::info!(
            "OpenAILLM: {} messages -> {} (model={})",
            api_messages.len(), url, self.config.model
        );

        let body = ChatRequest {
            model: self.config.model.clone(),
            messages: api_messages,
            stream: true,
            temperature: self.config.temperature,
            top_p: self.config.top_p,
            frequency_penalty: self.config.frequency_penalty,
            presence_penalty: self.config.presence_penalty,
            seed: self.config.seed,
            max_completion_tokens: self.config.max_completion_tokens,
            service_tier: self.config.service_tier.clone(),
            tools,
            tool_choice,
        };

        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| PipecatError::pipeline(format!("OpenAILLM: request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(PipecatError::pipeline(
                format!("OpenAILLM: HTTP {} — {}", status, body),
            ));
        }

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut tool_accum: HashMap<u32, PartialToolCall> = HashMap::new();

        'outer: while let Some(chunk) = stream.next().await {
            let bytes = chunk.map_err(|e| {
                PipecatError::pipeline(format!("OpenAILLM: stream read error: {}", e))
            })?;
            buffer.push_str(&String::from_utf8_lossy(&bytes));

            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].trim_end_matches('\r').trim().to_string();
                buffer = buffer[pos + 1..].to_string();

                if line.is_empty() {
                    continue;
                }

                let data = match line.strip_prefix("data: ") {
                    Some(d) => d,
                    None => continue,
                };

                if data == "[DONE]" {
                    log::debug!("OpenAILLM: stream complete");
                    break 'outer;
                }

                match serde_json::from_str::<ChatChunk>(data) {
                    Ok(chunk) => {
                        if let Some(choice) = chunk.choices.first() {
                            if let Some(content) = &choice.delta.content {
                                if !content.is_empty() {
                                    processor.push_frame(
                                        Frame::llm_text(content.clone()),
                                        FrameDirection::Downstream,
                                    ).await?;
                                }
                            }
                            if let Some(tool_calls) = &choice.delta.tool_calls {
                                for tc in tool_calls {
                                    let entry = tool_accum.entry(tc.index).or_insert_with(|| {
                                        PartialToolCall {
                                            id: String::new(),
                                            name: String::new(),
                                            arguments: String::new(),
                                        }
                                    });
                                    if let Some(id) = &tc.id {
                                        entry.id = id.clone();
                                    }
                                    if let Some(func) = &tc.function {
                                        if let Some(name) = &func.name {
                                            entry.name = name.clone();
                                        }
                                        if let Some(args) = &func.arguments {
                                            entry.arguments.push_str(args);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("OpenAILLM: chunk parse error: {} — raw: {}", e, data);
                    }
                }
            }
        }

        if tool_accum.is_empty() {
            Ok(InferenceOutcome::Text)
        } else {
            let mut calls: Vec<(u32, PartialToolCall)> = tool_accum.into_iter().collect();
            calls.sort_by_key(|(idx, _)| *idx);
            let tool_calls: Vec<ToolCall> =
                calls.into_iter().map(|(_, tc)| tc.into_tool_call()).collect();
            log::info!(
                "OpenAILLM: model requested {} tool call(s): [{}]",
                tool_calls.len(),
                tool_calls
                    .iter()
                    .map(|tc| tc.function_name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            Ok(InferenceOutcome::ToolCalls(tool_calls))
        }
    }

    // -----------------------------------------------------------------------
    // Inference with tool execution loop
    // -----------------------------------------------------------------------

    /// Full inference with tool execution and re-invocation loop.
    async fn run_inference(
        &self,
        context: Arc<Mutex<LLMContext>>,
        processor: &FrameProcessor,
    ) -> Result<()> {
        let mut round = 0;

        loop {
            if round >= self.config.max_tool_rounds {
                log::warn!(
                    "OpenAILLM: max tool rounds ({}) reached",
                    self.config.max_tool_rounds
                );
                break;
            }
            round += 1;

            if let Some(budget) = self.config.resolve_context_window_tokens() {
                context.lock().unwrap().trim_to_context_budget(budget);
            }

            match self.run_stream(&context, processor).await? {
                InferenceOutcome::Text => break,
                InferenceOutcome::ToolCalls(tool_calls) => {
                    // Append assistant message with tool_calls
                    context
                        .lock()
                        .unwrap()
                        .add_assistant_tool_calls(None, tool_calls.clone());

                    processor
                        .push_frame(Frame::function_call_start(), FrameDirection::Downstream)
                        .await?;

                    for tc in &tool_calls {
                        processor
                            .push_frame(
                                Frame::function_call_in_progress(FunctionCallData {
                                    id: tc.id.clone(),
                                    function_name: tc.function_name.clone(),
                                    arguments: tc.arguments.clone(),
                                }),
                                FrameDirection::Downstream,
                            )
                            .await?;

                        // Look up handler in the shared registry
                        let handler = {
                            let reg = self.registry.lock().unwrap();
                            reg.get(&tc.function_name).cloned()
                        };

                        // Execute handler — resolve to (summary, Option<raw_data>)
                        let (summary, raw_data) = match handler {
                            Some(RegistryHandler::Simple(f)) => {
                                log::info!(
                                    "OpenAILLM: executing simple '{}' (id={})",
                                    tc.function_name, tc.id
                                );
                                let result = f(tc.arguments.clone()).await;
                                (result, None)
                            }
                            Some(RegistryHandler::Data(f)) => {
                                log::info!(
                                    "OpenAILLM: executing data '{}' (id={})",
                                    tc.function_name, tc.id
                                );
                                let output = f(tc.arguments.clone()).await;
                                (output.summary, output.full_data)
                            }
                            None => {
                                log::warn!(
                                    "OpenAILLM: no handler for '{}'",
                                    tc.function_name
                                );
                                (
                                    format!(
                                        "{{\"error\": \"function '{}' is not registered\"}}",
                                        tc.function_name
                                    ),
                                    None,
                                )
                            }
                        };

                        // Raw data frame downstream (UI/logging) — LLM never sees this
                        if let Some(data) = raw_data {
                            processor
                                .push_frame(
                                    Frame::function_call_raw_result(FunctionCallRawResultData {
                                        id: tc.id.clone(),
                                        function_name: tc.function_name.clone(),
                                        raw_data: data,
                                    }),
                                    FrameDirection::Downstream,
                                )
                                .await?;
                        }

                        // Summary result frame — LLM sees this on next round
                        processor
                            .push_frame(
                                Frame::function_call_result(FunctionCallResultData {
                                    id: tc.id.clone(),
                                    function_name: tc.function_name.clone(),
                                    result: summary.clone(),
                                }),
                                FrameDirection::Downstream,
                            )
                            .await?;

                        // Only summary goes into LLM context
                        context.lock().unwrap().add_tool_result(&tc.id, &summary);
                    }

                    processor
                        .push_frame(Frame::function_call_end(), FrameDirection::Downstream)
                        .await?;

                    // --- Transition hook ---
                    if let Some(hook) = self.transition_hook.read().unwrap().as_ref() {
                        hook(&context);
                    }

                    log::info!("OpenAILLM: re-invoking inference (round {})", round + 1);
                }
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FrameHandler
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameHandler for OpenAILLMHandler {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match &frame.inner {
            // ----- Lifecycle: StartFrame -----
            // Initialise cacheable tools (connect, cache schemas, etc.)
            FrameInner::System(SystemFrame::Start(_)) => {
                log::info!("OpenAILLM: StartFrame — initialising tools...");
                self.start_tools().await;
                processor.push_frame(frame, direction).await?;
            }

            // ----- Lifecycle: EndFrame -----
            // Graceful shutdown — flush caches, return connections
            FrameInner::Control(ControlFrame::End { .. }) => {
                log::info!("OpenAILLM: EndFrame — stopping tools...");
                self.stop_tools().await;
                processor.push_frame(frame, direction).await?;
            }

            // ----- Lifecycle: CancelFrame -----
            // Abrupt shutdown — cancel in-flight work, then stop
            FrameInner::System(SystemFrame::Cancel { .. }) => {
                log::warn!("OpenAILLM: CancelFrame — cancelling tools...");
                self.cancel_tools().await;
                processor.push_frame(frame, direction).await?;
            }

            // ----- Inference trigger -----
            FrameInner::Data(DataFrame::LLMContextFrame(context)) => {
                let context = context.clone();
                processor
                    .push_frame(
                        Frame::llm_full_response_start(),
                        FrameDirection::Downstream,
                    )
                    .await?;
                if let Err(e) = self.run_inference(context, processor).await {
                    log::error!("OpenAILLM: inference error: {}", e);
                    processor.push_error(e.to_string(), false).await?;
                }
                processor
                    .push_frame(
                        Frame::llm_full_response_end(),
                        FrameDirection::Downstream,
                    )
                    .await?;
            }

            // ----- Pass-through -----
            _ => {
                processor.push_frame(frame, direction).await?;
            }
        }

        Ok(())
    }

    fn can_generate_metrics(&self) -> bool {
        true
    }
}
