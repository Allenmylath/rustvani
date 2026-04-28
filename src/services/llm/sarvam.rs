//! Sarvam LLM service.
//!
//! Direct HTTP to https://api.sarvam.ai/v1/chat/completions.
//! Uses the OpenAI adapter for message/tool conversion since Sarvam's API
//! is OpenAI-compatible.
//!
//! Pipeline position:
//!   LLMUserAggregator → SarvamLLMHandler → LLMAssistantAggregator
//!
//! Frames consumed:
//!   - LLMContextFrame → triggers inference
//!
//! Frames produced:
//!   - LLMFullResponseStartFrame (before first token)
//!   - LLMTextFrame              (one per SSE content chunk)
//!   - LLMFullResponseEndFrame   (after [DONE] or on error)
//!   - ErrorFrame                (on HTTP or stream failure)

use async_trait::async_trait;
use futures::StreamExt;
use log;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::adapters::base::LLMAdapter;
use crate::adapters::openai::OpenAILLMAdapter;
use crate::context::LLMContext;
use crate::error::{PipecatError, Result};
use crate::frames::{
    DataFrame, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor,
};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SarvamLLMConfig {
    pub api_key: String,
    /// e.g. "sarvam-m", "sarvam-30b", "sarvam-105b"
    pub model: String,
    pub base_url: String,
    pub temperature: Option<f32>,
    /// Controls CoT thinking mode. Any value ("low"/"medium"/"high") enables
    /// thinking. Set to None to use non-think mode (fast, no <think> block).
    /// Recommended: None for voice pipelines.
    pub reasoning_effort: Option<String>,
}

impl Default for SarvamLLMConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: "sarvam-30b".to_string(),
            base_url: "https://api.sarvam.ai/v1".to_string(),
            temperature: Some(0.2),
            reasoning_effort: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Sarvam API wire types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    /// Messages as provider-formatted JSON (produced by the adapter).
    messages: Vec<Value>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    /// Omit entirely for non-think mode. Any value enables CoT thinking.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    /// Tool definitions. Omitted when no tools are configured.
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Value>>,
    /// Tool choice. Omitted when no tools are configured.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
}

#[derive(Deserialize)]
struct ChatChunk {
    choices: Vec<ChunkChoice>,
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
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

pub struct SarvamLLMHandler {
    config: SarvamLLMConfig,
    client: Client,
    adapter: OpenAILLMAdapter,
}

impl SarvamLLMHandler {
    pub fn new(config: SarvamLLMConfig) -> Self {
        Self {
            config,
            client: Client::new(),
            adapter: OpenAILLMAdapter::new(),
        }
    }

    pub fn into_processor(self) -> FrameProcessor {
        FrameProcessor::new("SarvamLLM", Box::new(self), false)
    }

    /// POST to Sarvam, parse SSE stream, push LLMTextFrames downstream.
    async fn run_inference(
        &self,
        context: std::sync::Arc<std::sync::Mutex<LLMContext>>,
        processor: &FrameProcessor,
    ) -> Result<()> {
        // Lock context, extract what we need, release lock immediately
        let (api_messages, tools, tool_choice) = {
            let ctx = context.lock().unwrap();
            let messages = ctx.to_api_messages();

            // Convert through the adapter (same format as OpenAI)
            let converted = self.adapter.convert_messages(&messages);

            let tools = ctx.tools.as_ref().map(|t| {
                self.adapter.to_provider_tools_format(t)
            });

            let tool_choice = ctx.tool_choice.as_ref().map(|tc| {
                self.adapter.to_provider_tool_choice(tc)
            });

            (converted, tools, tool_choice)
        };

        let url = format!("{}/chat/completions", self.config.base_url);

        log::info!(
            "SarvamLLM: {} messages → {} (model={}, reasoning_effort={:?})",
            api_messages.len(),
            url,
            self.config.model,
            self.config.reasoning_effort,
        );

        let body = ChatRequest {
            model: self.config.model.clone(),
            messages: api_messages,
            stream: true,
            temperature: self.config.temperature,
            reasoning_effort: self.config.reasoning_effort.clone(),
            tools,
            tool_choice,
        };

        let response = self
            .client
            .post(&url)
            .header("api-subscription-key", &self.config.api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| PipecatError::pipeline(format!("SarvamLLM: request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(PipecatError::pipeline(format!(
                "SarvamLLM: HTTP {} — {}",
                status, body
            )));
        }

        // --- SSE line-by-line parsing ---
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        'outer: while let Some(chunk) = stream.next().await {
            let bytes = chunk.map_err(|e| {
                PipecatError::pipeline(format!("SarvamLLM: stream read error: {}", e))
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
                    log::debug!("SarvamLLM: stream complete");
                    break 'outer;
                }

                match serde_json::from_str::<ChatChunk>(data) {
                    Ok(chunk) => {
                        if let Some(choice) = chunk.choices.first() {
                            if let Some(content) = &choice.delta.content {
                                if !content.is_empty() {
                                    processor
                                        .push_frame(
                                            Frame::llm_text(content.clone()),
                                            FrameDirection::Downstream,
                                        )
                                        .await?;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("SarvamLLM: chunk parse error: {} — raw: {}", e, data);
                    }
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
impl FrameHandler for SarvamLLMHandler {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match &frame.inner {
            FrameInner::Data(DataFrame::LLMContextFrame(context)) => {
                let context = context.clone();

                processor
                    .push_frame(Frame::llm_full_response_start(), FrameDirection::Downstream)
                    .await?;

                if let Err(e) = self.run_inference(context, processor).await {
                    log::error!("SarvamLLM: inference error: {}", e);
                    processor.push_error(e.to_string(), false).await?;
                }

                // Always push end frame — aggregator needs it to reset cleanly
                processor
                    .push_frame(Frame::llm_full_response_end(), FrameDirection::Downstream)
                    .await?;
            }
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
