//! OpenAI LLM service (chat completions, SSE streaming).
//!
//! POST to https://api.openai.com/v1/chat/completions with stream=true.
//! Auth: Authorization: Bearer {api_key}
//! Tool/function calls are not handled — text output only.
//!
//! Pipeline position:
//!   LLMUserAggregator → OpenAILLMHandler → LLMAssistantAggregator
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

use crate::context::LLMContext;
use crate::error::{PipecatError, Result};
use crate::frames::{
    DataFrame, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor,
};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct OpenAILLMConfig {
    pub api_key: String,
    /// e.g. "gpt-4.1", "gpt-4o", "gpt-4o-mini"
    pub model: String,
    /// Override for Azure OpenAI or local proxies. Default: https://api.openai.com/v1
    pub base_url: String,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<i64>,
    /// Maps to max_completion_tokens (preferred) or max_tokens for older models.
    pub max_completion_tokens: Option<u32>,
    /// Service tier: "auto", "flex", "priority". None omits the field.
    pub service_tier: Option<String>,
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
        }
    }
}

// ---------------------------------------------------------------------------
// OpenAI API wire types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<crate::context::Message>,
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
}

#[derive(Deserialize)]
struct ChatChunk {
    choices: Vec<ChunkChoice>,
    // usage is only present on the final chunk when stream_options.include_usage=true.
    // We don't request it for now, so this is always None.
    #[allow(dead_code)]
    usage: Option<serde_json::Value>,
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
    // role is present on the first chunk only ("assistant"). Ignored.
    #[allow(dead_code)]
    role: Option<String>,
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

pub struct OpenAILLMHandler {
    config: OpenAILLMConfig,
    client: Client,
}

impl OpenAILLMHandler {
    pub fn new(config: OpenAILLMConfig) -> Self {
        Self {
            config,
            client: Client::new(),
        }
    }

    pub fn into_processor(self) -> FrameProcessor {
        FrameProcessor::new("OpenAILLM", Box::new(self), false)
    }

    /// POST to OpenAI chat completions, parse SSE stream, push LLMTextFrames.
    async fn run_inference(
        &self,
        context: std::sync::Arc<std::sync::Mutex<LLMContext>>,
        processor: &FrameProcessor,
    ) -> Result<()> {
        let messages = context.lock().unwrap().to_api_messages();

        let url = format!("{}/chat/completions", self.config.base_url);

        log::info!(
            "OpenAILLM: {} messages → {} (model={})",
            messages.len(),
            url,
            self.config.model,
        );

        let body = ChatRequest {
            model: self.config.model.clone(),
            messages,
            stream: true,
            temperature: self.config.temperature,
            top_p: self.config.top_p,
            frequency_penalty: self.config.frequency_penalty,
            presence_penalty: self.config.presence_penalty,
            seed: self.config.seed,
            max_completion_tokens: self.config.max_completion_tokens,
            service_tier: self.config.service_tier.clone(),
        };

        let response = self
            .client
            .post(&url)
            .header(
                "Authorization",
                format!("Bearer {}", self.config.api_key),
            )
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| PipecatError::pipeline(format!("OpenAILLM: request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(PipecatError::pipeline(format!(
                "OpenAILLM: HTTP {} — {}",
                status, body
            )));
        }

        // --- SSE line-by-line parsing (identical pattern to sarvam.rs) ---
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

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
                        log::warn!("OpenAILLM: chunk parse error: {} — raw: {}", e, data);
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
impl FrameHandler for OpenAILLMHandler {
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
                    log::error!("OpenAILLM: inference error: {}", e);
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