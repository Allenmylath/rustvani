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

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use futures::StreamExt;
use log;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::adapters::base::LLMAdapter;
use crate::adapters::openai::OpenAILLMAdapter;
use crate::billing::{BillingCollector, BillingEvent};
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
    /// Optional billing collector — records estimated token usage per call.
    billing: Option<Arc<dyn BillingCollector>>,
}

impl SarvamLLMHandler {
    pub fn new(config: SarvamLLMConfig) -> Self {
        Self {
            config,
            client: Client::new(),
            adapter: OpenAILLMAdapter::new(),
            billing: None,
        }
    }

    pub fn with_billing(mut self, billing: Arc<dyn BillingCollector>) -> Self {
        self.billing = Some(billing);
        self
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
        let (api_messages, tools, tool_choice, estimated_input_tokens) = {
            let ctx = context.lock().unwrap();
            let messages = ctx.to_api_messages();

            // Estimate input tokens: ~4 chars per token (rough multilingual average)
            use crate::context::Message as Msg;
            let estimated_input = messages.iter()
                .map(|m| match m {
                    Msg::System { content }             => content.chars().count() as u32,
                    Msg::User { content }               => content.chars().count() as u32,
                    Msg::Assistant { content: Some(c), .. } => c.chars().count() as u32,
                    Msg::ToolResult { content, .. }     => content.chars().count() as u32,
                    _                                   => 0,
                })
                .sum::<u32>()
                / 4
                + 1; // ensure at least 1

            // Convert through the adapter (same format as OpenAI)
            let converted = self.adapter.convert_messages(&messages);

            let tools = ctx.tools.as_ref().map(|t| {
                self.adapter.to_provider_tools_format(t)
            });

            let tool_choice = ctx.tool_choice.as_ref().map(|tc| {
                self.adapter.to_provider_tool_choice(tc)
            });

            (converted, tools, tool_choice, estimated_input)
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
        let mut output_chars: usize = 0;

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
                                    output_chars += content.chars().count();
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

        // Sarvam API does not return token counts — emit estimated usage.
        if let Some(bc) = &self.billing {
            let estimated_output = (output_chars as u32 / 4).max(1);
            bc.record(BillingEvent::LlmUsage {
                session_id:    bc.session_id(),
                provider:      "sarvam".to_string(),
                model:         self.config.model.clone(),
                input_tokens:  estimated_input_tokens,
                output_tokens: estimated_output,
                estimated:     true,
                occurred_at:   Utc::now(),
            });
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::billing::NoopBillingCollector;
    use crate::context::Message;
    use std::sync::Arc;
    use super::{SarvamLLMConfig, SarvamLLMHandler};

    // Replicate the estimation formula used in run_inference so tests stay in sync.
    fn estimate_input_tokens(messages: &[Message]) -> u32 {
        messages.iter()
            .map(|m| match m {
                Message::System { content }                   => content.chars().count() as u32,
                Message::User   { content }                   => content.chars().count() as u32,
                Message::Assistant { content: Some(c), .. }  => c.chars().count() as u32,
                Message::ToolResult { content, .. }           => content.chars().count() as u32,
                _                                             => 0,
            })
            .sum::<u32>() / 4 + 1
    }

    #[test]
    fn token_estimation_system_plus_user() {
        // "You are helpful." = 17 chars, "Hello!" = 6 chars → total 23 → 23/4+1 = 6
        let msgs = vec![
            Message::System { content: "You are helpful.".into() },
            Message::User   { content: "Hello!".into() },
        ];
        assert_eq!(estimate_input_tokens(&msgs), 23 / 4 + 1);
    }

    #[test]
    fn token_estimation_assistant_with_content() {
        // "I can help you." = 15 chars → 15/4+1 = 4
        let msgs = vec![
            Message::Assistant { content: Some("I can help you.".into()), tool_calls: None },
        ];
        assert_eq!(estimate_input_tokens(&msgs), 15 / 4 + 1);
    }

    #[test]
    fn token_estimation_assistant_without_content_contributes_zero() {
        let msgs = vec![
            Message::Assistant { content: None, tool_calls: None },
        ];
        // 0 chars / 4 + 1 = 1 (minimum)
        assert_eq!(estimate_input_tokens(&msgs), 1);
    }

    #[test]
    fn token_estimation_empty_context_returns_one() {
        assert_eq!(estimate_input_tokens(&[]), 1);
    }

    #[test]
    fn token_estimation_tool_result() {
        // "result data here" = 16 chars → 16/4+1 = 5
        let msgs = vec![
            Message::ToolResult {
                tool_call_id: "call_abc".into(),
                content: "result data here".into(),
            },
        ];
        assert_eq!(estimate_input_tokens(&msgs), 16 / 4 + 1);
    }

    #[test]
    fn token_estimation_mixed_context_sums_all_content_bearing_variants() {
        let msgs = vec![
            Message::System     { content: "sys".into() },                                 // 3
            Message::User       { content: "user msg".into() },                            // 8
            Message::Assistant  { content: Some("reply".into()), tool_calls: None },       // 5
            Message::Assistant  { content: None, tool_calls: None },                       // 0
            Message::ToolResult { tool_call_id: "x".into(), content: "tool".into() },     // 4
        ];
        let total_chars = 3u32 + 8 + 5 + 0 + 4; // 20
        assert_eq!(estimate_input_tokens(&msgs), total_chars / 4 + 1);
    }

    #[test]
    fn with_billing_sets_field() {
        let h = SarvamLLMHandler::new(SarvamLLMConfig::default())
            .with_billing(Arc::new(NoopBillingCollector));
        assert!(h.billing.is_some());
    }
}
