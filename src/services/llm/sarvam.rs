//! Sarvam LLM service.
//!
//! Direct HTTP to https://api.sarvam.ai/v1/chat/completions.
//! No OpenAI client, no adapter chain — plain reqwest + SSE parsing.
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
}

impl Default for SarvamLLMConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: "sarvam-m".to_string(),
            base_url: "https://api.sarvam.ai/v1".to_string(),
            temperature: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Sarvam API wire types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<crate::context::Message>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
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
}

impl SarvamLLMHandler {
    pub fn new(config: SarvamLLMConfig) -> Self {
        Self {
            config,
            client: Client::new(),
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
        // Snapshot messages — release lock before any await
        let messages = context.lock().unwrap().to_api_messages();

        let url = format!("{}/chat/completions", self.config.base_url);

        log::info!(
            "SarvamLLM: {} messages → {} (model={})",
            messages.len(),
            url,
            self.config.model
        );

        let body = ChatRequest {
            model: self.config.model.clone(),
            messages,
            stream: true,
            temperature: self.config.temperature,
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

            // Drain all complete lines from buffer
            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].trim_end_matches('\r').trim().to_string();
                buffer = buffer[pos + 1..].to_string();

                if line.is_empty() {
                    continue;
                }

                let data = match line.strip_prefix("data: ") {
                    Some(d) => d,
                    None => continue, // e.g. "event: ...", ignore
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
