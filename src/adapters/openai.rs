//! OpenAI-specific adapter.
//!
//! Converts rustvani's universal schemas and messages to OpenAI's
//! chat completions wire format.
//!
//! Mirrors pipecat's `OpenAILLMAdapter`.

use serde_json::{json, Value};

use super::base::LLMAdapter;
use super::schemas::{AdapterType, ToolChoice, ToolsSchema};
use crate::context::Message;

// ---------------------------------------------------------------------------
// Adapter
// ---------------------------------------------------------------------------

/// OpenAI adapter — converts universal types to OpenAI chat completions format.
#[derive(Debug, Clone, Default)]
pub struct OpenAILLMAdapter;

impl OpenAILLMAdapter {
    pub fn new() -> Self {
        Self
    }
}

impl LLMAdapter for OpenAILLMAdapter {
    fn to_provider_tools_format(&self, tools: &ToolsSchema) -> Vec<Value> {
        let mut result: Vec<Value> = tools
            .standard_tools
            .iter()
            .map(|func| {
                json!({
                    "type": "function",
                    "function": func.to_default_dict()
                })
            })
            .collect();

        // Append any OpenAI-specific custom tools
        if let Some(custom) = &tools.custom_tools {
            if let Some(openai_tools) = custom.get(&AdapterType::OpenAI) {
                result.extend(openai_tools.iter().cloned());
            }
        }

        result
    }

    fn to_provider_tool_choice(&self, choice: &ToolChoice) -> Value {
        choice.to_openai_value()
    }

    fn convert_messages(&self, messages: &[Message]) -> Vec<Value> {
        messages
            .iter()
            .map(|msg| {
                // For now, simple text messages. When Message gains
                // tool_calls / tool_call_id fields, this will expand
                // to emit the full OpenAI message shape.
                json!({
                    "role": msg.role,
                    "content": msg.content
                })
            })
            .collect()
    }
}
