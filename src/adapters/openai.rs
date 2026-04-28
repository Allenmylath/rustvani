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
        messages.iter().map(|msg| Self::message_to_value(msg)).collect()
    }
}

impl OpenAILLMAdapter {
    /// Convert a single `Message` enum variant to OpenAI's JSON shape.
    fn message_to_value(msg: &Message) -> Value {
        match msg {
            Message::System { content } => {
                json!({ "role": "system", "content": content })
            }

            Message::User { content } => {
                json!({ "role": "user", "content": content })
            }

            Message::Assistant {
                content,
                tool_calls,
            } => {
                let mut obj = serde_json::Map::new();
                obj.insert("role".into(), Value::String("assistant".into()));

                // content can be null when the model only produces tool calls
                match content {
                    Some(text) => obj.insert("content".into(), Value::String(text.clone())),
                    None => obj.insert("content".into(), Value::Null),
                };

                if let Some(calls) = tool_calls {
                    let calls_json: Vec<Value> = calls
                        .iter()
                        .map(|tc| {
                            json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function_name,
                                    "arguments": tc.arguments,
                                }
                            })
                        })
                        .collect();
                    obj.insert("tool_calls".into(), Value::Array(calls_json));
                }

                Value::Object(obj)
            }

            Message::ToolResult {
                tool_call_id,
                content,
            } => {
                json!({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content,
                })
            }
        }
    }
}
