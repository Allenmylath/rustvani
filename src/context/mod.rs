//! Shared conversation context.
//!
//! Owned by both aggregators via `Arc<Mutex<LLMContext>>`.
//! The LLM service reads it; the aggregators write to it.

use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::adapters::schemas::{ToolChoice, ToolsSchema};

// ---------------------------------------------------------------------------
// ToolCall — a single function invocation requested by the model
// ---------------------------------------------------------------------------

/// A function call the model wants to execute.
///
/// Streamed as argument-string fragments during SSE; by the time this struct
/// is constructed, `arguments` is the fully accumulated JSON string.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique call ID assigned by the model (e.g. `"call_abc123"`).
    pub id: String,
    /// Name of the function to invoke.
    pub function_name: String,
    /// Raw JSON string of the function arguments.
    pub arguments: String,
}

// ---------------------------------------------------------------------------
// Message — type-safe conversation turn
// ---------------------------------------------------------------------------

/// A single turn in the conversation.
///
/// Each variant enforces what fields are valid for that role, unlike the
/// Python dict approach where any key can appear on any message.
#[derive(Debug, Clone)]
pub enum Message {
    /// System-level instruction. Typically the first message.
    System { content: String },

    /// User turn — transcribed speech or typed text.
    User { content: String },

    /// Assistant turn — may be text, tool calls, or both.
    Assistant {
        /// `None` when the model responds with only tool calls.
        content: Option<String>,
        /// `None` for plain text responses.
        tool_calls: Option<Vec<ToolCall>>,
    },

    /// Result of a tool/function call, sent back to the model.
    ToolResult {
        /// Matches the `id` of the `ToolCall` this responds to.
        tool_call_id: String,
        /// Serialized result (typically JSON).
        content: String,
    },
}

// ---------------------------------------------------------------------------
// LLMContext
// ---------------------------------------------------------------------------

/// Shared conversation context passed between aggregators and the LLM service.
#[derive(Debug, Clone)]
pub struct LLMContext {
    /// System prompt — prepended as `Message::System` in `to_api_messages()`.
    pub system_prompt: Option<String>,
    /// Conversation history (user turns, assistant turns, tool results).
    pub messages: Vec<Message>,
    /// Available tools for this context. `None` = no function calling.
    pub tools: Option<ToolsSchema>,
    /// How the model should pick tools. `None` = provider default (usually "auto").
    pub tool_choice: Option<ToolChoice>,
}

impl LLMContext {
    pub fn new(system_prompt: Option<String>) -> Self {
        Self {
            system_prompt,
            messages: Vec::new(),
            tools: None,
            tool_choice: None,
        }
    }

    /// Create a context with tools configured.
    pub fn with_tools(
        system_prompt: Option<String>,
        tools: ToolsSchema,
        tool_choice: Option<ToolChoice>,
    ) -> Self {
        Self {
            system_prompt,
            messages: Vec::new(),
            tools: Some(tools),
            tool_choice,
        }
    }

    // ---- Convenience push methods ----

    /// Append any message variant.
    pub fn push_message(&mut self, msg: Message) {
        self.messages.push(msg);
    }

    /// Append a user turn.
    pub fn add_user_message(&mut self, content: impl Into<String>) {
        self.messages.push(Message::User {
            content: content.into(),
        });
    }

    /// Append a plain-text assistant turn (no tool calls).
    pub fn add_assistant_message(&mut self, content: impl Into<String>) {
        self.messages.push(Message::Assistant {
            content: Some(content.into()),
            tool_calls: None,
        });
    }

    /// Append an assistant turn that contains tool calls.
    pub fn add_assistant_tool_calls(
        &mut self,
        content: Option<String>,
        tool_calls: Vec<ToolCall>,
    ) {
        self.messages.push(Message::Assistant {
            content,
            tool_calls: Some(tool_calls),
        });
    }

    /// Append a tool result.
    pub fn add_tool_result(
        &mut self,
        tool_call_id: impl Into<String>,
        content: impl Into<String>,
    ) {
        self.messages.push(Message::ToolResult {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
        });
    }

    /// Build the full messages array for the API call.
    ///
    /// System prompt is prepended as the first message if present.
    /// The adapter then converts these `Message` variants into the
    /// provider's wire format.
    pub fn to_api_messages(&self) -> Vec<Message> {
        let mut result = Vec::new();
        if let Some(sys) = &self.system_prompt {
            result.push(Message::System {
                content: sys.clone(),
            });
        }
        result.extend(self.messages.clone());
        result
    }
}

/// Convenience: create a shared context ready for pipeline use.
pub fn shared_context(system_prompt: Option<String>) -> Arc<Mutex<LLMContext>> {
    Arc::new(Mutex::new(LLMContext::new(system_prompt)))
}

/// Convenience: create a shared context with tools configured.
pub fn shared_context_with_tools(
    system_prompt: Option<String>,
    tools: ToolsSchema,
    tool_choice: Option<ToolChoice>,
) -> Arc<Mutex<LLMContext>> {
    Arc::new(Mutex::new(LLMContext::with_tools(
        system_prompt,
        tools,
        tool_choice,
    )))
}
