//! Base adapter trait for LLM provider integration.
//!
//! Mirrors pipecat's `BaseLLMAdapter` — provides a standard interface for
//! converting between rustvani's universal types and provider-specific wire
//! formats.
//!
//! Each provider (OpenAI, Gemini, Anthropic, …) implements this trait once.
//! The LLM service handler is generic over `A: LLMAdapter` so the compiler
//! monomorphizes the conversion code — zero-cost abstraction over Python's
//! dynamic dispatch.

use serde_json::Value;

use super::schemas::{FunctionSchema, ToolChoice, ToolsSchema};
use crate::context::Message;

// ---------------------------------------------------------------------------
// Adapter output — what the service needs to build its HTTP request
// ---------------------------------------------------------------------------

/// Provider-ready invocation parameters returned by the adapter.
///
/// The service handler destructures this to build its HTTP/WebSocket request.
#[derive(Debug, Clone)]
pub struct LLMInvocationParams {
    /// Messages in the provider's expected format (JSON values).
    pub messages: Vec<Value>,

    /// Tools in the provider's format. `None` = no tools this turn.
    pub tools: Option<Vec<Value>>,

    /// Tool choice in the provider's format. `None` = omit from request.
    pub tool_choice: Option<Value>,
}

// ---------------------------------------------------------------------------
// The trait
// ---------------------------------------------------------------------------

/// Converts rustvani's universal context + schemas into provider-specific
/// wire format.
///
/// Designed to be used as a generic bound: `struct Handler<A: LLMAdapter>`.
/// Only one adapter is active per handler instance, so the compiler
/// monomorphizes everything — no vtable overhead.
pub trait LLMAdapter: Send + Sync {
    /// Convert standard tools schema → provider's tool definitions.
    ///
    /// Returns a vec of JSON values ready to drop into the request body.
    fn to_provider_tools_format(&self, tools: &ToolsSchema) -> Vec<Value>;

    /// Convert a `ToolChoice` enum → the provider's JSON representation.
    fn to_provider_tool_choice(&self, choice: &ToolChoice) -> Value;

    /// Convert universal `Message` list → provider's message format.
    ///
    /// Handles role mapping (e.g. "developer" → "user" for providers that
    /// don't support it), content format conversion, and tool_calls /
    /// tool-result message shapes.
    fn convert_messages(&self, messages: &[Message]) -> Vec<Value>;

    /// Build the complete invocation params from context pieces.
    ///
    /// Default implementation wires together the other methods. Override
    /// only if your provider needs non-standard assembly.
    fn get_invocation_params(
        &self,
        messages: &[Message],
        tools: Option<&ToolsSchema>,
        tool_choice: Option<&ToolChoice>,
        builtin_tools: &[FunctionSchema],
    ) -> LLMInvocationParams {
        let converted_messages = self.convert_messages(messages);

        let converted_tools = tools.map(|t| {
            // Merge built-in tools if any
            let effective = if builtin_tools.is_empty() {
                t.clone()
            } else {
                t.merge_standard(builtin_tools)
            };
            self.to_provider_tools_format(&effective)
        });

        let converted_choice = tool_choice.map(|c| self.to_provider_tool_choice(c));

        LLMInvocationParams {
            messages: converted_messages,
            tools: converted_tools,
            tool_choice: converted_choice,
        }
    }
}
