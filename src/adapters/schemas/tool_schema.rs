//! Tools schema and tool-choice types.
//!
//! Mirrors pipecat's `ToolsSchema` — bundles standard function schemas with
//! an escape hatch for provider-specific custom tools.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::function_schema::FunctionSchema;

// ---------------------------------------------------------------------------
// AdapterType — which provider do custom tools target?
// ---------------------------------------------------------------------------

/// Supported adapter types for custom (provider-specific) tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterType {
    OpenAI,
    Gemini,
}

// ---------------------------------------------------------------------------
// ToolChoice — how the model should pick tools
// ---------------------------------------------------------------------------

/// Controls how the model selects tools.
///
/// Maps to OpenAI's `tool_choice` parameter; adapters convert to their
/// provider's equivalent.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// `"auto"` — model decides whether to call a tool.
    Auto,
    /// `"none"` — model must not call any tool.
    None,
    /// `"required"` — model must call at least one tool.
    Required,
    /// Force a specific function: `{"type": "function", "function": {"name": "..."}}`
    Function { name: String },
}

impl ToolChoice {
    /// Serialize to the OpenAI wire format value.
    pub fn to_openai_value(&self) -> Value {
        match self {
            ToolChoice::Auto => Value::String("auto".into()),
            ToolChoice::None => Value::String("none".into()),
            ToolChoice::Required => Value::String("required".into()),
            ToolChoice::Function { name } => serde_json::json!({
                "type": "function",
                "function": { "name": name }
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// ToolsSchema — the bundle
// ---------------------------------------------------------------------------

/// Unified container for tool definitions.
///
/// `standard_tools` holds provider-agnostic `FunctionSchema`s that every
/// adapter knows how to convert.
///
/// `custom_tools` is the escape hatch: raw JSON blobs keyed by `AdapterType`
/// for provider-specific features (e.g. OpenAI's `code_interpreter` tool)
/// that don't fit `FunctionSchema`.
#[derive(Debug, Clone)]
pub struct ToolsSchema {
    pub standard_tools: Vec<FunctionSchema>,
    pub custom_tools: Option<HashMap<AdapterType, Vec<Value>>>,
}

impl ToolsSchema {
    /// Create a schema with only standard tools.
    pub fn new(tools: Vec<FunctionSchema>) -> Self {
        Self {
            standard_tools: tools,
            custom_tools: None,
        }
    }

    /// Create a schema with both standard and custom tools.
    pub fn with_custom(
        tools: Vec<FunctionSchema>,
        custom: HashMap<AdapterType, Vec<Value>>,
    ) -> Self {
        Self {
            standard_tools: tools,
            custom_tools: Some(custom),
        }
    }

    /// Merge another set of standard tools (used internally for built-in tools).
    pub fn merge_standard(&self, extra: &[FunctionSchema]) -> Self {
        let mut merged = self.standard_tools.clone();
        merged.extend_from_slice(extra);
        Self {
            standard_tools: merged,
            custom_tools: self.custom_tools.clone(),
        }
    }
}
