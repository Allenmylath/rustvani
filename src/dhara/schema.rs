//! Dhara JSON schema — serde types for `dhara.json`.
//!
//! These types map 1:1 to the JSON structure. The loader deserializes
//! into these, then the validator checks semantic correctness before
//! building the runtime DharaManager.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Top-level flow definition
// ---------------------------------------------------------------------------

/// Root of a `dhara.json` file.
#[derive(Debug, Serialize, Deserialize)]
pub struct DharaFlowDef {
    /// Unique identifier for this flow (e.g. "interview", "pizza_order").
    pub id: String,

    /// Schema version for forward compatibility.
    #[serde(default = "default_version")]
    pub version: String,

    /// Human-readable description of this flow.
    #[serde(default)]
    pub description: Option<String>,

    /// Universal system prompt — the bot's persona, tone, constraints.
    /// Applies to all nodes unless a node sets `system_prompt_override`.
    pub system_prompt: String,

    /// Name of the first node to activate on connection.
    pub initial_node: String,

    /// All nodes in the flow graph, keyed by name.
    pub nodes: HashMap<String, NodeDef>,

    /// All functions available in this flow, keyed by name.
    /// Each function is referenced by one or more nodes.
    #[serde(default)]
    pub functions: HashMap<String, FunctionDef>,
}

fn default_version() -> String {
    "1.0".to_string()
}

// ---------------------------------------------------------------------------
// Node definition
// ---------------------------------------------------------------------------

/// A single node (stage) in the conversation flow.
#[derive(Debug, Serialize, Deserialize)]
pub struct NodeDef {
    /// Optional system prompt override for this node.
    /// If set, replaces the universal system_prompt while in this node.
    #[serde(default)]
    pub system_prompt_override: Option<String>,

    /// Task messages injected when entering this node.
    #[serde(default)]
    pub task_messages: Vec<TaskMessage>,

    /// How to handle existing conversation context on entry.
    #[serde(default)]
    pub context_strategy: ContextStrategyDef,

    /// Whether to trigger LLM inference immediately on entering this node.
    #[serde(default = "default_true")]
    pub respond_immediately: bool,

    /// Function names available at this node.
    /// Each name must exist in the top-level `functions` map.
    #[serde(default)]
    pub functions: Vec<String>,
}

fn default_true() -> bool {
    true
}

// ---------------------------------------------------------------------------
// Task message
// ---------------------------------------------------------------------------

/// A message injected into the LLM context when entering a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMessage {
    /// Role: "system", "user", or "assistant".
    pub role: String,

    /// Message content.
    pub content: String,
}

// ---------------------------------------------------------------------------
// Context strategy
// ---------------------------------------------------------------------------

/// How to update conversation context on node entry.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ContextStrategyDef {
    /// Keep existing history, append task messages.
    #[default]
    Append,
    /// Clear history, start fresh.
    Reset,
}

// ---------------------------------------------------------------------------
// Function definition
// ---------------------------------------------------------------------------

/// A tool/function that the LLM can call.
///
/// The schema (description + parameters) is sent to the LLM.
/// The handler implementation lives in `functions.rs`.
/// Transitions are defined here — the handler just returns a result string,
/// and the framework uses the transition map to decide the next node.
#[derive(Debug, Serialize, Deserialize)]
pub struct FunctionDef {
    /// Description shown to the LLM (part of the tool schema).
    pub description: String,

    /// JSON Schema for the function's parameters.
    #[serde(default = "empty_object_schema")]
    pub parameters: Value,

    /// Transition map: status → next node name.
    ///
    /// `"default"` is the standard transition.
    /// Handlers can return `{"status": "error"}` to trigger `"on_error"`, etc.
    /// If a handler returns a status not in this map, `"default"` is used.
    #[serde(default)]
    pub transitions: HashMap<String, String>,
}

fn empty_object_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {},
        "required": []
    })
}
