//! Dhara node configuration.
//!
//! A node represents a single stage in a conversation flow.
//! Each node defines its own system prompt, task messages, available tools,
//! and how context should be managed on entry.

use crate::adapters::schemas::ToolsSchema;
use crate::context::Message;

// ---------------------------------------------------------------------------
// ContextStrategy
// ---------------------------------------------------------------------------

/// How to update the conversation context when entering a new node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextStrategy {
    /// Keep existing conversation history, append new task messages.
    /// Good for progressive conversations where history matters.
    Append,

    /// Clear conversation history, start fresh with new task messages.
    /// Good for topic changes or when history would confuse the model.
    Reset,
}

impl Default for ContextStrategy {
    fn default() -> Self {
        Self::Append
    }
}

// ---------------------------------------------------------------------------
// NodeConfig
// ---------------------------------------------------------------------------

/// Configuration for a single stage in a conversation flow.
///
/// Each node defines what the LLM should do, what tools it can use,
/// and how context transitions work.
///
/// # Example
/// ```rust
/// use rustvani::dhara::NodeConfig;
/// use rustvani::context::Message;
///
/// let greeting_node = NodeConfig::new("greeting")
///     .with_system_prompt("You are a friendly assistant.")
///     .with_task_message("Greet the user and ask how you can help.")
///     .with_respond_immediately(true);
/// ```
#[derive(Debug, Clone)]
pub struct NodeConfig {
    /// Unique name for this node (used in logs and debugging).
    pub name: String,

    /// System prompt override. If `Some`, replaces the context's system_prompt
    /// when entering this node. If `None`, the existing system_prompt is kept.
    pub system_prompt: Option<String>,

    /// Developer/task messages that tell the LLM what to do at this stage.
    /// These are appended or replace the context based on `context_strategy`.
    pub task_messages: Vec<Message>,

    /// Tools available at this stage. `None` means no tools (text-only).
    /// Replaces whatever tools were on the context previously.
    pub tools: Option<ToolsSchema>,

    /// How to update context when entering this node.
    pub context_strategy: ContextStrategy,

    /// Whether to trigger LLM inference immediately upon entering this node.
    /// If `false`, the node waits for the next user turn to trigger inference.
    pub respond_immediately: bool,
}

impl NodeConfig {
    /// Create a new node with the given name and sensible defaults.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            system_prompt: None,
            task_messages: Vec::new(),
            tools: None,
            context_strategy: ContextStrategy::default(),
            respond_immediately: true,
        }
    }

    /// Builder: set the system prompt override.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Builder: add a single task message (as a developer/system instruction).
    pub fn with_task_message(mut self, content: impl Into<String>) -> Self {
        self.task_messages.push(Message::System {
            content: content.into(),
        });
        self
    }

    /// Builder: add a user-role task message.
    pub fn with_user_task_message(mut self, content: impl Into<String>) -> Self {
        self.task_messages.push(Message::User {
            content: content.into(),
        });
        self
    }

    /// Builder: set the full task messages list.
    pub fn with_task_messages(mut self, messages: Vec<Message>) -> Self {
        self.task_messages = messages;
        self
    }

    /// Builder: set the tools available at this node.
    pub fn with_tools(mut self, tools: ToolsSchema) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Builder: set the context strategy.
    pub fn with_context_strategy(mut self, strategy: ContextStrategy) -> Self {
        self.context_strategy = strategy;
        self
    }

    /// Builder: set whether to trigger LLM immediately on entering this node.
    pub fn with_respond_immediately(mut self, respond: bool) -> Self {
        self.respond_immediately = respond;
        self
    }
}
