use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

/// A single message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Shared conversation context — owned by both aggregators via Arc<Mutex<>>.
///
/// The LLM reads it. The aggregators write to it.
/// System prompt is prepended at API call time — not stored as a message.
#[derive(Debug, Clone)]
pub struct LLMContext {
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
}

impl LLMContext {
    pub fn new(system_prompt: Option<String>) -> Self {
        Self {
            system_prompt,
            messages: Vec::new(),
        }
    }

    pub fn add_message(&mut self, role: impl Into<String>, content: impl Into<String>) {
        self.messages.push(Message {
            role: role.into(),
            content: content.into(),
        });
    }

    /// Build the full messages array for the Sarvam API call.
    /// System prompt is prepended as the first message if present.
    pub fn to_api_messages(&self) -> Vec<Message> {
        let mut result = Vec::new();
        if let Some(sys) = &self.system_prompt {
            result.push(Message {
                role: "system".to_string(),
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
