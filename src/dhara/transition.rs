//! Transition results for dhara flow handlers.
//!
//! When a tool handler executes, it returns a `TransitionResult` that tells
//! the DharaManager whether to stay on the current node or move to a new one.

/// What a function handler returns to control conversation flow.
#[derive(Debug, Clone)]
pub enum TransitionResult {
    /// Stay on the current node. The result string is sent back to the LLM.
    Stay(String),

    /// Transition to a named node. The result string is sent back to the
    /// LLM, then DharaManager swaps to the named node's configuration.
    ///
    /// The node must have been registered via `DharaManager::register_node`.
    Transition {
        /// Result to return to the LLM for this tool call.
        result: String,
        /// Name of the next node to transition to.
        next_node: String,
    },
}

impl TransitionResult {
    /// Convenience: stay with a result.
    pub fn stay(result: impl Into<String>) -> Self {
        Self::Stay(result.into())
    }

    /// Convenience: transition to a named node with a result.
    pub fn transition(result: impl Into<String>, next_node: impl Into<String>) -> Self {
        Self::Transition {
            result: result.into(),
            next_node: next_node.into(),
        }
    }

    /// Get the result string regardless of variant.
    pub fn result(&self) -> &str {
        match self {
            Self::Stay(r) => r,
            Self::Transition { result, .. } => result,
        }
    }

    /// Returns `true` if this result triggers a node transition.
    pub fn is_transition(&self) -> bool {
        matches!(self, Self::Transition { .. })
    }
}
