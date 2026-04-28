//! Dhara — conversation flow management.
//!
//! "Dhara" (ധാര) means flow/stream. This module provides a node-based
//! conversation flow system where the LLM moves between stages, each with
//! its own tools, system prompt, and task messages.
//!
//! ```text
//! dhara/
//!   mod.rs          ← you are here
//!   node.rs         ← NodeConfig, ContextStrategy
//!   transition.rs   ← TransitionResult
//!   manager.rs      ← DharaManager
//! ```

pub mod manager;
pub mod node;
pub mod transition;

pub use manager::{DharaHandlerFn, DharaHandlerFuture, DharaManager};
pub use node::{ContextStrategy, NodeConfig};
pub use transition::TransitionResult;
