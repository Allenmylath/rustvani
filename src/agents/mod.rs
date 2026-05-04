//! Multi-agent coordination for rustvani.
//!
//! Provides a message bus, agent registry, base agent implementation,
//! runner orchestration, and task coordination — all built on top of the
//! existing frame pipeline architecture without breaking it.

pub mod base;
pub mod bus;
pub mod registry;
pub mod runner;
pub mod task;

pub use base::{Agent, BaseAgent};
pub use bus::{
    AgentBus, AgentRegistryEntry, BusMessage, BusPayload, BusSubscriber, LocalAgentBus,
    TaskStatus,
};
pub use registry::{AgentInfo, AgentRegistry};
pub use runner::AgentRunner;
pub use task::{TaskContext, TaskHandle, TaskResult, TaskUpdate};
