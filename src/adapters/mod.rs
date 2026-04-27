//! LLM provider adapters.
//!
//! Each adapter converts rustvani's universal context/schema types into
//! the wire format expected by a specific LLM provider.
//!
//! ```text
//! adapters/
//!   mod.rs              ← you are here
//!   base.rs             ← LLMAdapter trait
//!   openai.rs           ← OpenAI chat completions adapter
//!   schemas/
//!     mod.rs
//!     function_schema.rs  ← FunctionSchema
//!     tools_schema.rs     ← ToolsSchema, ToolChoice, AdapterType
//! ```

pub mod base;
pub mod openai;
pub mod schemas;

pub use base::{LLMAdapter, LLMInvocationParams};
pub use openai::OpenAILLMAdapter;
pub use schemas::{AdapterType, FunctionSchema, ToolChoice, ToolsSchema};
