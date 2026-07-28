//! LLM services.
//!
//! Each backend is gated by the feature that supplies its transport dependency,
//! so `--no-default-features` builds pull in only what was asked for.
//! `function_registry` has no external dependencies and is always compiled —
//! `tools` and `dhara` use it unconditionally.

pub mod function_registry;
#[cfg(feature = "llm-openai")]
pub mod openai;
// Gated on `stt-sarvam` rather than a `llm-sarvam` feature: that is the feature
// carrying the reqwest dependency this module needs, and it matches how the
// re-export is gated in lib.rs.
#[cfg(feature = "stt-sarvam")]
pub mod sarvam;

pub use function_registry::FunctionRegistry;
#[cfg(feature = "llm-openai")]
pub use openai::{OpenAILLMConfig, OpenAILLMHandler};
#[cfg(feature = "stt-sarvam")]
pub use sarvam::{SarvamLLMConfig, SarvamLLMHandler};
