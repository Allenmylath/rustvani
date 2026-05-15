//! Dhara — declarative conversation flow management.
//!
//! "Dhara" (ധാര) means flow/stream. Define conversation flows in JSON,
//! implement handlers in Rust, and let the framework wire everything.
//!
//! # Architecture
//!
//! ```text
//! dhara/<flow_id>/
//!   dhara.json      ← flow graph, prompts, transitions, function schemas
//!   functions.rs    ← handler implementations
//!
//! main.rs           ← load, build, plug into pipeline
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! // 1. Register handlers (from your functions.rs)
//! let mut handlers = DharaFunctionRegistry::new();
//! handlers.register("begin_interview", |args, ctx| async move { ... });
//! handlers.register("end_conversation", |args, ctx| async move { ... });
//!
//! // 2. Load and validate the flow
//! let dhara = Dhara::load("dhara/interview")?;
//!
//! // 3. Build runtime pieces for this connection
//! let built = dhara.build(&handlers, my_state, conn_id)?;
//!
//! // 4. Wire into pipeline
//! let mut llm = OpenAILLMHandler::with_shared_registry(config, built.llm_registry);
//! llm.set_transition_hook(built.hook);
//! // ... built.context goes to aggregators, RAVI, etc.
//! // ... after PipelineTask::new(), call built.dhara_ctx.set_push_sender(task.push_sender())
//! ```
//!
//! # Modules
//!
//! ```text
//! dhara/
//!   mod.rs          ← you are here
//!   schema.rs       ← JSON deserialization types for dhara.json
//!   dctx.rs         ← DharaContext (runtime context for handlers)
//!   registry.rs     ← DharaFunctionRegistry (handler registration)
//!   loader.rs       ← Dhara (load, validate, build)
//!   node.rs         ← NodeConfig (legacy, kept for backward compat)
//!   transition.rs   ← TransitionResult (legacy, kept for backward compat)
//!   manager.rs      ← DharaManager (legacy, kept for backward compat)
//! ```

pub mod schema;
pub mod dctx;
pub mod registry;
pub mod loader;

// Legacy modules — kept for backward compatibility during migration
pub mod manager;
pub mod node;
pub mod transition;

// --- New API (preferred) ---
pub use loader::{Dhara, DharaBuild, DharaError};
pub use dctx::DharaContext;
pub use registry::{DharaFunctionRegistry, HandlerResult};

// --- Legacy API (deprecated, use new API for new flows) ---
pub use manager::{DharaHandlerFn, DharaHandlerFuture, DharaManager};
pub use node::{ContextStrategy, NodeConfig};
pub use transition::TransitionResult;
