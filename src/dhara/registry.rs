//! Dhara function registry — maps function names to handler implementations.
//!
//! This is separate from the LLM's FunctionRegistry. Dhara handlers return
//! plain result strings; the framework handles transitions based on the JSON.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use super::dctx::DharaContext;

// ---------------------------------------------------------------------------
// Handler types
// ---------------------------------------------------------------------------

/// Future returned by a dhara handler.
pub type HandlerFuture = Pin<Box<dyn Future<Output = HandlerResult> + Send>>;

/// What a handler returns.
pub struct HandlerResult {
    /// Result string sent back to the LLM as the tool response.
    pub result: String,

    /// Optional transition status key (matches against `transitions` in dhara.json).
    /// If `None`, "default" is used.
    pub status: Option<String>,
}

impl HandlerResult {
    /// Simple result with default transition.
    pub fn ok(result: impl Into<String>) -> Self {
        Self { result: result.into(), status: None }
    }

    /// Result with a named status for conditional transitions.
    ///
    /// ```rust,ignore
    /// HandlerResult::with_status(result, "on_error")
    /// ```
    pub fn with_status(result: impl Into<String>, status: impl Into<String>) -> Self {
        Self { result: result.into(), status: Some(status.into()) }
    }
}

/// A dhara handler function signature.
///
/// Receives:
/// - `args`: raw JSON arguments from the LLM
/// - `ctx`: shared DharaContext with pipeline access, state, etc.
///
/// Returns a `HandlerResult`.
pub type DharaHandlerFn = Arc<
    dyn Fn(String, DharaContext) -> HandlerFuture + Send + Sync
>;

// ---------------------------------------------------------------------------
// DharaFunctionRegistry
// ---------------------------------------------------------------------------

/// Registry of handler implementations, keyed by function name.
///
/// Populated in the flow's `functions.rs` via `register()` calls.
/// The loader validates that every function in `dhara.json` has a
/// matching entry here.
pub struct DharaFunctionRegistry {
    handlers: HashMap<String, DharaHandlerFn>,
}

impl DharaFunctionRegistry {
    pub fn new() -> Self {
        Self { handlers: HashMap::new() }
    }

    /// Register a handler by name.
    ///
    /// ```rust,ignore
    /// reg.register("begin_interview", |args, ctx| Box::pin(async move {
    ///     let parsed: Value = serde_json::from_str(&args).unwrap_or_default();
    ///     let state = ctx.state::<Mutex<InterviewState>>().unwrap();
    ///     // ... do work ...
    ///     HandlerResult::ok(json!({ "status": "started" }).to_string())
    /// }));
    /// ```
    pub fn register<F, Fut>(&mut self, name: impl Into<String>, handler: F)
    where
        F: Fn(String, DharaContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = HandlerResult> + Send + 'static,
    {
        let handler = Arc::new(move |args: String, ctx: DharaContext| -> HandlerFuture {
            Box::pin(handler(args, ctx))
        });
        self.handlers.insert(name.into(), handler);
    }

    /// Get a handler by name.
    pub fn get(&self, name: &str) -> Option<&DharaHandlerFn> {
        self.handlers.get(name)
    }

    /// Check if a function name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.handlers.contains_key(name)
    }

    /// List all registered function names.
    pub fn names(&self) -> Vec<&str> {
        self.handlers.keys().map(|s| s.as_str()).collect()
    }
}
