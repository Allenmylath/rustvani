//! Function call registry for LLM tool execution.
//!
//! The LLM handler owns a `FunctionRegistry` and uses it to look up and
//! execute functions when the model emits `tool_calls`.
//!
//! Mirrors pipecat's function registration on `LLMService`, but as a
//! standalone struct so it can be built before the handler and passed in.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// The future returned by a handler — `Pin<Box<dyn Future<Output = String> + Send>>`.
pub type HandlerFuture = Pin<Box<dyn Future<Output = String> + Send>>;

/// A boxed, cloneable async function handler.
///
/// Receives the raw JSON arguments string, returns the result string.
pub type HandlerFn = Arc<dyn Fn(String) -> HandlerFuture + Send + Sync>;

/// Registry of tool/function handlers keyed by function name.
///
/// # Example
/// ```rust
/// use rustvani::services::llm::FunctionRegistry;
///
/// let mut registry = FunctionRegistry::new();
///
/// registry.register("get_weather", |args: String| async move {
///     // parse args, do work, return result
///     format!("{{\"temp\": 31, \"city\": \"Kochi\"}}")
/// });
/// ```
pub struct FunctionRegistry {
    handlers: HashMap<String, HandlerFn>,
}

impl FunctionRegistry {
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    /// Register an async function handler.
    ///
    /// `handler` receives the raw JSON arguments string and returns a result
    /// string (typically JSON). The handler must be `Send + Sync + 'static`.
    pub fn register<F, Fut>(&mut self, name: impl Into<String>, handler: F)
    where
        F: Fn(String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = String> + Send + 'static,
    {
        let name = name.into();
        log::debug!("FunctionRegistry: registered handler for '{}'", name);
        self.handlers
            .insert(name, Arc::new(move |args| Box::pin(handler(args))));
    }

    /// Look up a handler by function name.
    pub fn get(&self, name: &str) -> Option<&HandlerFn> {
        self.handlers.get(name)
    }

    /// Check if a handler is registered.
    pub fn has(&self, name: &str) -> bool {
        self.handlers.contains_key(name)
    }

    /// Number of registered handlers.
    pub fn len(&self) -> usize {
        self.handlers.len()
    }

    /// True if no handlers are registered.
    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }

    /// Iterate over registered function names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.handlers.keys().map(|s| s.as_str())
    }
}

impl Default for FunctionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for FunctionRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionRegistry")
            .field("handlers", &self.handlers.keys().collect::<Vec<_>>())
            .finish()
    }
}
