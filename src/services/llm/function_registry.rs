//! Function call registry for LLM tool execution.
//!
//! The LLM handler owns a `FunctionRegistry` and uses it to look up and
//! execute functions when the model emits `tool_calls`.
//!
//! Two handler breeds:
//!
//! - **Simple** (`register`): returns a plain `String` → fed directly into LLM
//!   context as-is. Identical to the original behaviour. Use for actions,
//!   state mutations, notifications — anything where the full result is safe
//!   to show the model.
//!
//! - **Data** (`register_data`): returns `ToolCallOutput { summary, full_data }`
//!   → `summary` goes into LLM context, `full_data` (if `Some`) is emitted as
//!   a `FunctionCallRawResultFrame` downstream. Use for DB queries, search
//!   results, or any call that may return large payloads the LLM shouldn't
//!   see raw.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

// ---------------------------------------------------------------------------
// ToolCallOutput — return type for data handlers
// ---------------------------------------------------------------------------

/// Output from a data tool handler.
///
/// `summary` is mandatory — it is what the LLM sees in its context.
/// Even trivial tools must return a human-readable summary.
///
/// `full_data` is optional — when `Some`, it is emitted as a
/// `FunctionCallRawResultFrame` downstream for logging, UI, or storage.
/// Set to `None` for tools that have no structured output worth emitting.
#[derive(Debug, Clone)]
pub struct ToolCallOutput {
    /// What the LLM sees. Never empty — provide at least a placeholder.
    pub summary: String,
    /// Full structured payload. `None` suppresses the raw result frame.
    pub full_data: Option<Value>,
}

impl ToolCallOutput {
    /// Convenience constructor: summary only, no raw data frame.
    pub fn summary_only(summary: impl Into<String>) -> Self {
        Self { summary: summary.into(), full_data: None }
    }

    /// Convenience constructor: summary + structured payload.
    pub fn with_data(summary: impl Into<String>, data: Value) -> Self {
        Self { summary: summary.into(), full_data: Some(data) }
    }
}

// ---------------------------------------------------------------------------
// Handler type aliases
// ---------------------------------------------------------------------------

/// Future returned by a simple handler.
pub type HandlerFuture = Pin<Box<dyn Future<Output = String> + Send>>;

/// Future returned by a data handler.
pub type DataHandlerFuture = Pin<Box<dyn Future<Output = ToolCallOutput> + Send>>;

/// Boxed simple handler — receives raw JSON args, returns result string.
pub type HandlerFn = Arc<dyn Fn(String) -> HandlerFuture + Send + Sync>;

/// Boxed data handler — receives raw JSON args, returns `ToolCallOutput`.
pub type DataHandlerFn = Arc<dyn Fn(String) -> DataHandlerFuture + Send + Sync>;

// ---------------------------------------------------------------------------
// RegistryHandler — the enum stored in the map
// ---------------------------------------------------------------------------

/// Discriminates between the two handler breeds stored in the registry.
pub enum RegistryHandler {
    /// Plain string return — goes to LLM context as-is.
    Simple(HandlerFn),
    /// Structured return — summary to LLM, full_data as raw result frame.
    Data(DataHandlerFn),
}

// ---------------------------------------------------------------------------
// FunctionRegistry
// ---------------------------------------------------------------------------

/// Registry of tool/function handlers keyed by function name.
///
/// # Example
/// ```rust
/// use rustvani::services::llm::FunctionRegistry;
/// use rustvani::services::llm::function_registry::ToolCallOutput;
/// use serde_json::json;
///
/// let mut registry = FunctionRegistry::new();
///
/// // Simple tool — full result goes to LLM
/// registry.register("send_notification", |_args: String| async move {
///     "notification sent".to_string()
/// });
///
/// // Data tool — summary to LLM, raw rows as a downstream frame
/// registry.register_data("search_cases", |args: String| async move {
///     let rows = vec![json!({"id": 1}), json!({"id": 2})]; // DB result
///     ToolCallOutput::with_data(
///         format!("found {} cases", rows.len()),
///         json!(rows),
///     )
/// });
/// ```
pub struct FunctionRegistry {
    handlers: HashMap<String, RegistryHandler>,
}

impl FunctionRegistry {
    pub fn new() -> Self {
        Self { handlers: HashMap::new() }
    }

    // -----------------------------------------------------------------------
    // Registration
    // -----------------------------------------------------------------------

    /// Register a simple handler. Returns `String` — goes to LLM context.
    pub fn register<F, Fut>(&mut self, name: impl Into<String>, handler: F)
    where
        F: Fn(String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = String> + Send + 'static,
    {
        let name = name.into();
        log::debug!("FunctionRegistry: registered simple handler for '{}'", name);
        self.handlers.insert(
            name,
            RegistryHandler::Simple(Arc::new(move |args| Box::pin(handler(args)))),
        );
    }

    /// Register a data handler. Returns `ToolCallOutput`:
    /// - `summary` → LLM context
    /// - `full_data` → `FunctionCallRawResultFrame` downstream (if `Some`)
    pub fn register_data<F, Fut>(&mut self, name: impl Into<String>, handler: F)
    where
        F: Fn(String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ToolCallOutput> + Send + 'static,
    {
        let name = name.into();
        log::debug!("FunctionRegistry: registered data handler for '{}'", name);
        self.handlers.insert(
            name,
            RegistryHandler::Data(Arc::new(move |args| Box::pin(handler(args)))),
        );
    }

    // -----------------------------------------------------------------------
    // Lookup
    // -----------------------------------------------------------------------

    /// Look up a handler by function name.
    pub fn get(&self, name: &str) -> Option<&RegistryHandler> {
        self.handlers.get(name)
    }

    /// Check if a handler is registered.
    pub fn has(&self, name: &str) -> bool {
        self.handlers.contains_key(name)
    }

    // -----------------------------------------------------------------------
    // Introspection
    // -----------------------------------------------------------------------

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
