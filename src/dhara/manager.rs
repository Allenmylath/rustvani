//! Dhara flow manager.
//!
//! Orchestrates conversation flows by managing node transitions, swapping
//! tools and function registries, and updating context as the conversation
//! moves between stages.
//!
//! All nodes are registered upfront. Tool handlers return `TransitionResult`
//! to signal whether to stay or move to a named node.
//!
//! # Example
//!
//! ```rust,ignore
//! // 1. Create shared pieces
//! let context = shared_context(None);
//! let registry = Arc::new(Mutex::new(FunctionRegistry::new()));
//!
//! // 2. Create manager
//! let mut dhara = DharaManager::new(context.clone(), registry.clone());
//!
//! // 3. Register nodes with their tools + handlers
//! dhara.register_node("greeting", greeting_node, vec![
//!     ("check_weather", weather_handler),
//!     ("end_conversation", end_handler),
//! ]);
//! dhara.register_node("farewell", farewell_node, vec![]);
//!
//! // 4. Set initial node
//! dhara.set_initial_node("greeting");
//!
//! // 5. Create handler with shared registry + transition hook
//! let mut llm = OpenAILLMHandler::with_shared_registry(config, registry);
//! llm.set_transition_hook(dhara.create_transition_hook());
//!
//! // 6. Build pipeline
//! ```

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use log;

use super::node::{ContextStrategy, NodeConfig};
use super::transition::TransitionResult;
use crate::context::LLMContext;
use crate::services::llm::function_registry::FunctionRegistry;
use crate::services::llm::openai::TransitionHook;

// ---------------------------------------------------------------------------
// Dhara handler types
// ---------------------------------------------------------------------------

/// Future returned by a dhara handler.
pub type DharaHandlerFuture = Pin<Box<dyn Future<Output = TransitionResult> + Send>>;

/// A dhara-aware async function handler.
///
/// Receives the raw JSON arguments string, returns `TransitionResult`.
pub type DharaHandlerFn = Arc<dyn Fn(String) -> DharaHandlerFuture + Send + Sync>;

// ---------------------------------------------------------------------------
// Internal: registered node with its handlers
// ---------------------------------------------------------------------------

struct RegisteredNode {
    config: NodeConfig,
    handlers: Vec<(String, DharaHandlerFn)>,
}

// ---------------------------------------------------------------------------
// DharaManager
// ---------------------------------------------------------------------------

/// Orchestrates conversation flows across multiple nodes.
///
/// Nodes are registered upfront with their configs and handlers.
/// The manager swaps the shared `FunctionRegistry` and updates the
/// shared `LLMContext` on each transition.
pub struct DharaManager {
    context: Arc<Mutex<LLMContext>>,
    registry: Arc<Mutex<FunctionRegistry>>,
    nodes: HashMap<String, RegisteredNode>,
    current_node: Option<String>,
    /// Pending transition — set by wrapped handlers, consumed by the hook.
    pending_transition: Arc<Mutex<Option<String>>>,
}

impl DharaManager {
    /// Create a new DharaManager.
    ///
    /// `context` and `registry` must be the same `Arc`s passed to the
    /// `OpenAILLMHandler` so they share state.
    pub fn new(
        context: Arc<Mutex<LLMContext>>,
        registry: Arc<Mutex<FunctionRegistry>>,
    ) -> Self {
        Self {
            context,
            registry,
            nodes: HashMap::new(),
            current_node: None,
            pending_transition: Arc::new(Mutex::new(None)),
        }
    }

    /// Register a node by name with its config and tool handlers.
    ///
    /// `handlers` is a list of `(function_name, handler_fn)` pairs.
    /// The handler functions return `TransitionResult` — they can signal
    /// a transition to another registered node.
    pub fn register_node(
        &mut self,
        name: impl Into<String>,
        config: NodeConfig,
        handlers: Vec<(impl Into<String>, DharaHandlerFn)>,
    ) {
        let name = name.into();
        let handlers: Vec<(String, DharaHandlerFn)> = handlers
            .into_iter()
            .map(|(n, h)| (n.into(), h))
            .collect();
        log::debug!("Dhara: registered node '{}' with {} handler(s)", name, handlers.len());
        self.nodes.insert(name, RegisteredNode { config, handlers });
    }

    /// Set the initial node. Applies immediately (no pending transition).
    ///
    /// Panics if the node name is not registered.
    pub fn set_initial_node(&mut self, name: &str) {
        let node = self.nodes.get(name).unwrap_or_else(|| {
            panic!("Dhara: node '{}' not registered", name)
        });

        log::info!("Dhara: setting initial node '{}'", name);
        self.current_node = Some(name.to_string());

        // Apply context
        Self::apply_node_to_context(&self.context, &node.config);

        // Build registry
        self.build_registry_for_node(name);
    }

    /// Get the current node name.
    pub fn current_node(&self) -> Option<&str> {
        self.current_node.as_deref()
    }

    /// Create the transition hook for `OpenAILLMHandler`.
    ///
    /// Call this after registering all nodes. Pass the returned hook to
    /// `handler.set_transition_hook(hook)`.
    pub fn create_transition_hook(&self) -> TransitionHook {
        let pending = self.pending_transition.clone();
        let registry = self.registry.clone();
        let context_ref = self.context.clone();

        // Clone the node data into the hook closure.
        // We need owned copies since the closure outlives &self.
        let nodes: HashMap<String, (NodeConfig, Vec<(String, DharaHandlerFn)>)> = self
            .nodes
            .iter()
            .map(|(name, rn)| {
                (
                    name.clone(),
                    (rn.config.clone(), rn.handlers.clone()),
                )
            })
            .collect();

        let pending_for_rebuild = pending.clone();

        Arc::new(move |context: &Arc<Mutex<LLMContext>>| {
            let next_name = pending.lock().unwrap().take();

            if let Some(name) = next_name {
                if let Some((config, handlers)) = nodes.get(&name) {
                    log::info!("Dhara: transitioning to node '{}'", name);

                    // 1. Apply context changes
                    Self::apply_node_to_context(context, config);

                    // 2. Rebuild registry with wrapped handlers
                    {
                        let mut reg = registry.lock().unwrap();
                        *reg = FunctionRegistry::new();

                        for (fn_name, handler) in handlers {
                            let handler = handler.clone();
                            let pt = pending_for_rebuild.clone();

                            reg.register(fn_name.clone(), move |args: String| {
                                let handler = handler.clone();
                                let pt = pt.clone();

                                async move {
                                    let result = handler(args).await;
                                    match result {
                                        TransitionResult::Stay(r) => r,
                                        TransitionResult::Transition { result, next_node } => {
                                            log::info!(
                                                "Dhara: handler requested transition to '{}'",
                                                next_node
                                            );
                                            *pt.lock().unwrap() = Some(next_node);
                                            result
                                        }
                                    }
                                }
                            });
                        }
                    }

                    log::info!("Dhara: transition to '{}' complete", name);
                } else {
                    log::error!("Dhara: node '{}' not registered, ignoring transition", name);
                }
            }
        })
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Apply a node's config to the shared context.
    fn apply_node_to_context(context: &Arc<Mutex<LLMContext>>, config: &NodeConfig) {
        let mut ctx = context.lock().unwrap();

        // System prompt
        if let Some(prompt) = &config.system_prompt {
            ctx.system_prompt = Some(prompt.clone());
        }

        // Context strategy
        match config.context_strategy {
            ContextStrategy::Reset => {
                ctx.messages.clear();
            }
            ContextStrategy::Append => {}
        }

        // Task messages
        for msg in &config.task_messages {
            ctx.push_message(msg.clone());
        }

        // Tools
        ctx.tools = config.tools.clone();
        ctx.tool_choice = None; // provider default ("auto")
    }

    /// Build the function registry for a named node.
    ///
    /// Wraps each dhara handler to intercept `TransitionResult::Transition`
    /// and store the pending transition.
    fn build_registry_for_node(&self, name: &str) {
        let node = self.nodes.get(name).expect("node must be registered");
        let mut reg = self.registry.lock().unwrap();
        *reg = FunctionRegistry::new();

        for (fn_name, handler) in &node.handlers {
            let handler = handler.clone();
            let pending = self.pending_transition.clone();

            reg.register(fn_name.clone(), move |args: String| {
                let handler = handler.clone();
                let pending = pending.clone();

                async move {
                    let result = handler(args).await;
                    match result {
                        TransitionResult::Stay(r) => r,
                        TransitionResult::Transition { result, next_node } => {
                            log::info!(
                                "Dhara: handler requested transition to '{}'",
                                next_node
                            );
                            *pending.lock().unwrap() = Some(next_node);
                            result
                        }
                    }
                }
            });
        }
    }
}
