//! Dhara loader — load, validate, and build a flow from `dhara.json`.
//!
//! ```rust,ignore
//! // In functions.rs:
//! pub fn register(reg: &mut DharaFunctionRegistry) { ... }
//!
//! // In main.rs:
//! let dhara = Dhara::from_json(include_str!("../../dhara/interview/dhara.json"))?;
//! // or
//! let dhara = Dhara::load("dhara/interview")?;
//!
//! let built = dhara.build(&handlers, my_state, conn_id)?;
//! // built.context, built.registry, built.hook — plug into pipeline
//! ```

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex};

use super::dctx::DharaContext;
use super::registry::{DharaFunctionRegistry, DharaHandlerFn};
use super::schema::*;
use crate::adapters::schemas::{FunctionSchema, ToolsSchema};
use crate::context::{LLMContext, Message};
use crate::services::llm::function_registry::FunctionRegistry;
use crate::services::llm::openai::TransitionHook;

// ---------------------------------------------------------------------------
// Validation errors
// ---------------------------------------------------------------------------

/// Errors found during flow validation.
#[derive(Debug, thiserror::Error)]
pub enum DharaError {
    #[error("Failed to read dhara.json: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Failed to parse dhara.json: {0}")]
    ParseError(#[from] serde_json::Error),

    #[error("Validation failed:\n{}", .0.join("\n"))]
    ValidationErrors(Vec<String>),
}

// ---------------------------------------------------------------------------
// Validated flow definition (immutable after validation)
// ---------------------------------------------------------------------------

/// A validated, immutable flow definition.
///
/// Created by `Dhara::load()` or `Dhara::from_json()`.
/// Call `.build()` to create the runtime pieces for a connection.
pub struct Dhara {
    def: DharaFlowDef,
}

impl Dhara {
    /// Load a flow from a directory containing `dhara.json`.
    ///
    /// ```rust,ignore
    /// let dhara = Dhara::load("dhara/interview")?;
    /// ```
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, DharaError> {
        let path = dir.as_ref().join("dhara.json");
        let json_str = std::fs::read_to_string(&path)?;
        Self::from_json(&json_str)
    }

    /// Load a flow from a JSON string (useful with `include_str!`).
    ///
    /// ```rust,ignore
    /// let dhara = Dhara::from_json(include_str!("../../dhara/interview/dhara.json"))?;
    /// ```
    pub fn from_json(json_str: &str) -> Result<Self, DharaError> {
        let def: DharaFlowDef = serde_json::from_str(json_str)?;
        Self::validate(&def)?;
        Ok(Self { def })
    }

    /// Access the raw flow definition.
    pub fn def(&self) -> &DharaFlowDef {
        &self.def
    }

    /// Flow ID.
    pub fn id(&self) -> &str {
        &self.def.id
    }

    // -----------------------------------------------------------------------
    // Validation (the "compiler")
    // -----------------------------------------------------------------------

    fn validate(def: &DharaFlowDef) -> Result<(), DharaError> {
        let mut errors = Vec::new();

        // 1. initial_node must exist
        if !def.nodes.contains_key(&def.initial_node) {
            errors.push(format!(
                "initial_node '{}' does not exist in nodes",
                def.initial_node
            ));
        }

        // 2. Every function referenced by a node must exist in top-level functions
        for (node_name, node) in &def.nodes {
            for fn_name in &node.functions {
                if !def.functions.contains_key(fn_name) {
                    errors.push(format!(
                        "node '{}' references function '{}' which is not defined in functions",
                        node_name, fn_name
                    ));
                }
            }
        }

        // 3. Every transition target must be a valid node
        for (fn_name, fn_def) in &def.functions {
            for (status, target_node) in &fn_def.transitions {
                if !def.nodes.contains_key(target_node) {
                    errors.push(format!(
                        "function '{}' transition '{}' targets node '{}' which does not exist",
                        fn_name, status, target_node
                    ));
                }
            }

            // Warn (not error) if no transitions defined — might be a terminal action
            if fn_def.transitions.is_empty() {
                log::warn!(
                    "Dhara: function '{}' has no transitions defined (stays on current node)",
                    fn_name
                );
            }
        }

        // 4. Check for orphan nodes (unreachable from initial_node)
        let mut reachable = HashSet::new();
        let mut queue = vec![def.initial_node.clone()];
        while let Some(node_name) = queue.pop() {
            if !reachable.insert(node_name.clone()) {
                continue;
            }
            if let Some(node) = def.nodes.get(&node_name) {
                for fn_name in &node.functions {
                    if let Some(fn_def) = def.functions.get(fn_name) {
                        for target in fn_def.transitions.values() {
                            if !reachable.contains(target) {
                                queue.push(target.clone());
                            }
                        }
                    }
                }
            }
        }
        for node_name in def.nodes.keys() {
            if !reachable.contains(node_name) {
                errors.push(format!(
                    "node '{}' is unreachable from initial_node '{}'",
                    node_name, def.initial_node
                ));
            }
        }

        // 5. Every top-level function should be referenced by at least one node
        let referenced: HashSet<&str> = def.nodes.values()
            .flat_map(|n| n.functions.iter().map(|s| s.as_str()))
            .collect();
        for fn_name in def.functions.keys() {
            if !referenced.contains(fn_name.as_str()) {
                log::warn!(
                    "Dhara: function '{}' is defined but not referenced by any node",
                    fn_name
                );
            }
        }

        // 6. Task message roles must be valid
        for (node_name, node) in &def.nodes {
            for (i, msg) in node.task_messages.iter().enumerate() {
                match msg.role.as_str() {
                    "system" | "user" | "assistant" => {}
                    other => {
                        errors.push(format!(
                            "node '{}' task_message[{}] has invalid role '{}'",
                            node_name, i, other
                        ));
                    }
                }
            }
        }

        if errors.is_empty() {
            log::info!(
                "Dhara: flow '{}' validated — {} nodes, {} functions",
                def.id, def.nodes.len(), def.functions.len()
            );
            Ok(())
        } else {
            Err(DharaError::ValidationErrors(errors))
        }
    }

    // -----------------------------------------------------------------------
    // Build — creates runtime pieces for one connection
    // -----------------------------------------------------------------------

    /// Build the runtime pieces for a single connection.
    ///
    /// `handler_registry` contains the handler implementations from `functions.rs`.
    /// `state` is the flow-specific state (e.g. `InterviewState`).
    /// `conn_id` is the connection identifier for logging.
    ///
    /// Returns a `DharaBuild` with context, LLM registry, and transition hook
    /// ready to plug into the pipeline.
    ///
    /// # Errors
    ///
    /// Returns `DharaError::ValidationErrors` if any function in `dhara.json`
    /// is missing from the handler registry.
    pub fn build(
        &self,
        handler_registry: &DharaFunctionRegistry,
        state: Arc<dyn Any + Send + Sync>,
        conn_id: u64,
    ) -> Result<DharaBuild, DharaError> {
        // Cross-validate: every function in JSON must have a handler
        let mut missing = Vec::new();
        for fn_name in self.def.functions.keys() {
            if !handler_registry.contains(fn_name) {
                missing.push(format!(
                    "function '{}' defined in dhara.json but no handler registered in functions.rs",
                    fn_name
                ));
            }
        }
        if !missing.is_empty() {
            return Err(DharaError::ValidationErrors(missing));
        }

        // Create shared pieces
        let context = Arc::new(Mutex::new(LLMContext::new(None)));
        let llm_registry = Arc::new(Mutex::new(FunctionRegistry::new()));
        let dhara_ctx = DharaContext::new(state, conn_id);

        // Pending transition state
        let pending_transition: Arc<Mutex<Option<(String, Option<String>)>>> =
            Arc::new(Mutex::new(None));

        // Apply initial node
        let initial = &self.def.initial_node;
        self.apply_node_to_context(&context, initial);
        dhara_ctx.set_current_node(initial);
        self.build_llm_registry_for_node(
            &llm_registry,
            initial,
            handler_registry,
            &dhara_ctx,
            &pending_transition,
        );

        // Build transition hook
        let hook = self.create_transition_hook(
            context.clone(),
            llm_registry.clone(),
            handler_registry,
            dhara_ctx.clone(),
            pending_transition,
        );

        Ok(DharaBuild {
            context,
            llm_registry,
            hook,
            dhara_ctx,
        })
    }

    // -----------------------------------------------------------------------
    // Internal: apply node config to LLM context
    // -----------------------------------------------------------------------

    fn apply_node_to_context(
        &self,
        context: &Arc<Mutex<LLMContext>>,
        node_name: &str,
    ) {
        let node = &self.def.nodes[node_name];
        let mut ctx = context.lock().unwrap();

        // System prompt: override or universal
        let prompt = node.system_prompt_override
            .as_deref()
            .unwrap_or(&self.def.system_prompt);
        ctx.system_prompt = Some(prompt.to_string());

        // Context strategy
        match node.context_strategy {
            ContextStrategyDef::Reset => ctx.messages.clear(),
            ContextStrategyDef::Append => {}
        }

        // Task messages
        for msg in &node.task_messages {
            ctx.push_message(task_message_to_message(msg));
        }

        // Tools — build from function schemas in JSON
        let tools: Vec<FunctionSchema> = node.functions.iter()
            .filter_map(|fn_name| {
                self.def.functions.get(fn_name).map(|fn_def| {
                    FunctionSchema::new(fn_name, &fn_def.description)
                        .with_parameters(fn_def.parameters.clone())
                })
            })
            .collect();

        ctx.tools = if tools.is_empty() {
            None
        } else {
            Some(ToolsSchema::new(tools))
        };
        ctx.tool_choice = None;
    }

    // -----------------------------------------------------------------------
    // Internal: build LLM function registry for a node
    // -----------------------------------------------------------------------

    fn build_llm_registry_for_node(
        &self,
        llm_registry: &Arc<Mutex<FunctionRegistry>>,
        node_name: &str,
        handler_registry: &DharaFunctionRegistry,
        dhara_ctx: &DharaContext,
        pending: &Arc<Mutex<Option<(String, Option<String>)>>>,
    ) {
        let node = &self.def.nodes[node_name];
        let mut reg = llm_registry.lock().unwrap();
        *reg = FunctionRegistry::new();

        for fn_name in &node.functions {
            let handler = handler_registry.get(fn_name).unwrap().clone();
            let ctx = dhara_ctx.clone();
            let pending = pending.clone();
            let fn_name_owned = fn_name.clone();

            reg.register(fn_name.clone(), move |args: String| {
                let handler = handler.clone();
                let ctx = ctx.clone();
                let pending = pending.clone();
                let fn_name = fn_name_owned.clone();

                async move {
                    let result = handler(args, ctx).await;
                    let status = result.status.clone();

                    // Store pending transition (function name + status)
                    *pending.lock().unwrap() = Some((fn_name, status));

                    result.result
                }
            });
        }
    }

    // -----------------------------------------------------------------------
    // Internal: create the transition hook for OpenAILLMHandler
    // -----------------------------------------------------------------------

    fn create_transition_hook(
        &self,
        context: Arc<Mutex<LLMContext>>,
        llm_registry: Arc<Mutex<FunctionRegistry>>,
        handler_registry: &DharaFunctionRegistry,
        dhara_ctx: DharaContext,
        pending: Arc<Mutex<Option<(String, Option<String>)>>>,
    ) -> TransitionHook {
        // Clone what the closure needs
        let def = self.clone_def();
        let handlers = self.clone_handlers(handler_registry);

        Arc::new(move |_ctx: &Arc<Mutex<LLMContext>>| {
            let transition_info = pending.lock().unwrap().take();

            if let Some((fn_name, status)) = transition_info {
                // Look up the transition target from the JSON
                let target_node = def.functions.get(&fn_name)
                    .and_then(|fn_def| {
                        let key = status.as_deref().unwrap_or("default");
                        fn_def.transitions.get(key)
                            .or_else(|| fn_def.transitions.get("default"))
                    })
                    .cloned();

                if let Some(node_name) = target_node {
                    log::info!(
                        "Dhara [{}]: transitioning to '{}' (triggered by '{}')",
                        def.id, node_name, fn_name
                    );

                    // 1. Apply node context
                    apply_node_static(&def, &context, &node_name);

                    // 2. Update dhara context
                    dhara_ctx.set_current_node(&node_name);

                    // 3. Rebuild LLM registry
                    if let Some(node) = def.nodes.get(&node_name) {
                        let mut reg = llm_registry.lock().unwrap();
                        *reg = FunctionRegistry::new();

                        for fn_ref in &node.functions {
                            if let Some(handler) = handlers.get(fn_ref) {
                                let handler = handler.clone();
                                let ctx = dhara_ctx.clone();
                                let pending = pending.clone();
                                let fn_name_owned = fn_ref.clone();

                                reg.register(fn_ref.clone(), move |args: String| {
                                    let handler = handler.clone();
                                    let ctx = ctx.clone();
                                    let pending = pending.clone();
                                    let fn_name = fn_name_owned.clone();

                                    async move {
                                        let result = handler(args, ctx).await;
                                        let status = result.status.clone();
                                        *pending.lock().unwrap() = Some((fn_name, status));
                                        result.result
                                    }
                                });
                            }
                        }
                    }

                    log::info!("Dhara [{}]: transition to '{}' complete", def.id, node_name);
                }
                // If no transition target found, stay on current node (no-op)
            }
        })
    }

    // -----------------------------------------------------------------------
    // Helpers for cloning into closures
    // -----------------------------------------------------------------------

    fn clone_def(&self) -> DharaFlowDef {
        // Re-serialize and deserialize for a clean clone.
        // DharaFlowDef could derive Clone, but this avoids adding it
        // to all sub-types for now.
        let json = serde_json::to_string(&self.def)
            .expect("DharaFlowDef should serialize");
        serde_json::from_str(&json)
            .expect("DharaFlowDef should deserialize")
    }

    fn clone_handlers(
        &self,
        registry: &DharaFunctionRegistry,
    ) -> HashMap<String, DharaHandlerFn> {
        self.def.functions.keys()
            .filter_map(|name| {
                registry.get(name).map(|h| (name.clone(), h.clone()))
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Helper: convert TaskMessage to context Message
// ---------------------------------------------------------------------------

fn task_message_to_message(msg: &TaskMessage) -> Message {
    match msg.role.as_str() {
        "user" => Message::User { content: msg.content.clone() },
        "assistant" => Message::Assistant {
            content: Some(msg.content.clone()),
            tool_calls: None,
        },
        _ => Message::System { content: msg.content.clone() },
    }
}

// ---------------------------------------------------------------------------
// Static helper (used inside closure where &self isn't available)
// ---------------------------------------------------------------------------

fn apply_node_static(
    def: &DharaFlowDef,
    context: &Arc<Mutex<LLMContext>>,
    node_name: &str,
) {
    let node = match def.nodes.get(node_name) {
        Some(n) => n,
        None => return,
    };

    let mut ctx = context.lock().unwrap();

    let prompt = node.system_prompt_override
        .as_deref()
        .unwrap_or(&def.system_prompt);
    ctx.system_prompt = Some(prompt.to_string());

    match node.context_strategy {
        ContextStrategyDef::Reset => ctx.messages.clear(),
        ContextStrategyDef::Append => {}
    }

    for msg in &node.task_messages {
        ctx.push_message(task_message_to_message(msg));
    }

    let tools: Vec<FunctionSchema> = node.functions.iter()
        .filter_map(|fn_name| {
            def.functions.get(fn_name).map(|fn_def| {
                FunctionSchema::new(fn_name, &fn_def.description)
                    .with_parameters(fn_def.parameters.clone())
            })
        })
        .collect();

    ctx.tools = if tools.is_empty() { None } else { Some(ToolsSchema::new(tools)) };
    ctx.tool_choice = None;
}

// ---------------------------------------------------------------------------
// Build result
// ---------------------------------------------------------------------------

/// Output of `Dhara::build()` — everything needed to wire into a pipeline.
pub struct DharaBuild {
    /// Shared LLM context (pass to aggregators and RAVI).
    pub context: Arc<Mutex<LLMContext>>,

    /// Shared LLM function registry (pass to `OpenAILLMHandler::with_shared_registry`).
    pub llm_registry: Arc<Mutex<FunctionRegistry>>,

    /// Transition hook (pass to `handler.set_transition_hook(hook)`).
    pub hook: TransitionHook,

    /// Dhara context — call `set_push_sender()` after `PipelineTask::new()`.
    pub dhara_ctx: DharaContext,
}
