//! Built-in tool trait with lifecycle management.
//!
//! Every built-in tool implements `BuiltinTool`. The LLM handler manages
//! the lifecycle:
//!
//! ```text
//! Pipeline build:
//!   handler.add_tool(tool)          → register_all() called
//!
//! StartFrame:
//!   tool.on_start(cancel_token)     → connect, cache, spawn background tasks
//!
//! Running:
//!   handlers execute via registry   → OnceCell reads, pool connections
//!
//! EndFrame:
//!   tool.on_stop()                  → flush caches, return connections, log
//!
//! CancelFrame:
//!   tool.on_cancel()                → cancel in-flight work, then on_stop()
//!   (also: cancel_token triggers    → background tasks exit via select!)
//! ```
//!
//! # Cancellation token
//!
//! The handler creates a `CancellationToken` and passes a child to each
//! tool's `on_start()`. Background tasks (`tokio::spawn`) should select
//! on the token so they exit cleanly:
//!
//! ```rust,ignore
//! tokio::spawn(async move {
//!     tokio::select! {
//!         _ = cancel.cancelled() => {
//!             log::info!("background task cancelled");
//!         }
//!         result = connection.await => {
//!             if let Err(e) = result {
//!                 log::error!("connection error: {}", e);
//!             }
//!         }
//!     }
//! });
//! ```
//!
//! # Cacheable vs non-cacheable
//!
//! - **Cacheable** (`is_cacheable() → true`): needs async init on StartFrame.
//!   Example: Postgres (connect + schema introspection).
//! - **Non-cacheable** (`is_cacheable() → false`): ready immediately.
//!   `on_start()` is a no-op. Example: a calculator tool.
//!
//! Both types receive lifecycle hooks — a non-cacheable tool might still
//! need `on_stop()` to flush metrics or log stats.

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::adapters::schemas::FunctionSchema;
use crate::error::Result;
use crate::services::llm::function_registry::FunctionRegistry;

// ---------------------------------------------------------------------------
// Lifecycle state (for debug / Drop warnings)
// ---------------------------------------------------------------------------

/// Tracks which lifecycle hooks have been called.
///
/// Tools can use this to log warnings in `Drop` if `on_stop` was never
/// called (i.e., someone forgot to handle EndFrame).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolLifecycleState {
    /// Just constructed. `on_start` not yet called.
    Created,
    /// `on_start` completed successfully. Running.
    Started,
    /// `on_stop` completed. Fully cleaned up.
    Stopped,
    /// `on_cancel` was called (which internally calls `on_stop`).
    Cancelled,
}

impl std::fmt::Display for ToolLifecycleState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Created => write!(f, "created"),
            Self::Started => write!(f, "started"),
            Self::Stopped => write!(f, "stopped"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

// ---------------------------------------------------------------------------
// BuiltinTool trait
// ---------------------------------------------------------------------------

/// A built-in tool that can be attached to an LLM handler.
///
/// Implements lifecycle hooks for startup, shutdown, and cancellation.
/// The handler orchestrates — tools manage their own resources.
#[async_trait]
pub trait BuiltinTool: Send + Sync {
    /// Human-readable name for logs (e.g. `"neon_postgres"`).
    fn name(&self) -> &str;

    /// Whether this tool needs async initialization on StartFrame.
    fn is_cacheable(&self) -> bool {
        false
    }

    // -------------------------------------------------------------------
    // Lifecycle hooks
    // -------------------------------------------------------------------

    /// Called when `StartFrame` flows through the pipeline.
    ///
    /// The handler passes a child `CancellationToken`. Tools should:
    /// 1. Store the token (e.g. in `OnceCell<CancellationToken>`)
    /// 2. Use it in any `tokio::spawn` via `tokio::select!`
    /// 3. Connect to external services, populate caches
    ///
    /// Non-cacheable tools can skip this (default is no-op).
    ///
    /// # Errors
    /// If this fails, the handler logs the error and continues without
    /// this tool's handlers. The tool is left in `Created` state.
    async fn on_start(&self, cancel: CancellationToken) -> Result<()> {
        let _ = cancel; // suppress unused warning for non-cacheable tools
        Ok(())
    }

    /// Called when `EndFrame` flows through the pipeline. Graceful shutdown.
    ///
    /// Tools should:
    /// - Flush session-specific caches
    /// - Return pool connections
    /// - Log final stats/metrics
    /// - Cancel background tasks (via stored CancellationToken)
    ///
    /// Default is no-op. Override if your tool holds resources.
    async fn on_stop(&self) -> Result<()> {
        Ok(())
    }

    /// Called when `CancelFrame` flows through the pipeline. Abrupt shutdown.
    ///
    /// **Default implementation calls `on_stop()`.** Override only if your
    /// tool needs to do something extra before stopping (e.g. cancel
    /// in-flight database queries via `cancel_token()`).
    ///
    /// The `CancellationToken` passed in `on_start()` is also triggered
    /// by the handler before calling `on_cancel()`, so background tasks
    /// using `select!` will already be exiting.
    ///
    /// # Override pattern
    /// ```rust,ignore
    /// async fn on_cancel(&self) -> Result<()> {
    ///     // Cancel in-flight queries
    ///     if let Some(cancel_token) = self.pg_cancel_token.get() {
    ///         cancel_token.cancel_query(tokio_postgres::NoTls).await.ok();
    ///     }
    ///     // Then do normal cleanup
    ///     self.on_stop().await
    /// }
    /// ```
    async fn on_cancel(&self) -> Result<()> {
        self.on_stop().await
    }

    // -------------------------------------------------------------------
    // Registration
    // -------------------------------------------------------------------

    /// Return the function schemas for this tool.
    ///
    /// These are merged into the `ToolsSchema` sent to the LLM.
    fn tool_schemas(&self) -> Vec<FunctionSchema>;

    /// Register handler functions into the shared registry.
    ///
    /// Called at `add_tool()` time. Handlers capture `Arc<OnceCell<...>>`
    /// references that are populated later in `on_start()`.
    fn register_all(&self, registry: &mut FunctionRegistry);
}
