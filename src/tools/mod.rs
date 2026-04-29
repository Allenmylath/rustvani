//! Built-in tools for LLM function calling.
//!
//! Each tool implements `BuiltinTool`. The LLM handler holds tools and
//! manages their lifecycle:
//!
//! - **Non-cacheable** tools are ready at construction.
//! - **Cacheable** tools are initialized on `StartFrame` (connect, introspect, etc.)
//!
//! ```text
//! tools/
//!   mod.rs              ← you are here
//!   traits.rs           ← BuiltinTool trait
//!   postgres/
//!     mod.rs            ← NeonPostgresTool (cacheable)
//!     cache.rs          ← SchemaCache + ResultSetCache
//!     filter.rs         ← Structured filter → parameterized SQL
//! ```

pub mod postgres;
pub mod traits;

pub use postgres::NeonPostgresTool;
pub use traits::BuiltinTool;
