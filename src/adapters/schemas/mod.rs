//! Provider-agnostic tool/function schemas.

pub mod function_schema;
pub mod tools_schema;

pub use function_schema::FunctionSchema;
pub use tools_schema::{AdapterType, ToolChoice, ToolsSchema};
