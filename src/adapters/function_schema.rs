//! Provider-agnostic function (tool) schema.
//!
//! Mirrors pipecat's `FunctionSchema` — a single callable tool the LLM can invoke.
//! Parameters are stored as raw JSON Schema (`serde_json::Value`) to stay
//! provider-neutral; each adapter converts to its wire format.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A single function the LLM may call.
///
/// Follows the OpenAI function-calling schema shape (which has become the
/// de-facto standard across providers), but is not tied to any provider.
///
/// # Example
/// ```rust
/// use serde_json::json;
/// use rustvani::adapters::schemas::FunctionSchema;
///
/// let get_weather = FunctionSchema::new("get_weather", "Get current weather for a city")
///     .with_parameters(json!({
///         "type": "object",
///         "properties": {
///             "city": { "type": "string", "description": "City name" }
///         },
///         "required": ["city"]
///     }));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionSchema {
    /// Tool name — must be unique within a single request.
    pub name: String,

    /// Human-readable description shown to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// JSON Schema describing the function's parameters.
    /// `None` means the function takes no arguments.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,

    /// If `true`, the model must follow the schema exactly (OpenAI structured outputs).
    /// Not all providers support this — adapters may ignore it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl FunctionSchema {
    /// Create a new function schema with name and description.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: Some(description.into()),
            parameters: None,
            strict: None,
        }
    }

    /// Builder: attach JSON Schema parameters.
    pub fn with_parameters(mut self, params: Value) -> Self {
        self.parameters = Some(params);
        self
    }

    /// Builder: enable strict schema enforcement.
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }

    /// Convert to the "default dict" shape expected by OpenAI's
    /// `ChatCompletionToolParam.function` field.
    ///
    /// ```json
    /// { "name": "...", "description": "...", "parameters": {...}, "strict": true }
    /// ```
    pub fn to_default_dict(&self) -> Value {
        let mut map = serde_json::Map::new();
        map.insert("name".into(), Value::String(self.name.clone()));
        if let Some(desc) = &self.description {
            map.insert("description".into(), Value::String(desc.clone()));
        }
        if let Some(params) = &self.parameters {
            map.insert("parameters".into(), params.clone());
        }
        if let Some(strict) = self.strict {
            map.insert("strict".into(), Value::Bool(strict));
        }
        Value::Object(map)
    }
}
