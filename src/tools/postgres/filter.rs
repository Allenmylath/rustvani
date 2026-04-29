//! Structured filter → parameterized SQL builder.
//!
//! The LLM sends structured JSON filters (validated by strict mode),
//! and this module converts them to safe, parameterized SQL WHERE clauses.
//! No raw SQL from the LLM ever touches the query.
//!
//! # Supported operators
//!
//! | Operator       | SQL               | Value required? |
//! |----------------|-------------------|-----------------|
//! | `=`            | `col = $N`        | Yes             |
//! | `!=`           | `col != $N`       | Yes             |
//! | `<`            | `col < $N`        | Yes             |
//! | `>`            | `col > $N`        | Yes             |
//! | `<=`           | `col <= $N`       | Yes             |
//! | `>=`           | `col >= $N`       | Yes             |
//! | `LIKE`         | `col LIKE $N`     | Yes             |
//! | `ILIKE`        | `col ILIKE $N`    | Yes             |
//! | `IN`           | `col IN (...)`    | Yes (array)     |
//! | `@>`           | `col @> $N`       | Yes (jsonb)     |
//! | `IS NULL`      | `col IS NULL`     | No              |
//! | `IS NOT NULL`  | `col IS NOT NULL` | No              |

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::cache::SchemaCache;

// ---------------------------------------------------------------------------
// Filter types
// ---------------------------------------------------------------------------

/// A single filter condition from the LLM.
///
/// Deserialized from strict-mode JSON, so the shape is guaranteed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCondition {
    /// Column name — validated against the schema cache.
    pub column: String,
    /// Operator — must be from the allowed set.
    pub op: String,
    /// Value to compare against. Optional for `IS NULL` / `IS NOT NULL`.
    #[serde(default)]
    pub value: Option<Value>,
}

/// Full refinement request from the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefineRequest {
    /// ID of the result set to refine.
    pub result_set_id: String,
    /// Filter conditions to apply.
    pub filters: Vec<FilterCondition>,
    /// Optional ORDER BY column (validated against schema).
    #[serde(default)]
    pub order_by: Option<String>,
    /// Optional LIMIT.
    #[serde(default)]
    pub limit: Option<i64>,
}

// ---------------------------------------------------------------------------
// Allowed operators
// ---------------------------------------------------------------------------

const ALLOWED_OPS: &[&str] = &[
    "=", "!=", "<", ">", "<=", ">=",
    "LIKE", "ILIKE", "IN", "@>",
    "IS NULL", "IS NOT NULL",
];

/// Operators that don't take a value.
const NULLARY_OPS: &[&str] = &["IS NULL", "IS NOT NULL"];

// ---------------------------------------------------------------------------
// SQL identifier validation
// ---------------------------------------------------------------------------

/// Check that a column name is a valid SQL identifier (no injection).
///
/// Only allows alphanumeric + underscore. No dots, quotes, spaces, etc.
fn is_valid_identifier(s: &str) -> bool {
    !s.is_empty()
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && s.chars().next().map_or(false, |c| c.is_ascii_alphabetic() || c == '_')
}

// ---------------------------------------------------------------------------
// Filter builder
// ---------------------------------------------------------------------------

/// Error from filter building.
#[derive(Debug, thiserror::Error)]
pub enum FilterError {
    #[error("invalid operator: '{0}' (allowed: {ALLOWED_OPS:?})")]
    InvalidOperator(String),

    #[error("column '{0}' not found in table '{1}'")]
    UnknownColumn(String, String),

    #[error("invalid column name: '{0}' (must be alphanumeric/underscore)")]
    InvalidIdentifier(String),

    #[error("operator '{0}' requires a value")]
    MissingValue(String),

    #[error("IN operator requires an array value")]
    InRequiresArray,

    #[error("order_by column '{0}' not found in table '{1}'")]
    InvalidOrderBy(String, String),
}

/// Built query fragment ready for execution.
#[derive(Debug)]
pub struct BuiltFilter {
    /// SQL WHERE clause (e.g. `"col1 = $1 AND col2 > $2"`).
    /// Empty string if no filters.
    pub where_clause: String,
    /// Parameter values in order ($1, $2, ...).
    /// Stored as strings — tokio-postgres accepts `&str` for text params
    /// and we cast in SQL for other types.
    pub params: Vec<String>,
    /// Optional ORDER BY clause (e.g. `"ORDER BY price ASC"`).
    pub order_by: Option<String>,
    /// Optional LIMIT clause.
    pub limit: Option<i64>,
}

/// Build a parameterized WHERE clause from structured filters.
///
/// Validates columns against the schema cache and operators against the
/// allowed set. Returns a `BuiltFilter` with the clause and params.
pub fn build_filter(
    table: &str,
    filters: &[FilterCondition],
    order_by: Option<&str>,
    limit: Option<i64>,
    schema_cache: &SchemaCache,
    param_offset: usize,
) -> std::result::Result<BuiltFilter, FilterError> {
    let table_cols = schema_cache.columns_for_table(table);

    let mut conditions = Vec::new();
    let mut params = Vec::new();
    let mut param_idx = param_offset + 1; // $1-based

    for filter in filters {
        // Validate operator
        let op_upper = filter.op.to_uppercase();
        if !ALLOWED_OPS.contains(&op_upper.as_str()) {
            return Err(FilterError::InvalidOperator(filter.op.clone()));
        }

        // Validate column name
        if !is_valid_identifier(&filter.column) {
            return Err(FilterError::InvalidIdentifier(filter.column.clone()));
        }

        // Validate column exists in table (if schema is available)
        if let Some(cols) = &table_cols {
            if !cols.iter().any(|c| c.name == filter.column) {
                return Err(FilterError::UnknownColumn(
                    filter.column.clone(),
                    table.to_string(),
                ));
            }
        }

        // Build condition
        if NULLARY_OPS.contains(&op_upper.as_str()) {
            conditions.push(format!("\"{}\" {}", filter.column, op_upper));
        } else {
            let value = filter
                .value
                .as_ref()
                .ok_or_else(|| FilterError::MissingValue(filter.op.clone()))?;

            match op_upper.as_str() {
                "IN" => {
                    let arr = value.as_array().ok_or(FilterError::InRequiresArray)?;
                    let placeholders: Vec<String> = arr
                        .iter()
                        .map(|v| {
                            let p = format!("${}", param_idx);
                            params.push(json_value_to_string(v));
                            param_idx += 1;
                            p
                        })
                        .collect();
                    conditions.push(format!(
                        "\"{}\" IN ({})",
                        filter.column,
                        placeholders.join(", ")
                    ));
                }
                "@>" => {
                    // JSONB contains — value is passed as JSON string
                    let json_str = serde_json::to_string(value)
                        .unwrap_or_else(|_| "null".to_string());
                    conditions.push(format!(
                        "\"{}\" @> ${}::jsonb",
                        filter.column, param_idx
                    ));
                    params.push(json_str);
                    param_idx += 1;
                }
                _ => {
                    conditions.push(format!(
                        "\"{}\" {} ${}",
                        filter.column, op_upper, param_idx
                    ));
                    params.push(json_value_to_string(value));
                    param_idx += 1;
                }
            }
        }
    }

    // Validate ORDER BY
    let order_clause = if let Some(ob) = order_by {
        if !is_valid_identifier(ob) {
            return Err(FilterError::InvalidIdentifier(ob.to_string()));
        }
        if let Some(cols) = &table_cols {
            if !cols.iter().any(|c| c.name == ob) {
                return Err(FilterError::InvalidOrderBy(
                    ob.to_string(),
                    table.to_string(),
                ));
            }
        }
        Some(format!("ORDER BY \"{}\"", ob))
    } else {
        None
    };

    let where_clause = if conditions.is_empty() {
        String::new()
    } else {
        conditions.join(" AND ")
    };

    Ok(BuiltFilter {
        where_clause,
        params,
        order_by: order_clause,
        limit,
    })
}

/// Convert a serde_json::Value to a string for parameterized queries.
fn json_value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "NULL".to_string(),
        // Arrays and objects — serialize as JSON string
        other => serde_json::to_string(other).unwrap_or_else(|_| "null".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_schema() -> SchemaCache {
        use super::super::cache::{CachedColumn, CachedTable};
        let mut cache = SchemaCache::empty();
        cache.tables.insert(
            "menu_items".to_string(),
            CachedTable {
                schema: "public".to_string(),
                name: "menu_items".to_string(),
                pk_column: Some("id".to_string()),
                columns: vec![
                    CachedColumn { name: "id".into(), data_type: "integer".into(), is_nullable: false, column_default: None },
                    CachedColumn { name: "name".into(), data_type: "text".into(), is_nullable: false, column_default: None },
                    CachedColumn { name: "price".into(), data_type: "numeric".into(), is_nullable: false, column_default: None },
                    CachedColumn { name: "vegetarian".into(), data_type: "boolean".into(), is_nullable: false, column_default: None },
                    CachedColumn { name: "allergens".into(), data_type: "jsonb".into(), is_nullable: true, column_default: None },
                    CachedColumn { name: "calories".into(), data_type: "integer".into(), is_nullable: true, column_default: None },
                ],
                vector_columns: vec![],
            },
        );
        cache
    }

    #[test]
    fn test_simple_equality() {
        let schema = mock_schema();
        let filters = vec![FilterCondition {
            column: "vegetarian".into(),
            op: "=".into(),
            value: Some(Value::Bool(true)),
        }];

        let result = build_filter("menu_items", &filters, None, None, &schema, 0).unwrap();
        assert_eq!(result.where_clause, "\"vegetarian\" = $1");
        assert_eq!(result.params, vec!["true"]);
    }

    #[test]
    fn test_multiple_filters() {
        let schema = mock_schema();
        let filters = vec![
            FilterCondition {
                column: "vegetarian".into(),
                op: "=".into(),
                value: Some(Value::Bool(true)),
            },
            FilterCondition {
                column: "calories".into(),
                op: "<".into(),
                value: Some(serde_json::json!(500)),
            },
        ];

        let result = build_filter("menu_items", &filters, Some("price"), Some(10), &schema, 0).unwrap();
        assert_eq!(result.where_clause, "\"vegetarian\" = $1 AND \"calories\" < $2");
        assert_eq!(result.order_by, Some("ORDER BY \"price\"".to_string()));
        assert_eq!(result.limit, Some(10));
    }

    #[test]
    fn test_unknown_column_rejected() {
        let schema = mock_schema();
        let filters = vec![FilterCondition {
            column: "nonexistent".into(),
            op: "=".into(),
            value: Some(Value::Bool(true)),
        }];

        let result = build_filter("menu_items", &filters, None, None, &schema, 0);
        assert!(matches!(result, Err(FilterError::UnknownColumn(_, _))));
    }

    #[test]
    fn test_invalid_operator_rejected() {
        let schema = mock_schema();
        let filters = vec![FilterCondition {
            column: "name".into(),
            op: "DROP TABLE".into(),
            value: Some(Value::String("lol".into())),
        }];

        let result = build_filter("menu_items", &filters, None, None, &schema, 0);
        assert!(matches!(result, Err(FilterError::InvalidOperator(_))));
    }

    #[test]
    fn test_jsonb_contains() {
        let schema = mock_schema();
        let filters = vec![FilterCondition {
            column: "allergens".into(),
            op: "@>".into(),
            value: Some(serde_json::json!(["peanuts"])),
        }];

        let result = build_filter("menu_items", &filters, None, None, &schema, 0).unwrap();
        assert_eq!(result.where_clause, "\"allergens\" @> $1::jsonb");
        assert_eq!(result.params, vec!["[\"peanuts\"]"]);
    }
}
