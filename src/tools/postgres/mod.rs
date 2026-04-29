//! Neon Postgres built-in tool for LLM function calling.
//!
//! **Cacheable tool** — construction is cheap (stores config only).
//! Connection and schema introspection happen in `on_start()`.
//!
//! # Lifecycle
//!
//! ```text
//! new(config)           → cheap, no I/O
//! handler.add_tool()    → register_all() wires handlers with OnceCell refs
//! StartFrame            → on_start(cancel) connects, caches schema, spawns conn task
//! ... frames flow ...   → handlers fire, read from OnceCells, use pool
//! EndFrame              → on_stop() flushes result cache, cancels background tasks
//! CancelFrame           → on_cancel() cancels in-flight queries, then on_stop()
//! ```
//!
//! # Data flow
//!
//! ```text
//! LLM calls pg_query("SELECT * FROM menu_items")
//!     ├── summary → LLM context: "result_set: rs_001, 200 items found"
//!     └── raw data → FunctionCallRawResultFrame → UI
//!
//! LLM calls pg_refine(rs_001, [{column: "vegetarian", op: "=", value: true}])
//!     ├── summary → LLM: "result_set: rs_002, 73 items found"
//!     └── raw data → UI updates
//! ```

pub mod cache;
pub mod filter;

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use log;
use native_tls::TlsConnector;
use postgres_native_tls::MakeTlsConnector;
use serde_json::{json, Value};
use tokio::sync::OnceCell;
use tokio_postgres::{types::ToSql, Client, Row};
use tokio_util::sync::CancellationToken;

use cache::{
    CachedColumn, CachedTable, ResultSetCache, SchemaCache, VectorColumnInfo,
};
use filter::{build_filter, RefineRequest};

use crate::adapters::schemas::FunctionSchema;
use crate::error::{PipecatError, Result};
use crate::services::llm::function_registry::{FunctionRegistry, ToolCallOutput};
use super::traits::{BuiltinTool, ToolLifecycleState};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for the Neon Postgres tool.
#[derive(Debug, Clone)]
pub struct NeonPostgresConfig {
    pub connection_url: String,
    pub max_result_sets: usize,
    pub read_only: bool,
    pub statement_timeout_ms: Option<u32>,
    pub schemas: Vec<String>,
}

impl NeonPostgresConfig {
    pub fn new(connection_url: impl Into<String>) -> Self {
        Self {
            connection_url: connection_url.into(),
            max_result_sets: 10,
            read_only: false,
            statement_timeout_ms: Some(10_000),
            schemas: vec!["public".to_string()],
        }
    }

    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    pub fn with_statement_timeout_ms(mut self, ms: u32) -> Self {
        self.statement_timeout_ms = Some(ms);
        self
    }

    pub fn with_schemas(mut self, schemas: Vec<String>) -> Self {
        self.schemas = schemas;
        self
    }

    pub fn with_max_result_sets(mut self, max: usize) -> Self {
        self.max_result_sets = max;
        self
    }
}

// ---------------------------------------------------------------------------
// NeonPostgresTool
// ---------------------------------------------------------------------------

/// Built-in Postgres tool for Neon DB with pgvector support.
///
/// Cheap to construct. Real work happens in lifecycle hooks.
pub struct NeonPostgresTool {
    config: NeonPostgresConfig,

    // --- Populated on on_start() ---
    client: Arc<OnceCell<Client>>,
    schema_cache: Arc<OnceCell<SchemaCache>>,

    // --- Available immediately, cleared on on_stop() ---
    result_cache: Arc<Mutex<ResultSetCache>>,

    // --- Lifecycle management ---
    cancel_token: OnceCell<CancellationToken>,
    state: Mutex<ToolLifecycleState>,
}

impl NeonPostgresTool {
    /// Create with explicit config. **No connection — just stores config.**
    pub fn new(config: NeonPostgresConfig) -> Self {
        let max_rs = config.max_result_sets;
        Self {
            config,
            client: Arc::new(OnceCell::new()),
            schema_cache: Arc::new(OnceCell::new()),
            result_cache: Arc::new(Mutex::new(ResultSetCache::new(max_rs))),
            cancel_token: OnceCell::new(),
            state: Mutex::new(ToolLifecycleState::Created),
        }
    }

    /// Create from `DATABASE_URL` env var. **No connection.**
    pub fn from_env() -> Self {
        let url = std::env::var("DATABASE_URL")
            .expect("DATABASE_URL env var not set");
        Self::new(NeonPostgresConfig::new(url))
    }

    /// Create from a connection URL. **No connection.**
    pub fn from_url(url: impl Into<String>) -> Self {
        Self::new(NeonPostgresConfig::new(url))
    }

    /// Get schema cache (available after `on_start()`).
    pub fn schema_cache(&self) -> Option<&SchemaCache> {
        self.schema_cache.get()
    }

    /// Current lifecycle state.
    pub fn lifecycle_state(&self) -> ToolLifecycleState {
        *self.state.lock().unwrap()
    }

    fn set_state(&self, new_state: ToolLifecycleState) {
        *self.state.lock().unwrap() = new_state;
    }

    fn require_client(client: &Arc<OnceCell<Client>>) -> std::result::Result<&Client, String> {
        client.get().ok_or_else(|| {
            "NeonPostgresTool not initialized. StartFrame must flow first.".to_string()
        })
    }

    fn require_schema(cache: &Arc<OnceCell<SchemaCache>>) -> std::result::Result<&SchemaCache, String> {
        cache.get().ok_or_else(|| {
            "Schema cache not loaded. StartFrame must flow first.".to_string()
        })
    }

    // -----------------------------------------------------------------------
    // Schema introspection
    // -----------------------------------------------------------------------

    async fn introspect_schema(
        client: &Client,
        schemas: &[String],
    ) -> std::result::Result<SchemaCache, Box<dyn std::error::Error + Send + Sync>> {
        let mut tables: std::collections::HashMap<String, CachedTable> =
            std::collections::HashMap::new();

        let rows = client.query(cache::introspect::COLUMNS_QUERY, &[]).await?;
        for row in &rows {
            let table_schema: String = row.get("table_schema");
            if !schemas.contains(&table_schema) { continue; }
            let table_name: String = row.get("table_name");
            let col = CachedColumn {
                name: row.get("column_name"),
                data_type: row.get("data_type"),
                is_nullable: row.get::<_, String>("is_nullable") == "YES",
                column_default: row.get("column_default"),
            };
            tables.entry(table_name.clone())
                .or_insert_with(|| CachedTable {
                    schema: table_schema, name: table_name,
                    pk_column: None, columns: Vec::new(), vector_columns: Vec::new(),
                })
                .columns.push(col);
        }

        let pk_rows = client.query(cache::introspect::PK_QUERY, &[]).await?;
        for row in &pk_rows {
            let table_name: String = row.get("table_name");
            let col_name: String = row.get("column_name");
            if let Some(table) = tables.get_mut(&table_name) {
                table.pk_column = Some(col_name);
            }
        }

        match client.query(cache::introspect::VECTOR_COLUMNS_QUERY, &[]).await {
            Ok(vec_rows) => {
                for row in &vec_rows {
                    let table_name: String = row.get("table_name");
                    let dimensions: i32 = row.get("dimensions");
                    let actual_dim = if dimensions > 4 { Some(dimensions - 4) } else { None };
                    if let Some(table) = tables.get_mut(&table_name) {
                        table.vector_columns.push(VectorColumnInfo {
                            column_name: row.get("column_name"),
                            dimensions: actual_dim,
                            index_type: None, distance_ops: None,
                        });
                    }
                }
                if let Ok(idx_rows) = client.query(cache::introspect::VECTOR_INDEXES_QUERY, &[]).await {
                    for row in &idx_rows {
                        let table_name: String = row.get("tablename");
                        let indexdef: String = row.get("indexdef");
                        if let Some(table) = tables.get_mut(&table_name) {
                            let idx_type = if indexdef.contains("ivfflat") { Some("ivfflat".into()) }
                                else if indexdef.contains("hnsw") { Some("hnsw".into()) }
                                else { None };
                            let dist_ops = if indexdef.contains("vector_cosine_ops") { Some("cosine".into()) }
                                else if indexdef.contains("vector_l2_ops") { Some("l2".into()) }
                                else if indexdef.contains("vector_ip_ops") { Some("inner_product".into()) }
                                else { None };
                            for vc in &mut table.vector_columns {
                                if indexdef.contains(&vc.column_name) {
                                    vc.index_type = idx_type.clone();
                                    vc.distance_ops = dist_ops.clone();
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => { log::debug!("NeonPostgresTool: pgvector not available ({})", e); }
        }

        Ok(SchemaCache { tables })
    }

    // -----------------------------------------------------------------------
    // Handler registration
    // -----------------------------------------------------------------------

    fn register_pg_schema(&self, registry: &mut FunctionRegistry) {
        let cache = self.schema_cache.clone();
        registry.register("pg_schema", move |_args: String| {
            let cache = cache.clone();
            async move {
                match cache.get() {
                    Some(sc) => sc.to_summary(),
                    None => "Error: schema not loaded yet".to_string(),
                }
            }
        });
    }

    fn register_pg_query(&self, registry: &mut FunctionRegistry) {
        let client = self.client.clone();
        let schema_cache = self.schema_cache.clone();
        let result_cache = self.result_cache.clone();
        let read_only = self.config.read_only;

        registry.register_data("pg_query", move |args: String| {
            let client = client.clone();
            let schema_cache = schema_cache.clone();
            let result_cache = result_cache.clone();

            async move {
                let db = match Self::require_client(&client) {
                    Ok(c) => c,
                    Err(e) => return ToolCallOutput::summary_only(e),
                };

                let parsed: Value = match serde_json::from_str(&args) {
                    Ok(v) => v,
                    Err(e) => return ToolCallOutput::summary_only(format!("Error: invalid arguments — {}", e)),
                };

                let query = match parsed.get("query").and_then(|q| q.as_str()) {
                    Some(q) => q,
                    None => return ToolCallOutput::summary_only("Error: 'query' field is required"),
                };

                let query_upper = query.trim().to_uppercase();
                if !query_upper.starts_with("SELECT") && !query_upper.starts_with("WITH") {
                    return ToolCallOutput::summary_only("Error: only SELECT and WITH (CTE) queries are allowed");
                }
                if query.contains(';') {
                    return ToolCallOutput::summary_only("Error: multiple statements not allowed");
                }

                let rows = if read_only {
                    let _ = db.execute("SET TRANSACTION READ ONLY", &[]).await;
                    match db.query(query, &[]).await {
                        Ok(r) => r,
                        Err(e) => return ToolCallOutput::summary_only(format!("Query error: {}", e)),
                    }
                } else {
                    match db.query(query, &[]).await {
                        Ok(r) => r,
                        Err(e) => return ToolCallOutput::summary_only(format!("Query error: {}", e)),
                    }
                };

                let json_rows = rows_to_json(&rows);
                let count = json_rows.len();

                let sc = schema_cache.get();
                let (table_name, pk_col) = match sc {
                    Some(sc) => detect_table_and_pk(query, sc),
                    None => (None, None),
                };

                let summary = if let (Some(table), Some(pk)) = (&table_name, &pk_col) {
                    let ids: Vec<Value> = json_rows.iter()
                        .filter_map(|row| row.get(pk.as_str()).cloned()).collect();
                    if !ids.is_empty() {
                        let rs_id = result_cache.lock().unwrap()
                            .store(table.clone(), pk.clone(), ids);
                        format!("result_set: {}, {} item(s) found from {}", rs_id, count, table)
                    } else {
                        format!("{} row(s) returned", count)
                    }
                } else {
                    format!("{} row(s) returned", count)
                };

                ToolCallOutput::with_data(summary, json!(json_rows))
            }
        });
    }

    fn register_pg_refine(&self, registry: &mut FunctionRegistry) {
        let client = self.client.clone();
        let schema_cache = self.schema_cache.clone();
        let result_cache = self.result_cache.clone();

        registry.register_data("pg_refine", move |args: String| {
            let client = client.clone();
            let schema_cache = schema_cache.clone();
            let result_cache = result_cache.clone();

            async move {
                let db = match Self::require_client(&client) {
                    Ok(c) => c,
                    Err(e) => return ToolCallOutput::summary_only(e),
                };
                let sc = match Self::require_schema(&schema_cache) {
                    Ok(s) => s,
                    Err(e) => return ToolCallOutput::summary_only(e),
                };

                let req: RefineRequest = match serde_json::from_str(&args) {
                    Ok(r) => r,
                    Err(e) => return ToolCallOutput::summary_only(
                        format!("Error: invalid refine request — {}", e)
                    ),
                };

                let (table, pk_col, cached_ids) = {
                    let cache = result_cache.lock().unwrap();
                    match cache.get(&req.result_set_id) {
                        Some(rs) => (rs.table.clone(), rs.pk_column.clone(), rs.ids.clone()),
                        None => return ToolCallOutput::summary_only(
                            format!("Error: result set '{}' not found. Use pg_query first.", req.result_set_id)
                        ),
                    }
                };

                if cached_ids.is_empty() {
                    return ToolCallOutput::summary_only("result_set is empty, nothing to refine");
                }

                let id_placeholders: Vec<String> = (1..=cached_ids.len())
                    .map(|i| format!("${}", i)).collect();
                let id_params: Vec<String> = cached_ids.iter()
                    .map(|v| match v {
                        Value::Number(n) => n.to_string(),
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    }).collect();

                let built = match build_filter(
                    &table, &req.filters, req.order_by.as_deref(),
                    req.limit, sc, cached_ids.len(),
                ) {
                    Ok(b) => b,
                    Err(e) => return ToolCallOutput::summary_only(format!("Filter error: {}", e)),
                };

                let mut sql = format!(
                    "SELECT * FROM \"{}\" WHERE \"{}\" IN ({})",
                    table, pk_col, id_placeholders.join(", ")
                );
                if !built.where_clause.is_empty() {
                    sql.push_str(&format!(" AND {}", built.where_clause));
                }
                if let Some(ob) = &built.order_by { sql.push_str(&format!(" {}", ob)); }
                if let Some(lim) = built.limit { sql.push_str(&format!(" LIMIT {}", lim)); }

                log::debug!("pg_refine SQL: {}", sql);

                let mut all_params: Vec<String> = id_params;
                all_params.extend(built.params);
                let param_refs: Vec<&(dyn ToSql + Sync)> = all_params.iter()
                    .map(|s| s as &(dyn ToSql + Sync)).collect();

                let rows = match db.query(&sql, &param_refs).await {
                    Ok(r) => r,
                    Err(e) => return ToolCallOutput::summary_only(format!("Query error: {}", e)),
                };

                let json_rows = rows_to_json(&rows);
                let count = json_rows.len();

                let new_ids: Vec<Value> = json_rows.iter()
                    .filter_map(|row| row.get(pk_col.as_str()).cloned()).collect();
                let rs_id = result_cache.lock().unwrap()
                    .store(table.clone(), pk_col, new_ids);

                ToolCallOutput::with_data(
                    format!("result_set: {}, {} item(s) found (refined from {})", rs_id, count, req.result_set_id),
                    json!(json_rows),
                )
            }
        });
    }

    fn register_pg_vector_search(&self, registry: &mut FunctionRegistry) {
        let client = self.client.clone();
        let schema_cache = self.schema_cache.clone();
        let result_cache = self.result_cache.clone();

        registry.register_data("pg_vector_search", move |args: String| {
            let client = client.clone();
            let schema_cache = schema_cache.clone();
            let result_cache = result_cache.clone();

            async move {
                let db = match Self::require_client(&client) {
                    Ok(c) => c,
                    Err(e) => return ToolCallOutput::summary_only(e),
                };

                let parsed: Value = match serde_json::from_str(&args) {
                    Ok(v) => v,
                    Err(e) => return ToolCallOutput::summary_only(format!("Error: invalid arguments — {}", e)),
                };

                let table = match parsed.get("table").and_then(|v| v.as_str()) {
                    Some(t) => t,
                    None => return ToolCallOutput::summary_only("Error: 'table' is required"),
                };
                let vector_column = match parsed.get("vector_column").and_then(|v| v.as_str()) {
                    Some(c) => c,
                    None => return ToolCallOutput::summary_only("Error: 'vector_column' is required"),
                };
                let query_vector = match parsed.get("query_vector").and_then(|v| v.as_array()) {
                    Some(arr) => arr.iter().filter_map(|v| v.as_f64()).collect::<Vec<f64>>(),
                    None => return ToolCallOutput::summary_only("Error: 'query_vector' must be an array of numbers"),
                };
                let top_k = parsed.get("top_k").and_then(|v| v.as_i64()).unwrap_or(10);
                let distance_metric = parsed.get("distance_metric").and_then(|v| v.as_str()).unwrap_or("cosine");
                let refine_from = parsed.get("refine_from").and_then(|v| v.as_str()).map(String::from);
                let select_columns = parsed.get("select_columns").and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect::<Vec<&str>>());

                if let Some(sc) = schema_cache.get() {
                    if let Some(vec_cols) = sc.vector_columns_for_table(table) {
                        if !vec_cols.iter().any(|vc| vc.column_name == vector_column) {
                            return ToolCallOutput::summary_only(
                                format!("Error: '{}' is not a vector column on '{}'", vector_column, table)
                            );
                        }
                    }
                }

                let (dist_op, dist_alias) = match distance_metric {
                    "cosine" => ("<=>", "cosine_distance"),
                    "l2" => ("<->", "l2_distance"),
                    "inner_product" => ("<#>", "neg_inner_product"),
                    other => return ToolCallOutput::summary_only(
                        format!("Error: unsupported distance metric '{}'", other)
                    ),
                };

                let select = match &select_columns {
                    Some(cols) => {
                        let mut parts: Vec<String> = cols.iter().map(|c| format!("\"{}\"", c)).collect();
                        parts.push(format!("\"{}\" {} $1::vector AS {}", vector_column, dist_op, dist_alias));
                        parts.join(", ")
                    }
                    None => format!("*, \"{}\" {} $1::vector AS {}", vector_column, dist_op, dist_alias),
                };

                let vec_str = format!("[{}]", query_vector.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","));

                let (where_clause, extra_params) = if let Some(rs_id) = &refine_from {
                    let cache = result_cache.lock().unwrap();
                    match cache.get(rs_id) {
                        Some(rs) => {
                            let pks: Vec<String> = rs.ids.iter().enumerate()
                                .map(|(i, _)| format!("${}", i + 2)).collect();
                            let params: Vec<String> = rs.ids.iter()
                                .map(|v| match v {
                                    Value::Number(n) => n.to_string(),
                                    Value::String(s) => s.clone(),
                                    other => other.to_string(),
                                }).collect();
                            (format!("WHERE \"{}\" IN ({})", rs.pk_column, pks.join(", ")), params)
                        }
                        None => return ToolCallOutput::summary_only(
                            format!("Error: result set '{}' not found", rs_id)
                        ),
                    }
                } else {
                    (String::new(), vec![])
                };

                let sql = format!(
                    "SELECT {} FROM \"{}\" {} ORDER BY \"{}\" {} $1::vector LIMIT {}",
                    select, table, where_clause, vector_column, dist_op, top_k
                );
                log::debug!("pg_vector_search SQL: {}", sql);

                let mut all_params: Vec<String> = vec![vec_str];
                all_params.extend(extra_params);
                let param_refs: Vec<&(dyn ToSql + Sync)> = all_params.iter()
                    .map(|s| s as &(dyn ToSql + Sync)).collect();

                let rows = match db.query(&sql, &param_refs).await {
                    Ok(r) => r,
                    Err(e) => return ToolCallOutput::summary_only(format!("Vector search error: {}", e)),
                };

                let json_rows = rows_to_json(&rows);
                let count = json_rows.len();

                let sc = schema_cache.get();
                let pk_col = sc.and_then(|s| s.pk_for_table(table)).unwrap_or("id").to_string();
                let ids: Vec<Value> = json_rows.iter()
                    .filter_map(|row| row.get(pk_col.as_str()).cloned()).collect();
                let rs_id = result_cache.lock().unwrap()
                    .store(table.to_string(), pk_col, ids);

                ToolCallOutput::with_data(
                    format!("result_set: {}, {} match(es) by {} similarity on {}.{}", rs_id, count, distance_metric, table, vector_column),
                    json!(json_rows),
                )
            }
        });
    }

    // -----------------------------------------------------------------------
    // Tool schemas
    // -----------------------------------------------------------------------

    fn build_tool_schemas() -> Vec<FunctionSchema> {
        vec![
            FunctionSchema::new(
                "pg_schema",
                "Inspect the database schema. Returns all tables, columns, types, \
                 primary keys, and vector columns. Call this first before writing queries.",
            ),
            FunctionSchema::new(
                "pg_query",
                "Execute a SQL SELECT query. Returns a result set ID and count. \
                 Full data is sent to the UI. Only SELECT/WITH queries allowed.",
            )
            .with_parameters(json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "SQL SELECT query" }
                },
                "required": ["query"],
                "additionalProperties": false
            }))
            .with_strict(true),
            FunctionSchema::new(
                "pg_refine",
                "Narrow a previous result set with structured filters. Do NOT write \
                 raw SQL — provide column/op/value filter conditions.",
            )
            .with_parameters(json!({
                "type": "object",
                "properties": {
                    "result_set_id": { "type": "string", "description": "e.g. 'rs_001'" },
                    "filters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": { "type": "string" },
                                "op": {
                                    "type": "string",
                                    "enum": ["=","!=","<",">","<=",">=","LIKE","ILIKE","IN","@>","IS NULL","IS NOT NULL"]
                                },
                                "value": { "description": "Omit for IS NULL / IS NOT NULL" }
                            },
                            "required": ["column", "op"],
                            "additionalProperties": false
                        }
                    },
                    "order_by": { "type": "string" },
                    "limit": { "type": "integer" }
                },
                "required": ["result_set_id", "filters"],
                "additionalProperties": false
            })),
            FunctionSchema::new(
                "pg_vector_search",
                "pgvector similarity search. Returns results ordered by distance. \
                 Optionally scope to a previous result set with refine_from.",
            )
            .with_parameters(json!({
                "type": "object",
                "properties": {
                    "table": { "type": "string" },
                    "vector_column": { "type": "string" },
                    "query_vector": { "type": "array", "items": { "type": "number" } },
                    "top_k": { "type": "integer" },
                    "distance_metric": { "type": "string", "enum": ["cosine","l2","inner_product"] },
                    "refine_from": { "type": "string" },
                    "select_columns": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["table", "vector_column", "query_vector", "top_k"],
                "additionalProperties": false
            })),
        ]
    }
}

// ---------------------------------------------------------------------------
// BuiltinTool implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl BuiltinTool for NeonPostgresTool {
    fn name(&self) -> &str {
        "neon_postgres"
    }

    fn is_cacheable(&self) -> bool {
        true
    }

    async fn on_start(&self, cancel: CancellationToken) -> Result<()> {
        log::info!("NeonPostgresTool: connecting to Neon...");

        self.cancel_token.set(cancel.clone()).map_err(|_| {
            PipecatError::pipeline("NeonPostgresTool: on_start called twice")
        })?;

        // Build TLS connector (native-tls delegates to OpenSSL on Linux,
        // Secure Transport on macOS — trusts system CA store, works with Neon)
        let connector = TlsConnector::builder()
            .build()
            .map_err(|e| PipecatError::pipeline(format!("Neon TLS build failed: {}", e)))?;
        let tls = MakeTlsConnector::new(connector);

        let (client, connection) = tokio_postgres::connect(
            &self.config.connection_url, tls,
        )
        .await
        .map_err(|e| PipecatError::pipeline(format!("Neon connect failed: {}", e)))?;

        tokio::spawn(async move {
            tokio::select! {
                _ = cancel.cancelled() => {
                    log::info!("NeonPostgresTool: connection task cancelled");
                }
                result = connection => {
                    if let Err(e) = result {
                        log::error!("NeonPostgresTool: connection dropped: {}", e);
                    }
                }
            }
        });

        if let Some(ms) = self.config.statement_timeout_ms {
            client
                .execute(&format!("SET statement_timeout = '{}'", ms), &[])
                .await
                .map_err(|e| PipecatError::pipeline(format!("SET timeout failed: {}", e)))?;
        }

        let schema = Self::introspect_schema(&client, &self.config.schemas)
            .await
            .map_err(|e| PipecatError::pipeline(format!("Schema introspection failed: {}", e)))?;

        log::info!(
            "NeonPostgresTool: cached {} table(s): [{}]",
            schema.tables.len(),
            schema.tables.keys().cloned().collect::<Vec<_>>().join(", ")
        );

        self.client.set(client).map_err(|_| {
            PipecatError::pipeline("NeonPostgresTool: client already set")
        })?;
        self.schema_cache.set(schema).map_err(|_| {
            PipecatError::pipeline("NeonPostgresTool: schema already cached")
        })?;

        self.set_state(ToolLifecycleState::Started);
        Ok(())
    }

    async fn on_stop(&self) -> Result<()> {
        let current = self.lifecycle_state();
        if current == ToolLifecycleState::Stopped || current == ToolLifecycleState::Cancelled {
            log::debug!("NeonPostgresTool: already stopped, skipping");
            return Ok(());
        }

        log::info!("NeonPostgresTool: stopping...");

        if let Some(token) = self.cancel_token.get() {
            token.cancel();
        }

        {
            let mut cache = self.result_cache.lock().unwrap();
            let count = cache.len();
            cache.clear();
            if count > 0 {
                log::debug!("NeonPostgresTool: cleared {} cached result set(s)", count);
            }
        }

        self.set_state(ToolLifecycleState::Stopped);
        log::info!("NeonPostgresTool: stopped");
        Ok(())
    }

    async fn on_cancel(&self) -> Result<()> {
        log::warn!("NeonPostgresTool: cancel requested");
        self.set_state(ToolLifecycleState::Cancelled);
        self.on_stop().await
    }

    fn tool_schemas(&self) -> Vec<FunctionSchema> {
        Self::build_tool_schemas()
    }

    fn register_all(&self, registry: &mut FunctionRegistry) {
        self.register_pg_schema(registry);
        self.register_pg_query(registry);
        self.register_pg_refine(registry);
        self.register_pg_vector_search(registry);
        log::info!("NeonPostgresTool: 4 handlers registered");
    }
}

// ---------------------------------------------------------------------------
// Drop — safety net
// ---------------------------------------------------------------------------

impl Drop for NeonPostgresTool {
    fn drop(&mut self) {
        let state = self.lifecycle_state();
        match state {
            ToolLifecycleState::Started => {
                log::warn!(
                    "NeonPostgresTool: dropped while still in '{}' state! \
                     on_stop() was never called. Resources may leak.",
                    state
                );
                if let Some(token) = self.cancel_token.get() {
                    token.cancel();
                }
            }
            ToolLifecycleState::Created => {
                log::debug!("NeonPostgresTool: dropped without ever starting (ok)");
            }
            _ => {
                log::debug!("NeonPostgresTool: dropped in '{}' state (clean)", state);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Debug
// ---------------------------------------------------------------------------

impl std::fmt::Debug for NeonPostgresTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeonPostgresTool")
            .field("state", &self.lifecycle_state())
            .field("initialized", &self.client.initialized())
            .field("tables", &self.schema_cache.get().map(|s| s.tables.len()))
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Row → JSON helpers
// ---------------------------------------------------------------------------

fn rows_to_json(rows: &[Row]) -> Vec<Value> {
    rows.iter()
        .map(|row| {
            let mut obj = serde_json::Map::new();
            for (i, col) in row.columns().iter().enumerate() {
                obj.insert(col.name().to_string(), column_to_json(row, i, col.type_()));
            }
            Value::Object(obj)
        })
        .collect()
}

fn column_to_json(row: &Row, idx: usize, pg_type: &tokio_postgres::types::Type) -> Value {
    use tokio_postgres::types::Type;
    match *pg_type {
        Type::BOOL => row.try_get::<_, Option<bool>>(idx).ok().flatten().map(Value::Bool).unwrap_or(Value::Null),
        Type::INT2 => row.try_get::<_, Option<i16>>(idx).ok().flatten().map(|v| json!(v)).unwrap_or(Value::Null),
        Type::INT4 => row.try_get::<_, Option<i32>>(idx).ok().flatten().map(|v| json!(v)).unwrap_or(Value::Null),
        Type::INT8 => row.try_get::<_, Option<i64>>(idx).ok().flatten().map(|v| json!(v)).unwrap_or(Value::Null),
        Type::FLOAT4 => row.try_get::<_, Option<f32>>(idx).ok().flatten().map(|v| json!(v)).unwrap_or(Value::Null),
        Type::FLOAT8 => row.try_get::<_, Option<f64>>(idx).ok().flatten().map(|v| json!(v)).unwrap_or(Value::Null),
        Type::JSON | Type::JSONB => row.try_get::<_, Option<Value>>(idx).ok().flatten().unwrap_or(Value::Null),
        Type::TEXT | Type::VARCHAR | Type::BPCHAR | Type::NAME =>
            row.try_get::<_, Option<String>>(idx).ok().flatten().map(Value::String).unwrap_or(Value::Null),
        Type::TIMESTAMP => row.try_get::<_, Option<chrono::NaiveDateTime>>(idx).ok().flatten()
            .map(|v| Value::String(v.to_string())).unwrap_or(Value::Null),
        Type::TIMESTAMPTZ => row.try_get::<_, Option<chrono::DateTime<chrono::Utc>>>(idx).ok().flatten()
            .map(|v| Value::String(v.to_rfc3339())).unwrap_or(Value::Null),
        Type::DATE => row.try_get::<_, Option<chrono::NaiveDate>>(idx).ok().flatten()
            .map(|v| Value::String(v.to_string())).unwrap_or(Value::Null),
        Type::UUID => row.try_get::<_, Option<uuid::Uuid>>(idx).ok().flatten()
            .map(|v| Value::String(v.to_string())).unwrap_or(Value::Null),
        Type::TEXT_ARRAY | Type::VARCHAR_ARRAY =>
            row.try_get::<_, Option<Vec<String>>>(idx).ok().flatten().map(|v| json!(v)).unwrap_or(Value::Null),
        Type::INT4_ARRAY =>
            row.try_get::<_, Option<Vec<i32>>>(idx).ok().flatten().map(|v| json!(v)).unwrap_or(Value::Null),
        _ => row.try_get::<_, Option<String>>(idx).ok().flatten().map(Value::String).unwrap_or(Value::Null),
    }
}

fn detect_table_and_pk(query: &str, schema_cache: &SchemaCache) -> (Option<String>, Option<String>) {
    let upper = query.to_uppercase();
    let from_pos = match upper.find(" FROM ") {
        Some(p) => p + 6,
        None => return (None, None),
    };
    let rest = query[from_pos..].trim();
    let table_name = rest.split_whitespace().next()
        .map(|s| s.trim_matches('"').trim_matches('\'')).unwrap_or("");
    if table_name.is_empty() || table_name.contains('(') {
        return (None, None);
    }
    let table = table_name.to_string();
    let pk = schema_cache.pk_for_table(&table).map(|s| s.to_string());
    (Some(table), pk)
}
