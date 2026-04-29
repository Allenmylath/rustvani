//! Schema cache and result set cache.
//!
//! - `SchemaCache`: introspects Neon DB at init, holds table/column/vector
//!   metadata in memory. Never hits the DB again (unless `refresh()` is called).
//!
//! - `ResultSetCache`: holds recent query result IDs so the LLM can refine
//!   previous results without re-querying the full dataset.

use std::collections::HashMap;
use std::time::Instant;

use serde_json::Value;

// ---------------------------------------------------------------------------
// Schema cache types
// ---------------------------------------------------------------------------

/// A single column's metadata.
#[derive(Debug, Clone)]
pub struct CachedColumn {
    pub name: String,
    pub data_type: String,
    pub is_nullable: bool,
    pub column_default: Option<String>,
}

/// pgvector column info.
#[derive(Debug, Clone)]
pub struct VectorColumnInfo {
    pub column_name: String,
    /// Dimension of the vector (e.g. 1536 for OpenAI ada-002).
    pub dimensions: Option<i32>,
    /// Index type if present (e.g. "ivfflat", "hnsw").
    pub index_type: Option<String>,
    /// Distance operator the index uses (e.g. "vector_cosine_ops").
    pub distance_ops: Option<String>,
}

/// A single table's metadata.
#[derive(Debug, Clone)]
pub struct CachedTable {
    pub schema: String,
    pub name: String,
    pub pk_column: Option<String>,
    pub columns: Vec<CachedColumn>,
    pub vector_columns: Vec<VectorColumnInfo>,
}

// ---------------------------------------------------------------------------
// SchemaCache
// ---------------------------------------------------------------------------

/// In-memory snapshot of the database schema.
///
/// Populated once at `NeonPostgresTool::new()`. The LLM calls `pg_schema`
/// to read from this — no DB roundtrip.
#[derive(Debug, Clone)]
pub struct SchemaCache {
    /// `table_name → CachedTable`
    pub tables: HashMap<String, CachedTable>,
}

impl SchemaCache {
    /// Create an empty cache (used in tests).
    pub fn empty() -> Self {
        Self {
            tables: HashMap::new(),
        }
    }

    /// Look up columns for a table. Returns `None` if table not in cache
    /// (we allow uncached tables so the tool works even if schema changed).
    pub fn columns_for_table(&self, table: &str) -> Option<&Vec<CachedColumn>> {
        self.tables.get(table).map(|t| &t.columns)
    }

    /// Get primary key column for a table.
    pub fn pk_for_table(&self, table: &str) -> Option<&str> {
        self.tables
            .get(table)
            .and_then(|t| t.pk_column.as_deref())
    }

    /// Get vector columns for a table.
    pub fn vector_columns_for_table(&self, table: &str) -> Option<&Vec<VectorColumnInfo>> {
        self.tables.get(table).map(|t| &t.vector_columns)
    }

    /// Format the schema as a compact string for the LLM.
    ///
    /// Output looks like:
    /// ```text
    /// TABLE: menu_items (pk: id)
    ///   id          integer       NOT NULL
    ///   name        text          NOT NULL
    ///   price       numeric(10,2) NOT NULL
    ///   allergens   jsonb
    ///   embedding   vector(1536)  [hnsw, cosine]
    ///
    /// TABLE: orders (pk: order_id)
    ///   ...
    /// ```
    pub fn to_summary(&self) -> String {
        if self.tables.is_empty() {
            return "No tables found in database.".to_string();
        }

        let mut tables: Vec<&CachedTable> = self.tables.values().collect();
        tables.sort_by(|a, b| a.name.cmp(&b.name));

        let mut out = String::new();
        for (i, table) in tables.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }

            let pk_info = match &table.pk_column {
                Some(pk) => format!(" (pk: {})", pk),
                None => String::new(),
            };
            out.push_str(&format!(
                "TABLE: {}.{}{}\n",
                table.schema, table.name, pk_info
            ));

            for col in &table.columns {
                let nullable = if col.is_nullable { "" } else { "NOT NULL" };
                let default = match &col.column_default {
                    Some(d) => format!(" DEFAULT {}", d),
                    None => String::new(),
                };

                // Check if this is a vector column
                let vector_info = table
                    .vector_columns
                    .iter()
                    .find(|vc| vc.column_name == col.name)
                    .map(|vc| {
                        let dim = vc
                            .dimensions
                            .map(|d| format!("({})", d))
                            .unwrap_or_default();
                        let idx = match (&vc.index_type, &vc.distance_ops) {
                            (Some(it), Some(ops)) => format!(" [{}, {}]", it, ops),
                            (Some(it), None) => format!(" [{}]", it),
                            _ => String::new(),
                        };
                        format!("vector{}{}", dim, idx)
                    });

                let dtype = vector_info.unwrap_or_else(|| col.data_type.clone());

                out.push_str(&format!(
                    "  {:<24} {:<18} {}{}\n",
                    col.name, dtype, nullable, default
                ));
            }
        }

        out
    }
}

/// SQL queries to introspect the schema. Run once at init.
pub mod introspect {
    /// Fetch all tables and their columns from information_schema.
    pub const COLUMNS_QUERY: &str = r#"
        SELECT
            c.table_schema,
            c.table_name,
            c.column_name,
            c.data_type,
            c.is_nullable,
            c.column_default,
            c.udt_name
        FROM information_schema.columns c
        JOIN information_schema.tables t
            ON c.table_schema = t.table_schema
            AND c.table_name = t.table_name
        WHERE t.table_type = 'BASE TABLE'
            AND c.table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY c.table_schema, c.table_name, c.ordinal_position
    "#;

    /// Fetch primary key columns.
    pub const PK_QUERY: &str = r#"
        SELECT
            tc.table_schema,
            tc.table_name,
            kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        WHERE tc.constraint_type = 'PRIMARY KEY'
            AND tc.table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY tc.table_schema, tc.table_name
    "#;

    /// Fetch pgvector columns and their index info.
    ///
    /// This query checks for columns with `vector` UDT and joins with
    /// pg_indexes to find index types.
    pub const VECTOR_COLUMNS_QUERY: &str = r#"
        SELECT
            c.table_schema,
            c.table_name,
            c.column_name,
            a.atttypmod as dimensions
        FROM information_schema.columns c
        JOIN pg_catalog.pg_attribute a
            ON a.attname = c.column_name
        JOIN pg_catalog.pg_class cl
            ON cl.relname = c.table_name
            AND a.attrelid = cl.oid
        JOIN pg_catalog.pg_namespace n
            ON n.nspname = c.table_schema
            AND cl.relnamespace = n.oid
        WHERE c.udt_name = 'vector'
            AND c.table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY c.table_schema, c.table_name
    "#;

    /// Fetch vector index details.
    pub const VECTOR_INDEXES_QUERY: &str = r#"
        SELECT
            schemaname,
            tablename,
            indexname,
            indexdef
        FROM pg_indexes
        WHERE indexdef LIKE '%vector%'
            AND schemaname NOT IN ('pg_catalog', 'information_schema')
    "#;
}

// ---------------------------------------------------------------------------
// Result set cache
// ---------------------------------------------------------------------------

/// A cached result set from a previous query.
#[derive(Debug, Clone)]
pub struct CachedResultSet {
    /// Unique identifier (e.g. "rs_001").
    pub id: String,
    /// Table the query was against.
    pub table: String,
    /// Primary key column name.
    pub pk_column: String,
    /// Cached primary key values.
    pub ids: Vec<Value>,
    /// When this was created (for eviction).
    pub created_at: Instant,
}

/// In-memory cache of recent query result sets.
///
/// Keyed by result set ID. Capped at `max_sets` entries (LRU eviction).
pub struct ResultSetCache {
    sets: HashMap<String, CachedResultSet>,
    counter: u64,
    max_sets: usize,
}

impl ResultSetCache {
    /// Create a new cache with the given max capacity.
    pub fn new(max_sets: usize) -> Self {
        Self {
            sets: HashMap::new(),
            counter: 0,
            max_sets,
        }
    }

    /// Store a new result set and return its ID.
    pub fn store(
        &mut self,
        table: String,
        pk_column: String,
        ids: Vec<Value>,
    ) -> String {
        // Evict oldest if at capacity
        if self.sets.len() >= self.max_sets {
            self.evict_oldest();
        }

        self.counter += 1;
        let id = format!("rs_{:04}", self.counter);

        self.sets.insert(
            id.clone(),
            CachedResultSet {
                id: id.clone(),
                table,
                pk_column,
                ids,
                created_at: Instant::now(),
            },
        );

        id
    }

    /// Look up a result set by ID.
    pub fn get(&self, id: &str) -> Option<&CachedResultSet> {
        self.sets.get(id)
    }

    /// Evict the oldest entry.
    fn evict_oldest(&mut self) {
        if let Some(oldest_key) = self
            .sets
            .iter()
            .min_by_key(|(_, v)| v.created_at)
            .map(|(k, _)| k.clone())
        {
            self.sets.remove(&oldest_key);
        }
    }

    /// Number of cached result sets.
    pub fn len(&self) -> usize {
        self.sets.len()
    }

    /// True if no result sets are cached.
    pub fn is_empty(&self) -> bool {
        self.sets.is_empty()
    }

    /// Clear all cached result sets.
    pub fn clear(&mut self) {
        self.sets.clear();
        self.counter = 0;
    }
}

impl std::fmt::Debug for ResultSetCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResultSetCache")
            .field("count", &self.sets.len())
            .field("max", &self.max_sets)
            .field("counter", &self.counter)
            .finish()
    }
}
