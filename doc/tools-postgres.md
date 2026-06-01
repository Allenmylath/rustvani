# Neon Postgres Tool

**Files:** `src/tools/postgres/`  
**Feature:** `db-postgres` (not enabled by default)

Built-in LLM tool for Neon Postgres with schema caching, parameterized queries, pgvector similarity search, and structured filters. The LLM never writes raw SQL — all queries are parameterized.

## Setup

```toml
[dependencies]
rustvani = { version = "0.2", features = ["db-postgres"] }
```

```bash
DATABASE_URL=postgres://user:pass@host/db
```

## Usage

```rust
use rustvani::tools::postgres::NeonPostgresTool;
use rustvani::services::llm::openai::OpenAILLMHandler;

let pg = Arc::new(NeonPostgresTool::from_env());

let mut llm = OpenAILLMHandler::new(config);
llm.add_tool(pg);
let processor = llm.into_processor();
```

## Registered Functions

When attached to an `OpenAILLMHandler`, the tool registers these functions into the shared `FunctionRegistry`:

| Function | Purpose |
|---|---|
| `pg_schema` | Introspect tables, columns, and types |
| `pg_query` | Execute a parameterized SELECT |
| `pg_refine` | Refine a query plan based on schema |
| `pg_vector_search` | pgvector similarity search with structured filters |

## Lifecycle

- `on_start` (cacheable): Connects to Postgres and caches schema
- `on_stop`: Returns connections, flushes caches
- `on_cancel`: Cancels in-flight queries, then `on_stop`

## Cargo Feature

```toml
[dependencies]
rustvani = { version = "0.2", features = ["db-postgres"] }
```
