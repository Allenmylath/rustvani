# Billing

**File:** `src/billing/`  
**Feature:** always available (no feature flag required)

Per-session usage tracking. Captures LLM token counts, TTS character counts, STT audio duration, and a full conversation **transcript** (user + assistant turns with timestamps). All data is linked by `session_id` and stored to PostgreSQL or logged as JSON.

---

## What gets recorded

| Data | Source | Table |
|---|---|---|
| LLM token usage (per call) | `LlmUsage` event | `billing_events` |
| TTS character count (per synthesis) | `TtsUsage` event | `billing_events` |
| STT audio duration (per turn) | `SttUsage` event | `billing_events` |
| Session start / end / duration | `SessionStart` / `SessionEnd` | `billing_sessions` |
| Aggregated session totals | `finalize_session()` | `billing_sessions` |
| Conversation transcript (every turn) | `Transcript` event | `session_transcripts` |

---

## Quick Start

```rust
use rustvani::{SessionBilling, BillingCollector};
use rustvani::billing::storage::postgres::PostgresBillingStorage;
use std::sync::Arc;
use uuid::Uuid;

// 1. Run migrations once at startup
PostgresBillingStorage::run_migrations(&pg_client).await?;

// 2. Create a collector per session
let session_id = Uuid::new_v4();
let storage = Arc::new(PostgresBillingStorage::with_arc(Arc::new(pg_client)));
let (billing, _handle) = SessionBilling::new(session_id, storage, 256);

// 3. Pass it into PipelineParams
let params = PipelineParams {
    billing_collector: Some(billing.clone()),
    billing_metadata: [
        ("user_id".into(), "u_42".into()),
        ("phone_number".into(), "+91-98765-43210".into()),
    ].into_iter().collect(),
    ..Default::default()
};
```

The pipeline hooks `SessionStart` and `SessionEnd` automatically via `PipelineParams`. STT/LLM/TTS services record their own events when a `billing_collector` is present.

---

## Transcript Recording

Transcript entries are recorded via `LLMUserAggregator` (user turns) and `LLMAssistantAggregator` (assistant turns). To enable them, use the `with_billing()` constructors instead of `new()`:

```rust
use std::sync::{Arc, Mutex};
use uuid::Uuid;

// Shared turn-id cells — link each transcript entry to its audio segment
let active_user_turn_id: Arc<Mutex<Option<Uuid>>> = Arc::new(Mutex::new(None));
let active_bot_turn_id:  Arc<Mutex<Option<Uuid>>> = Arc::new(Mutex::new(None));

let user_agg = LLMUserAggregator::with_billing(
    context.clone(),
    billing.clone(),
    active_user_turn_id.clone(),
);

let assistant_agg = LLMAssistantAggregator::with_billing(
    context.clone(),
    billing.clone(),
    active_bot_turn_id.clone(),
);
```

Both aggregators record a `session_transcripts` row on every completed turn. If the bot is **interrupted mid-response**, the partial text is recorded with `interrupted = true` — preserving what the user actually heard.

### TranscriptEntry fields

| Field | Type | Description |
|---|---|---|
| `turn_id` | `Uuid` | Links this entry to an audio segment (same UUID) |
| `session_id` | `Uuid` | Session this turn belongs to |
| `role` | `TranscriptRole` | `User` or `Assistant` |
| `text` | `String` | Full spoken text for the turn |
| `language` | `Option<String>` | ISO language code (user turns, from STT) |
| `interrupted` | `bool` | `true` if bot was cut short by user speech |
| `occurred_at` | `DateTime<Utc>` | When the turn completed |

### Querying the transcript

```sql
-- Full conversation transcript in order
SELECT role, text, language, interrupted, occurred_at
FROM session_transcripts
WHERE session_id = 'your-session-uuid'
ORDER BY occurred_at;

-- Only interrupted bot turns (what users heard before speaking over the bot)
SELECT text, occurred_at
FROM session_transcripts
WHERE session_id = 'your-session-uuid'
  AND role = 'assistant'
  AND interrupted = TRUE;
```

---

## Database Schema

### `billing_sessions`

```sql
session_id        UUID        PRIMARY KEY
started_at        TIMESTAMPTZ
ended_at          TIMESTAMPTZ
duration_secs     FLOAT8
finish_reason     TEXT         -- "end" | "stop" | "cancel"
llm_input_tokens  INTEGER      DEFAULT 0
llm_output_tokens INTEGER      DEFAULT 0
llm_calls         INTEGER      DEFAULT 0
tts_chars         INTEGER      DEFAULT 0
tts_calls         INTEGER      DEFAULT 0
stt_audio_ms      FLOAT8       DEFAULT 0
stt_calls         INTEGER      DEFAULT 0
metadata          JSONB        DEFAULT '{}'
```

### `billing_events`

Raw per-event log: one row per LLM call, TTS synthesis, or STT turn.

```sql
id                BIGSERIAL    PRIMARY KEY
session_id        UUID         FK → billing_sessions
event_type        TEXT         -- "llm" | "tts" | "stt"
provider          TEXT
model             TEXT         -- LLM only
input_tokens      INTEGER      -- LLM only
output_tokens     INTEGER      -- LLM only
estimated         BOOLEAN      -- LLM only (true = estimated, not reported)
char_count        INTEGER      -- TTS only
voice             TEXT         -- TTS only
audio_duration_ms FLOAT8       -- STT only
occurred_at       TIMESTAMPTZ
raw_json          JSONB
```

### `session_transcripts`

```sql
id          BIGSERIAL    PRIMARY KEY
session_id  UUID         FK → billing_sessions
turn_id     UUID         -- links to session_audio_segments.turn_id
role        TEXT         -- "user" | "assistant"
text        TEXT
language    TEXT         -- nullable
interrupted BOOLEAN      DEFAULT FALSE
occurred_at TIMESTAMPTZ
```

---

## Storage Backends

### PostgreSQL (production)

```rust
// Cargo.toml: features = ["db-postgres"]
use rustvani::billing::storage::postgres::PostgresBillingStorage;

let storage = Arc::new(PostgresBillingStorage::with_arc(pg_client.clone()));
PostgresBillingStorage::run_migrations(&pg_client).await?;
```

Requires `DATABASE_URL` or a manually constructed `tokio_postgres::Client`.

### Log (development / fallback)

```rust
use rustvani::LogBillingStorage;

let storage = Arc::new(LogBillingStorage);
// Writes JSON lines at INFO level — zero database required
```

`LogBillingStorage` is the default when no `DATABASE_URL` is set.

---

## BillingEvent enum

```rust
pub enum BillingEvent {
    SessionStart   { session_id, started_at, metadata },
    SessionEnd     { session_id, ended_at, finish_reason },
    LlmUsage       { session_id, provider, model, input_tokens, output_tokens, estimated, occurred_at },
    TtsUsage       { session_id, provider, voice, char_count, occurred_at },
    SttUsage       { session_id, provider, audio_duration_ms, occurred_at },
    Transcript(TranscriptEntry),
}
```

---

## Metadata

Arbitrary key-value pairs attached at session start. Useful for filtering in your analytics dashboard.

```rust
PipelineParams {
    billing_metadata: [
        ("user_id".into(), "u_42".into()),
        ("tenant_id".into(), "acme-corp".into()),
        ("phone_number".into(), "+91-98765-43210".into()),
        ("region".into(), "ap-south-1".into()),
    ].into_iter().collect(),
    ..
}
```

Querying by metadata:

```sql
-- All sessions for a specific user
SELECT session_id, duration_secs, llm_input_tokens
FROM billing_sessions
WHERE metadata->>'user_id' = 'u_42'
ORDER BY started_at DESC;
```

---

## Mid-session snapshot

Read accumulated totals before the session ends:

```rust
let snapshot = billing.snapshot();
println!("tokens so far: {}", snapshot.llm_input_tokens + snapshot.llm_output_tokens);
```

---

## BillingCollector trait

Implement this to plug in a custom sink:

```rust
use rustvani::billing::collector::BillingCollector;
use rustvani::billing::events::{BillingEvent, TranscriptEntry};

struct MyBilling;

impl BillingCollector for MyBilling {
    fn record(&self, event: BillingEvent) {
        // enqueue or process synchronously — must never block
    }
    fn session_id(&self) -> uuid::Uuid {
        self.session_id
    }
    // record_transcript() has a default impl that calls record(BillingEvent::Transcript(_))
}
```

`record()` is always sync and must not block — the audio path calls it on every STT/LLM/TTS completion.

---

## Pipeline position

Billing is invisible in the pipeline topology — it is injected through `PipelineParams` and the aggregator constructors, not as a processor stage.

```
transport.input()
  → stt          (records SttUsage)
  → user_agg     (records Transcript[user] if with_billing())
  → llm          (records LlmUsage)
  → assistant_agg (records Transcript[assistant] if with_billing())
  → tts          (records TtsUsage)
  → audio_capture (optional, see audio-capture.md)
  → transport.output()
```

`PipelineTask` records `SessionStart` on pipeline start and `SessionEnd` on finish.
