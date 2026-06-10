# Audio Capture

**File:** `src/audio_capture/`  
**Feature:** always available (no feature flag required)  
**Depends on:** `hound` (WAV encoding, already in `Cargo.toml`)

Per-turn audio recording for both user and bot speech. Each speaking turn is saved as a WAV file and indexed in PostgreSQL. Turns are linked to their transcript entries via a shared `turn_id` UUID so you can play back exactly what was said — including bot turns that were cut short by user interruption.

---

## How it works

`AudioCaptureProcessor` sits in the pipeline between TTS and the output transport. It accumulates raw PCM bytes while audio flows through and flushes a completed segment when the turn ends (or when interrupted). A background drain task encodes the PCM to WAV and writes to storage — never blocking the audio path.

```
transport.input()
  → stt → user_agg → llm → assistant_agg → tts
  → AudioCaptureProcessor   ← inserted here
  → transport.output()
```

### What triggers a flush

| Event | Speaker | `interrupted` |
|---|---|---|
| `VADUserStoppedSpeaking` | user | `false` |
| `BotStoppedSpeaking` | bot | `false` |
| `Interruption` (user speaks over bot) | bot | `true` |
| `End` / `Stop` / `Cancel` (session closes) | both | `true` |

---

## Quick Start

```rust
use std::sync::{Arc, Mutex};
use uuid::Uuid;
use rustvani::audio_capture::{
    AudioCaptureProcessor, SessionAudioCapture, LocalAudioStorage,
};

// Storage: write WAV files to ./recordings/
let audio_storage = Arc::new(LocalAudioStorage::new("./recordings"));

// Shared turn-id cells — same Arcs passed to LLMUserAggregator::with_billing()
// and LLMAssistantAggregator::with_billing()
let active_user_turn_id: Arc<Mutex<Option<Uuid>>> = Arc::new(Mutex::new(None));
let active_bot_turn_id:  Arc<Mutex<Option<Uuid>>> = Arc::new(Mutex::new(None));

let (audio_collector, _handle) = SessionAudioCapture::new(
    session_id,
    audio_storage,
    64,   // channel capacity — number of segments that can be buffered
);

let audio_proc = AudioCaptureProcessor::new(
    audio_collector,
    active_user_turn_id.clone(),
    active_bot_turn_id.clone(),
);
```

Pass the **same** `active_user_turn_id` / `active_bot_turn_id` cells to the aggregators so transcript and audio share a `turn_id`:

```rust
let user_agg = LLMUserAggregator::with_billing(
    context.clone(), billing.clone(), active_user_turn_id.clone());

let assistant_agg = LLMAssistantAggregator::with_billing(
    context.clone(), billing.clone(), active_bot_turn_id.clone());
```

Full pipeline assembly:

```rust
let processors = vec![
    transport.input(),
    stt,
    user_agg,
    llm,
    assistant_agg,
    tts,
    audio_proc,           // after TTS, before output transport
    transport.output(),
];
```

---

## How client-side interruptions are captured

When the user speaks over the bot, the browser sends `{"type": "client_vad_start"}` over WebSocket. This travels through the pipeline as a `VADUserStartedSpeaking` frame, which triggers `SystemFrame::Interruption`. `AudioCaptureProcessor` receives the interruption and immediately flushes the in-progress bot audio segment with `interrupted = true`.

The saved audio includes roughly 50–200ms of audio the client may not have played (the round-trip latency between the client stopping playback and the server processing the interruption). This tail is harmless — `interrupted = true` tells you where to expect the cutoff.

---

## Linking audio to transcript

`AudioCaptureProcessor` generates a `turn_id` UUID when each speaking turn starts and writes it to the shared `Arc<Mutex<Option<Uuid>>>` cell. The aggregators read the same cell when recording the transcript entry. Both rows end up with the same `turn_id`.

```sql
-- Play back a turn: find the audio for a transcript entry
SELECT t.role, t.text, t.interrupted, a.audio_url, a.duration_ms
FROM session_transcripts   t
JOIN session_audio_segments a ON a.turn_id = t.turn_id
WHERE t.session_id = 'your-session-uuid'
ORDER BY t.occurred_at;
```

---

## Database Schema

### `session_audio_segments`

```sql
id           BIGSERIAL   PRIMARY KEY
session_id   UUID        NOT NULL
turn_id      UUID                    -- links to session_transcripts.turn_id
speaker      TEXT        NOT NULL    -- "user" | "bot"
audio_url    TEXT        NOT NULL    -- file path or S3 URI
format       TEXT        DEFAULT 'wav'
sample_rate  INTEGER     NOT NULL
num_channels SMALLINT    NOT NULL
duration_ms  FLOAT8      NOT NULL
byte_size    BIGINT      NOT NULL
interrupted  BOOLEAN     DEFAULT FALSE
started_at   TIMESTAMPTZ NOT NULL
ended_at     TIMESTAMPTZ NOT NULL
```

Run migrations at startup:

```rust
use rustvani::audio_capture::storage::postgres::PostgresAudioMetaStorage;

PostgresAudioMetaStorage::run_migrations(&pg_client).await?;
```

---

## Storage Backends

### Local filesystem (development)

```rust
use rustvani::audio_capture::LocalAudioStorage;

let storage = Arc::new(LocalAudioStorage::new("/var/recordings"));
// Files saved as: /var/recordings/{session_id}/{segment_id}.wav
// Metadata logged as JSON at INFO level
```

### PostgreSQL metadata + local files (production)

Combine `LocalAudioStorage` (binary) with `PostgresAudioMetaStorage` (metadata rows) using a wrapper:

```rust
pub struct LocalWithPostgres {
    files: LocalAudioStorage,
    db:    PostgresAudioMetaStorage,
}

#[async_trait]
impl AudioStorage for LocalWithPostgres {
    async fn store_segment(&self, session_id, segment_id, speaker, data) -> Result<String> {
        self.files.store_segment(session_id, segment_id, speaker, data).await
    }
    async fn save_metadata(&self, session_id, meta) -> Result<()> {
        self.db.save_metadata(session_id, meta).await
    }
}
```

### Custom storage (S3, GCS, etc.)

Implement `AudioStorage`:

```rust
use rustvani::audio_capture::AudioStorage;

#[async_trait]
impl AudioStorage for MyS3Storage {
    async fn store_segment(&self, session_id: Uuid, segment_id: Uuid, speaker: &str, data: &[u8]) -> Result<String> {
        let key = format!("{session_id}/{segment_id}.wav");
        self.s3_client.put_object(&self.bucket, &key, data).await?;
        Ok(format!("s3://{}/{}", self.bucket, key))
    }

    async fn save_metadata(&self, session_id: Uuid, meta: &AudioSegmentMeta) -> Result<()> {
        self.pg.save_metadata(session_id, meta).await
    }
}
```

---

## Audio format

All files are encoded as **WAV** (16-bit signed PCM, mono or stereo depending on the input stream). Encoding happens in a `spawn_blocking` task so it never occupies the async runtime.

The PCM that arrives from the pipeline is already in the correct format:
- User audio: same sample rate as your STT config (typically 16 kHz mono)
- Bot audio: same sample rate as your TTS output

---

## Disabling audio capture

Simply don't add `AudioCaptureProcessor` to the pipeline. Everything else (billing, transcript) works independently.

To disable at compile time for a specific service that doesn't need it, just omit the processor from the pipeline vector — no feature flag required.

---

## AudioCaptureCollector trait

Implement this to use a custom channel or in-memory store (useful for tests):

```rust
use rustvani::audio_capture::{AudioCaptureCollector, PendingAudioSegment};

struct InMemoryAudioCapture {
    segments: Arc<Mutex<Vec<PendingAudioSegment>>>,
    session_id: Uuid,
}

impl AudioCaptureCollector for InMemoryAudioCapture {
    fn record_segment(&self, segment: PendingAudioSegment) {
        self.segments.lock().unwrap().push(segment);
    }
    fn session_id(&self) -> Uuid { self.session_id }
}
```

---

## PendingAudioSegment fields

| Field | Type | Description |
|---|---|---|
| `segment_id` | `Uuid` | Unique ID for this audio file |
| `turn_id` | `Option<Uuid>` | Matches `TranscriptEntry.turn_id` |
| `speaker` | `&'static str` | `"user"` or `"bot"` |
| `pcm` | `Vec<u8>` | Raw PCM bytes (16-bit LE signed) |
| `sample_rate` | `u32` | Samples per second |
| `num_channels` | `u16` | 1 = mono, 2 = stereo |
| `started_at` | `DateTime<Utc>` | When the turn started |
| `interrupted` | `bool` | `true` if cut short by interruption |

---

## Example: full session query

```sql
-- Complete call log with playback URLs in order
SELECT
    t.role,
    t.text,
    t.interrupted      AS transcript_interrupted,
    a.audio_url,
    a.duration_ms,
    a.interrupted      AS audio_interrupted,
    t.occurred_at
FROM session_transcripts    t
LEFT JOIN session_audio_segments a
    ON a.turn_id = t.turn_id
WHERE t.session_id = $1
ORDER BY t.occurred_at;
```
