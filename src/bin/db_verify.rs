//! Quick read-only check that billing/transcript data reached Postgres.
//!
//! Connects with DATABASE_URL, prints the latest billing_sessions rows plus
//! per-session event counts and transcript length.
//!
//!   cargo run --bin db_verify

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    let url = std::env::var("DATABASE_URL")?;

    let connector = native_tls::TlsConnector::builder().build()?;
    let tls = postgres_native_tls::MakeTlsConnector::new(connector);
    let (client, connection) = tokio_postgres::connect(&url, tls).await?;
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("connection dropped: {e}");
        }
    });

    let total: i64 = client
        .query_one("SELECT count(*) FROM billing_sessions", &[])
        .await?
        .get(0);
    println!("billing_sessions total rows: {total}\n");

    let rows = client
        .query(
            "SELECT session_id, started_at, ended_at, duration_secs, finish_reason,
                    llm_input_tokens, llm_output_tokens, llm_calls,
                    tts_chars, tts_calls, stt_audio_ms, stt_calls,
                    jsonb_array_length(transcript_json) AS transcript_len,
                    status, last_checkpoint_at
             FROM billing_sessions
             ORDER BY created_at DESC
             LIMIT 5",
            &[],
        )
        .await?;

    for r in &rows {
        let sid: uuid::Uuid = r.get("session_id");
        let dur: Option<f64> = r.get("duration_secs");
        let reason: Option<String> = r.get("finish_reason");
        let llm_in: i32 = r.get("llm_input_tokens");
        let llm_out: i32 = r.get("llm_output_tokens");
        let llm_calls: i32 = r.get("llm_calls");
        let tts_chars: i32 = r.get("tts_chars");
        let tts_calls: i32 = r.get("tts_calls");
        let stt_ms: f64 = r.get("stt_audio_ms");
        let stt_calls: i32 = r.get("stt_calls");
        let tlen: Option<i32> = r.get("transcript_len");
        let status: String = r.get("status");
        let last_ckpt: Option<chrono::DateTime<chrono::Utc>> = r.get("last_checkpoint_at");

        // event rows for this session, plus how many carry a stable event_id
        let ev: i64 = client
            .query_one(
                "SELECT count(*) FROM billing_events WHERE session_id = $1",
                &[&sid],
            )
            .await?
            .get(0);
        let ev_with_id: i64 = client
            .query_one(
                "SELECT count(*) FROM billing_events WHERE session_id = $1 AND event_id IS NOT NULL",
                &[&sid],
            )
            .await?
            .get(0);

        println!("session {sid}  [status={status}]");
        println!("  duration={:?}s  finish={:?}  last_checkpoint={:?}", dur, reason, last_ckpt);
        println!("  LLM in={llm_in} out={llm_out} calls={llm_calls}");
        println!("  TTS chars={tts_chars} calls={tts_calls}  STT ms={stt_ms:.0} calls={stt_calls}");
        println!("  billing_events rows={ev} (with event_id={ev_with_id})  transcript entries={:?}", tlen);
        println!();
    }

    Ok(())
}
