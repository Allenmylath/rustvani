//! Full pipeline integration test — billing + transcription + recording.
//!
//! Topology:
//!   ChannelTransport.input()
//!     → SarvamStt (with_billing)
//!     → LLMUserAggregator (with_billing, shares user turn-id)
//!     → OpenAILLM (with_billing)
//!     → LLMAssistantAggregator (with_billing, shares bot turn-id)
//!     → DeepgramTts (with_billing)
//!     → AudioCaptureProcessor (records user + bot turns)
//!     → ChannelTransport.output()
//!
//! Feeds the first 10s of tests/test.wav through the realistic transport path
//! (VAD runs inside the transport), waits for the bot response, closes the
//! pipeline, then prints where the three artifacts landed:
//!   - billing summary   (captured in-process + logged + finalize_session)
//!   - transcript        (recordings/{session}/transcript.json)
//!   - recording         (recordings/{session}/recording.wav)
//!
//! Run:
//!   cargo run --bin full_pipeline_test
//! (reads SARVAM_API_KEY / OPENAI_API_KEY / DEEPGRAM_API_KEY from .env)

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc;
use uuid::Uuid;

use rustvani::{
    audio_capture::{AudioCaptureProcessor, LocalAudioStorage, SessionAudioCapture},
    billing::{
        events::{SessionSummary, TranscriptEntry},
        BillingEvent, BillingStorage, SessionBilling,
    },
    context::shared_context,
    error::Result as PcResult,
    pipeline::{PipelineParams, PipelineTask},
    processors::{
        llm_assistant_aggregator::LLMAssistantAggregator,
        llm_user_aggregator::LLMUserAggregator,
    },
    services::{
        DeepgramTtsConfig, DeepgramTtsHandler, OpenAILLMConfig, OpenAILLMHandler,
        SarvamSttConfig, SarvamSttHandler,
    },
    system_clock, Frame, FrameDirection, SileroVadNative, VadParams,
    transport::{ChannelMessage, ChannelTransport, TransportParams},
};

const CHUNK_BYTES: usize = 512 * 2; // 512 samples * 2 bytes (16-bit mono)
const WAV_PATH: &str = "tests/test.wav";
const MAX_SECS: f32 = 10.0;
const RECORDINGS_DIR: &str = "recordings";

// ---------------------------------------------------------------------------
// CapturingBillingStorage — logs like LogBillingStorage AND keeps the final
// summary + transcripts so we can print them after the drain task finalises.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Captured {
    summary: Option<SessionSummary>,
    transcripts: Vec<TranscriptEntry>,
    event_count: usize,
}

struct CapturingBillingStorage {
    captured: Mutex<Captured>,
}

impl CapturingBillingStorage {
    fn new() -> Self {
        Self { captured: Mutex::new(Captured::default()) }
    }
}

#[async_trait]
impl BillingStorage for CapturingBillingStorage {
    async fn record_event(&self, event: &BillingEvent) -> PcResult<()> {
        self.captured.lock().unwrap().event_count += 1;
        log::info!("billing_event: {}", serde_json::to_string(event).unwrap_or_default());
        Ok(())
    }

    async fn finalize_session(
        &self,
        summary: &SessionSummary,
        transcripts: &[TranscriptEntry],
    ) -> PcResult<()> {
        let mut c = self.captured.lock().unwrap();
        c.summary = Some(summary.clone());
        c.transcripts = transcripts.to_vec();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// WAV reader — first `max_secs` seconds only
// ---------------------------------------------------------------------------

fn read_wav_pcm(path: &str, max_secs: f32) -> std::result::Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| format!("Failed to open {}: {}", path, e))?;
    let spec = reader.spec();
    println!("[wav] {}Hz  {}ch  {}bit", spec.sample_rate, spec.channels, spec.bits_per_sample);
    assert_eq!(spec.sample_rate, 16_000, "Need 16kHz");
    assert_eq!(spec.channels, 1, "Need mono");

    let samples: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
    let max_samples = (spec.sample_rate as f32 * max_secs) as usize;
    let samples = &samples[..samples.len().min(max_samples)];

    println!("[wav] using {:.1}s  ({} samples)", samples.len() as f32 / 16_000.0, samples.len());
    Ok(samples.iter().flat_map(|s| s.to_le_bytes()).collect())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let sarvam_key   = std::env::var("SARVAM_API_KEY").map_err(|_| "SARVAM_API_KEY not set")?;
    let openai_key   = std::env::var("OPENAI_API_KEY").map_err(|_| "OPENAI_API_KEY not set")?;
    let deepgram_key = std::env::var("DEEPGRAM_API_KEY").map_err(|_| "DEEPGRAM_API_KEY not set")?;

    let wav_path = std::env::args().nth(1).unwrap_or_else(|| WAV_PATH.to_string());

    println!("=== rustvani full pipeline test (billing + transcript + recording) ===\n");

    let session_id = Uuid::new_v4();
    let session_dir: PathBuf = PathBuf::from(RECORDINGS_DIR).join(session_id.to_string());
    println!("[session] id = {session_id}");
    println!("[session] dir = {}\n", session_dir.display());

    let pcm_bytes = read_wav_pcm(&wav_path, MAX_SECS)?;

    // -----------------------------------------------------------------------
    // Billing + audio capture sessions
    // -----------------------------------------------------------------------
    let billing_storage = Arc::new(CapturingBillingStorage::new());
    let (billing, billing_handle) = SessionBilling::new_with_dir(
        session_id,
        billing_storage.clone(),
        256,
        Some(session_dir.clone()),
    );

    let audio_storage = Arc::new(LocalAudioStorage::new(RECORDINGS_DIR));
    let (audio_collector, audio_handle) =
        SessionAudioCapture::new(session_id, audio_storage, 64);

    // Shared turn-id cells link transcript entries to audio segments.
    let active_user_turn_id: Arc<Mutex<Option<Uuid>>> = Arc::new(Mutex::new(None));
    let active_bot_turn_id:  Arc<Mutex<Option<Uuid>>> = Arc::new(Mutex::new(None));

    // -----------------------------------------------------------------------
    // Transport + processors
    // -----------------------------------------------------------------------
    let (incoming_tx, incoming_rx) = mpsc::channel::<ChannelMessage>(100);
    let (outgoing_tx, mut outgoing_rx) = mpsc::channel::<ChannelMessage>(100);

    let vad_analyzer = Arc::new(SileroVadNative::new(16_000)?);

    let transport = ChannelTransport::new(
        "ChannelTransport",
        TransportParams {
            audio_in_enabled:         true,
            audio_in_sample_rate:     Some(16_000),
            audio_in_channels:        1,
            audio_in_passthrough:     true,
            audio_in_stream_on_start: true,
            audio_out_enabled:        true,
            audio_out_sample_rate:    Some(24_000), // Deepgram default
            audio_out_channels:       1,
            audio_out_10ms_chunks:    4,
            vad_analyzer:             Some(vad_analyzer),
            vad_params:               VadParams {
                confidence: 0.4,
                min_volume: 0.1,
                ..VadParams::default()
            },
            ..TransportParams::default()
        },
        incoming_rx,
    );

    let context = shared_context(Some(
        "You are a helpful voice assistant. Reply in ONE short sentence — no lists, \
         no preamble. Keep it brief and conversational.".into(),
    ));

    let stt = SarvamSttHandler::new(SarvamSttConfig {
        api_key:  sarvam_key,
        model:    "saaras:v3".to_string(),
        language: Some("en-IN".to_string()),
        mode:     Some("transcribe".to_string()),
        ..SarvamSttConfig::default()
    })
    .with_billing(billing.clone())
    .into_processor();

    let user_agg = LLMUserAggregator::with_billing(
        context.clone(),
        billing.clone(),
        active_user_turn_id.clone(),
    );

    let llm = OpenAILLMHandler::new(OpenAILLMConfig {
        api_key: openai_key,
        model:   "gpt-4o-mini".into(),
        ..OpenAILLMConfig::default()
    })
    .with_billing(billing.clone())
    .into_processor();

    let assistant_agg = LLMAssistantAggregator::with_billing(
        context.clone(),
        billing.clone(),
        active_bot_turn_id.clone(),
    );

    let tts = DeepgramTtsHandler::new(DeepgramTtsConfig {
        api_key: deepgram_key,
        ..DeepgramTtsConfig::default()
    })?
    .with_billing(billing.clone())
    .into_processor();

    let audio_proc = AudioCaptureProcessor::new(
        audio_collector.clone(),
        active_user_turn_id.clone(),
        active_bot_turn_id.clone(),
    );

    // -----------------------------------------------------------------------
    // Pipeline — billing_collector wired so SessionStart/SessionEnd auto-emit
    // -----------------------------------------------------------------------
    let task = PipelineTask::new(
        vec![
            transport.input(),
            stt,
            user_agg,
            llm,
            assistant_agg,
            tts,
            audio_proc,
            transport.output(),
        ],
        PipelineParams {
            // Interruptions ON (realistic): when the user resumes over the bot,
            // the bot yields, so the recording is clean turn-taking with only a
            // tiny barge-in tail instead of the user's 2nd utterance overlapping
            // a long bot reply. Billing stays populated through interruptions via
            // the LLM drop-guard estimate and the TTS Cleared/End billing fixes.
            allow_interruptions: true,
            enable_usage_metrics: true,
            billing_collector: Some(billing.clone()),
            ..PipelineParams::default()
        },
    );

    // Two clones of the push sender: one for the transport (to inject VAD /
    // audio frames), one kept in main so we can send the closing End frame.
    let push_tx_transport = task.push_sender();
    let push_tx_main = task.push_sender();

    let pipeline_handle = tokio::spawn(async move {
        if let Err(e) = task.run(system_clock(), None).await {
            log::error!("Pipeline error: {e}");
        }
        println!("[pipeline] stopped");
    });

    let transport_handle = tokio::spawn(async move {
        transport.run(push_tx_transport, outgoing_tx).await;
        println!("[transport] stopped");
    });

    // Drain the transport output CONCURRENTLY from the start. The bot reply can
    // be many seconds of audio; if we only drained after feeding, the 100-slot
    // output channel would fill, the transport would block sending it, stop
    // reading input, and deadlock the feeder. (Counters via shared atomics.)
    let tts_audio_bytes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let text_msgs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let drain_handle = {
        let bytes = tts_audio_bytes.clone();
        let msgs = text_msgs.clone();
        tokio::spawn(async move {
            use std::sync::atomic::Ordering;
            while let Some(msg) = outgoing_rx.recv().await {
                match msg {
                    ChannelMessage::Audio(b) => { bytes.fetch_add(b.len(), Ordering::Relaxed); }
                    ChannelMessage::Text(_)  => { msgs.fetch_add(1, Ordering::Relaxed); }
                    _ => {}
                }
            }
        })
    };

    tokio::time::sleep(Duration::from_millis(200)).await;
    println!("[main] pipeline running — multi-turn conversation\n");

    // -----------------------------------------------------------------------
    // Multi-turn conversation, reusing the same two audio strips.
    //
    // The 10s clip splits into two natural utterances:
    //   strip A ≈ "...how do I get to Dublin?"          (first ~3.5s)
    //   strip B ≈ "...I wouldn't start from here, Sonny" (the rest)
    //
    // Conversation:
    //   Turn 1  user: A          → wait → bot replies          (clean turn)
    //   Turn 2  user: B          → wait → bot replies          (clean turn)
    //   Turn 3  user: A, bot starts replying, user barges in
    //           with B while the bot is still speaking         (BOTH speak)
    // -----------------------------------------------------------------------
    let split = (3.5 * 16_000.0) as usize * 2; // 3.5s of 16-bit mono, byte offset
    let split = split.min(pcm_bytes.len());
    let strip_a = pcm_bytes[..split].to_vec();
    let strip_b = pcm_bytes[split..].to_vec();
    println!(
        "[main] strip A = {:.1}s, strip B = {:.1}s\n",
        strip_a.len() as f32 / 32_000.0,
        strip_b.len() as f32 / 32_000.0,
    );

    // --- Turn 1: user speaks (A), bot replies ---
    println!("[main] turn 1 — user speaks (strip A)");
    feed_strip(&incoming_tx, &strip_a).await;
    tokio::time::sleep(Duration::from_secs(6)).await; // let the bot reply, then pause

    // --- Turn 2: user speaks (B), bot replies ---
    println!("[main] turn 2 — user speaks (strip B)");
    feed_strip(&incoming_tx, &strip_b).await;
    tokio::time::sleep(Duration::from_secs(6)).await;

    // --- Turn 3: user speaks (A), then barges in (B) while bot is replying ---
    println!("[main] turn 3 — user speaks (strip A), then both speak (barge-in with strip B)");
    feed_strip(&incoming_tx, &strip_a).await;
    tokio::time::sleep(Duration::from_millis(1500)).await; // bot starts replying...
    feed_strip(&incoming_tx, &strip_b).await;              // ...user talks over it
    tokio::time::sleep(Duration::from_secs(7)).await;

    println!("[main] conversation done — closing pipeline\n");

    // Close the pipeline: stop feeding (drop incoming) and push an End frame.
    println!("[main] sending End frame to close pipeline\n");
    drop(incoming_tx);
    let _ = push_tx_main.send((Frame::end(), FrameDirection::Downstream)).await;

    // Wait for the output drain to finish (transport closes outgoing on shutdown).
    let drain = tokio::time::timeout(Duration::from_secs(10), drain_handle).await;
    if drain.is_err() {
        println!("[main] drain timed out (transport still open) — continuing");
    }

    // -----------------------------------------------------------------------
    // Wait for pipeline + drain tasks, then drop collectors so drain tasks end
    // -----------------------------------------------------------------------
    let _ = tokio::time::timeout(Duration::from_secs(5), pipeline_handle).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), transport_handle).await;

    // Drop every remaining BillingCollector / AudioCaptureCollector clone so
    // the drain tasks observe channel-close and finalise.
    drop(billing);
    drop(audio_collector);

    let _ = tokio::time::timeout(Duration::from_secs(10), billing_handle).await;
    let _ = tokio::time::timeout(Duration::from_secs(10), audio_handle).await;

    // -----------------------------------------------------------------------
    // Report
    // -----------------------------------------------------------------------
    use std::sync::atomic::Ordering;
    print_report(
        &billing_storage,
        &session_dir,
        tts_audio_bytes.load(Ordering::Relaxed),
        text_msgs.load(Ordering::Relaxed),
    )
    .await;

    Ok(())
}

/// Feed one audio strip into the transport at ~1x realtime (32 ms per
/// 512-sample chunk), so the recording's wall-clock timeline matches the audio.
async fn feed_strip(incoming_tx: &mpsc::Sender<ChannelMessage>, pcm: &[u8]) {
    for chunk in pcm.chunks(CHUNK_BYTES).filter(|c| c.len() == CHUNK_BYTES) {
        if incoming_tx.send(ChannelMessage::Audio(chunk.to_vec())).await.is_err() {
            println!("[main] incoming channel closed — stopping feed");
            return;
        }
        tokio::time::sleep(Duration::from_millis(32)).await;
    }
}

async fn print_report(
    storage: &CapturingBillingStorage,
    session_dir: &PathBuf,
    tts_audio_bytes: usize,
    text_msgs: usize,
) {
    println!("\n========================================================");
    println!("                     RESULTS");
    println!("========================================================\n");

    // ---- BILLING ----
    println!("---- BILLING ----");
    let captured = storage.captured.lock().unwrap();
    println!("billing events recorded: {}", captured.event_count);
    match &captured.summary {
        Some(s) => {
            println!("session_id     : {}", s.session_id);
            println!("duration_secs  : {:?}", s.duration_secs);
            println!("finish_reason  : {:?}", s.finish_reason);
            println!("LLM   in={} out={} calls={}", s.llm_input_tokens, s.llm_output_tokens, s.llm_calls);
            println!("TTS   chars={} calls={}", s.tts_chars, s.tts_calls);
            println!("STT   audio_ms={:.0} calls={}", s.stt_audio_ms, s.stt_calls);

            // Persist the summary next to transcript.json / recording.wav so
            // billing is stored on disk, not just printed.
            let billing_json = session_dir.join("billing.json");
            match serde_json::to_string_pretty(s) {
                Ok(j) => match tokio::fs::write(&billing_json, &j).await {
                    Ok(()) => println!("billing.json   : {} ({} bytes on disk)", billing_json.display(), j.len()),
                    Err(e) => println!("!! billing.json write failed at {}: {e}", billing_json.display()),
                },
                Err(e) => println!("!! billing serialize failed: {e}"),
            }
        }
        None => println!("!! no finalised billing summary (drain task did not finalise)"),
    }

    // ---- TRANSCRIPT ----
    println!("\n---- TRANSCRIPT ----");
    println!("transcript entries (in-memory): {}", captured.transcripts.len());
    for t in &captured.transcripts {
        println!("  [{}] {}{}", t.role.as_str(), t.text,
            if t.interrupted { "  (interrupted)" } else { "" });
    }
    let transcript_json = session_dir.join("transcript.json");
    match tokio::fs::read_to_string(&transcript_json).await {
        Ok(s) => println!("transcript.json : {} ({} bytes on disk)", transcript_json.display(), s.len()),
        Err(e) => println!("!! transcript.json missing at {}: {e}", transcript_json.display()),
    }

    // ---- RECORDING ----
    println!("\n---- RECORDING ----");
    println!("TTS audio drained from transport: {tts_audio_bytes} bytes, text msgs: {text_msgs}");
    let recording = session_dir.join("recording.wav");
    match tokio::fs::metadata(&recording).await {
        Ok(m) => println!("recording.wav   : {} ({} bytes on disk)", recording.display(), m.len()),
        Err(e) => println!("!! recording.wav missing at {}: {e}", recording.display()),
    }

    // ---- VERDICT ----
    println!("\n---- VERDICT ----");
    let billing_ok = captured.summary.is_some();
    let transcript_ok = !captured.transcripts.is_empty();
    let recording_ok = tokio::fs::metadata(session_dir.join("recording.wav")).await.is_ok();
    println!("billing    : {}", if billing_ok { "OK" } else { "FAIL" });
    println!("transcript : {}", if transcript_ok { "OK" } else { "FAIL" });
    println!("recording  : {}", if recording_ok { "OK" } else { "FAIL" });
    println!("\n========================================================");
}
