//! 60db STT integration test — stream a WAV file and print transcripts.
//!
//! Run:
//!   SIXTYDB_API_KEY=sk_live_your_key \
//!     cargo run --example sixtydb_stt -- tests/test.wav
//!
//! What happens:
//!   1. Reads a 16kHz mono WAV file
//!   2. Connects to wss://api.60db.ai/ws/stt
//!   3. Streams audio chunks (~60ms) to 60db
//!   4. Prints every server message (speech_started, transcription, errors, etc.)
//!   5. Sends EndFrame and prints billing summary

use std::time::{Duration, Instant};

use rustvani::{
    frames::{
        AudioRawData, Frame, FrameDirection, FrameInner, FrameProcessor, StartFrameData,
        SystemFrame,
    },
    pipeline::{PipelineParams, PipelineTask},
    services::{SixtyDbAudioEnhancement, SixtyDbSttConfig, SixtyDbSttHandler},
    system_clock,
};
use tokio::sync::mpsc;

const CHUNK_MS: u64 = 60;
const SAMPLE_RATE: u32 = 16_000;
const BYTES_PER_MS: u32 = (SAMPLE_RATE * 2) / 1000; // 16-bit mono = 32 bytes/ms
const CHUNK_BYTES: usize = (CHUNK_MS as u32 * BYTES_PER_MS) as usize; // ~1920 bytes

/// Read a WAV file into raw PCM bytes. Must be 16kHz mono 16-bit.
fn read_wav_pcm(path: &str) -> std::result::Result<(Vec<u8>, u32), Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| format!("Failed to open {}: {}", path, e))?;
    let spec = reader.spec();
    println!("[wav] {}Hz  {}ch  {}bit", spec.sample_rate, spec.channels, spec.bits_per_sample);

    if spec.sample_rate != SAMPLE_RATE {
        println!("[warn] WAV is {}Hz — 60db config is {}Hz (auto-resampling enabled)",
            spec.sample_rate, SAMPLE_RATE);
    }
    if spec.channels != 1 {
        eprintln!("[error] WAV must be mono");
        std::process::exit(1);
    }

    let samples: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
    let duration_sec = samples.len() as f32 / spec.sample_rate as f32;
    println!("[wav] {:.2}s  ({} samples)  → {} chunks @ {}ms",
        duration_sec, samples.len(),
        samples.len() * 2 / CHUNK_BYTES,
        CHUNK_MS
    );

    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    Ok((bytes, spec.sample_rate))
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();

    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
    .init();

    let api_key = std::env::var("SIXTYDB_API_KEY")
        .expect("Set SIXTYDB_API_KEY env var");

    let wav_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/test.wav".to_string());

    let (pcm_bytes, wav_rate) = read_wav_pcm(&wav_path)?;

    println!("\n=== 60db STT Integration Test ===\n");
    println!("Endpoint: wss://api.60db.ai/ws/stt");
    println!("API key:  {}...", &api_key[..api_key.len().min(8)]);
    println!();

    // -----------------------------------------------------------------------
    // 1. Create the 60db STT handler
    // -----------------------------------------------------------------------
    let stt = SixtyDbSttHandler::new(SixtyDbSttConfig {
        api_key,
        sample_rate: SAMPLE_RATE,
        languages: vec!["en".to_string()],
        continuous_mode: true,
        utterance_end_ms: 500,
        interim_results_frequency: Some(300),
        audio_enhancement: SixtyDbAudioEnhancement::Adaptive,
        noise_reduction: true,
        ..Default::default()
    })
    .into_processor();

    // -----------------------------------------------------------------------
    // 2. Create a capture processor to observe everything the STT emits
    // -----------------------------------------------------------------------
    let (frame_tx, mut frame_rx) = mpsc::channel::<(String, Frame)>(100);

    let capture = FrameProcessor::new("Capture", Box::new(CaptureHandler { tx: frame_tx }), false);

    // Link: stt → capture
    stt.link(&capture);

    // -----------------------------------------------------------------------
    // 3. Start the pipeline
    // -----------------------------------------------------------------------
    let task = PipelineTask::new(
        vec![stt.clone(), capture],
        PipelineParams {
            allow_interruptions: true,
            ..PipelineParams::default()
        },
    );

    let push_tx = task.push_sender();

    let _pipeline = tokio::spawn(async move {
        if let Err(e) = task.run(system_clock(), None).await {
            eprintln!("[pipeline] error: {}", e);
        }
    });

    // Wait for pipeline to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // -----------------------------------------------------------------------
    // 4. Send StartFrame to trigger WebSocket connection
    // -----------------------------------------------------------------------
    push_tx
        .send((
            Frame::start(StartFrameData::default()),
            FrameDirection::Downstream,
        ))
        .await?;

    println!("[main] StartFrame sent — waiting for WebSocket handshake...\n");

    // Give the handler time to connect and complete the handshake
    tokio::time::sleep(Duration::from_millis(1500)).await;

    // -----------------------------------------------------------------------
    // 5. Stream audio chunks
    // -----------------------------------------------------------------------
    let chunks: Vec<&[u8]> = pcm_bytes
        .chunks(CHUNK_BYTES)
        .filter(|c| !c.is_empty())
        .collect();

    println!("[main] Streaming {} chunks (~{}ms each)...\n", chunks.len(), CHUNK_MS);

    let stream_start = Instant::now();
    for (i, chunk) in chunks.iter().enumerate() {
        let data = AudioRawData::new(chunk.to_vec(), wav_rate, 1);
        let frame = Frame::input_audio_raw(data);

        push_tx
            .send((frame, FrameDirection::Downstream))
            .await
            .map_err(|e| format!("send failed: {}", e))?;

        // Throttle to roughly real-time
        tokio::time::sleep(Duration::from_millis(CHUNK_MS)).await;

        // Print progress every second
        if i % 16 == 0 && i > 0 {
            let elapsed = stream_start.elapsed().as_secs_f32();
            print!("\r[main] sent {}/{} chunks  ({:.1}s elapsed)", i, chunks.len(), elapsed);
        }
    }
    println!();

    // -----------------------------------------------------------------------
    // 6. Graceful shutdown
    // -----------------------------------------------------------------------
    println!("\n[main] Audio complete — sending EndFrame...\n");
    push_tx
        .send((Frame::end(), FrameDirection::Downstream))
        .await
        .map_err(|e| format!("send failed: {}", e))?;

    // Drain remaining frames for up to 10 seconds
    let drain_deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    let mut received_transcriptions = 0usize;
    let mut received_errors = 0usize;

    while tokio::time::Instant::now() < drain_deadline {
        match tokio::time::timeout(Duration::from_millis(500), frame_rx.recv()).await {
            Ok(Some((name, frame))) => {
                match &frame.inner {
                    FrameInner::Data(rustvani::frames::DataFrame::Transcription(data)) => {
                        received_transcriptions += 1;
                        let status = if data.finalized { "✓ FINAL" } else { "⟳ FIRST EMIT" };
                        println!(
                            "[recv] {} | {} | lang={:?} | text=\"{}\"",
                            status, name, data.language, data.text
                        );
                    }
                    FrameInner::System(SystemFrame::UserStartedSpeaking { .. }) => {
                        println!("[recv] 🎤 speech_started (barge-in)");
                    }
                    FrameInner::System(SystemFrame::UserStoppedSpeaking { .. }) => {
                        println!("[recv] 🔇 user_stopped_speaking");
                    }
                    FrameInner::System(SystemFrame::Error(data)) => {
                        received_errors += 1;
                        println!("[recv] ⚠ error: {} (fatal={})", data.error, data.fatal);
                    }
                    other => {
                        println!("[recv] {} | {:?}", name, other);
                    }
                }
            }
            Ok(None) => break,
            Err(_) => {
                // Timeout — check if we got anything useful
                if received_transcriptions > 0 {
                    println!("[main] No more frames after 500ms — done.");
                    break;
                }
            }
        }
    }

    println!("\n=== Summary ===");
    println!("Transcriptions received: {}", received_transcriptions);
    println!("Errors received:         {}", received_errors);

    if received_transcriptions == 0 && received_errors == 0 {
        println!("\n[warn] No transcripts or errors received.");
        println!("       Common causes:");
        println!("       1. SIXTYDB_API_KEY is invalid or has no credits");
        println!("       2. WebSocket handshake failed (check logs above)");
        println!("       3. Audio format mismatch (WAV must be 16-bit PCM mono)");
        println!("       4. Server returned an error that wasn't captured");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Capture handler — prints and forwards every frame
// ---------------------------------------------------------------------------

use async_trait::async_trait;
use rustvani::frames::{FrameHandler};
use rustvani::error::Result;

struct CaptureHandler {
    tx: mpsc::Sender<(String, Frame)>,
}

#[async_trait]
impl FrameHandler for CaptureHandler {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        let name = frame.name().to_string();
        let _ = self.tx.send((name, frame.clone())).await;
        processor.push_frame(frame, direction).await
    }
}
