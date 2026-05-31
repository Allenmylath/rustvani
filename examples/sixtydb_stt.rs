//! 60db STT integration test — stream a WAV file and print transcripts.
//!
//! Run:
//!   cargo run --example sixtydb_stt -- tests/test.wav
//!
//! Requires SIXTYDB_API_KEY in .env or environment.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rustvani::{
    frames::{
        AudioRawData, Frame, FrameDirection, FrameInner, StartFrameData, SystemFrame,
    },
    services::{SixtyDbSttConfig, SixtyDbSttHandler},
};
use tokio::sync::mpsc;

fn read_wav_pcm(path: &str) -> std::result::Result<(Vec<u8>, u32), Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| format!("Failed to open {}: {}", path, e))?;
    let spec = reader.spec();
    println!("[wav] {}Hz  {}ch  {}bit", spec.sample_rate, spec.channels, spec.bits_per_sample);

    if spec.channels != 1 {
        eprintln!("[error] WAV must be mono");
        std::process::exit(1);
    }

    let samples: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
    let duration_sec = samples.len() as f32 / spec.sample_rate as f32;
    println!("[wav] {:.2}s  ({} samples)", duration_sec, samples.len());

    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    Ok((bytes, spec.sample_rate))
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();

    // Logs to file, terminal stays clean
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("sixtydb_stt.log")
        .expect("Failed to open sixtydb_stt.log");
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .target(env_logger::Target::Pipe(Box::new(std::io::LineWriter::new(log_file))))
        .init();

    let api_key = std::env::var("SIXTYDB_API_KEY")
        .expect("Set SIXTYDB_API_KEY in .env or environment");

    let wav_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/test.wav".to_string());

    let (pcm_bytes, sample_rate) = read_wav_pcm(&wav_path)?;

    println!("\n=== 60db STT Integration Test ===");
    println!("API key:  {}...", &api_key[..api_key.len().min(8)]);
    println!();

    // Default config: linear PCM @ 16kHz (browser format)
    let stt = SixtyDbSttHandler::new(SixtyDbSttConfig {
        api_key,
        languages: vec!["en".to_string()],
        continuous_mode: true,
        ..Default::default()
    })
    .into_processor();

    // Capture frames
    let (frame_tx, mut frame_rx) = mpsc::channel::<(String, Frame)>(256);
    let tx_clone = frame_tx.clone();
    stt.on_after_push_frame(move |f| {
        let _ = tx_clone.try_send((f.name().to_string(), f.clone()));
    });
    let tx_err = frame_tx.clone();
    stt.on_error(move |e| {
        let _ = tx_err.try_send((
            "ErrorFrame".to_string(),
            Frame::error(e.error.clone(), e.fatal, e.processor_name.clone()),
        ));
    });

    let transcriptions = Arc::new(AtomicUsize::new(0));
    let errors = Arc::new(AtomicUsize::new(0));
    let t1 = transcriptions.clone();
    let e1 = errors.clone();

    let printer = tokio::spawn(async move {
        while let Some((name, frame)) = frame_rx.recv().await {
            match &frame.inner {
                FrameInner::Data(rustvani::frames::DataFrame::Transcription(data)) => {
                    t1.fetch_add(1, Ordering::Relaxed);
                    let status = if data.finalized { "✓ FINAL" } else { "⟳ INTERIM" };
                    println!("[recv] {} | {} | \"{}\"", status, name, data.text);
                }
                FrameInner::System(SystemFrame::UserStartedSpeaking { .. }) => {
                    println!("[recv] 🎤 speech_started");
                }
                FrameInner::System(SystemFrame::Error(data)) => {
                    e1.fetch_add(1, Ordering::Relaxed);
                    println!("[recv] ⚠ error: {} (fatal={})", data.error, data.fatal);
                }
                _ => {}
            }
        }
    });

    println!("[main] Sending StartFrame...");
    stt.process_frame(Frame::start(StartFrameData::default()), FrameDirection::Downstream).await?;

    println!("[main] Waiting for WebSocket ready (~2s)...");
    tokio::time::sleep(Duration::from_millis(2000)).await;

    // Stream audio chunks (~60ms throttle)
    let chunk_ms = 60;
    let chunk_bytes = ((sample_rate * 2) / 1000 * chunk_ms) as usize;
    let chunks: Vec<&[u8]> = pcm_bytes.chunks(chunk_bytes).filter(|c| !c.is_empty()).collect();
    println!("[main] Streaming {} chunks...\n", chunks.len());

    let start = Instant::now();
    for (i, chunk) in chunks.iter().enumerate() {
        let data = AudioRawData::new(chunk.to_vec(), sample_rate, 1);
        stt.process_frame(Frame::input_audio_raw(data), FrameDirection::Downstream).await?;
        tokio::time::sleep(Duration::from_millis(chunk_ms as u64)).await;
        if i % 16 == 0 && i > 0 {
            print!("\r[main] sent {}/{} chunks  ({:.1}s)", i, chunks.len(), start.elapsed().as_secs_f32());
        }
    }
    println!();

    println!("\n[main] Sending EndFrame...");
    stt.process_frame(Frame::end(), FrameDirection::Downstream).await?;

    println!("[main] Waiting up to 10s for final transcripts...");
    drop(frame_tx);
    let _ = tokio::time::timeout(Duration::from_secs(10), printer).await;

    let tx_count = transcriptions.load(Ordering::Relaxed);
    let err_count = errors.load(Ordering::Relaxed);
    println!("\n=== Summary ===");
    println!("Transcriptions received: {}", tx_count);
    println!("Errors received:         {}", err_count);

    if tx_count == 0 && err_count == 0 {
        println!("\n[warn] No transcripts or errors. Check sixtydb_stt.log for details.");
    }

    Ok(())
}
