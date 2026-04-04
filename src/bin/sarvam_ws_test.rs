//! Standalone Sarvam WebSocket connection test.
//!
//! Reads a WAV file, sends it to Sarvam streaming STT, prints responses.
//! No pipeline, no VAD, no Docker needed.
//!
//! Run:
//!   SARVAM_API_KEY=your-key cargo run --bin sarvam_ws_test
//!   SARVAM_API_KEY=your-key cargo run --bin sarvam_ws_test -- /path/to/audio.wav

use std::time::Duration;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use futures::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::http::Request;
use tokio_tungstenite::tungstenite::Message;

const SARVAM_BASE_WSS: &str = "wss://api.sarvam.ai";
const STT_PATH: &str        = "/speech-to-text/ws";

fn build_url() -> String {
    format!(
        "{}{}?model=saaras%3Av3&sample_rate=16000&input_audio_codec=wav&flush_signal=true&language-code=en-IN&mode=transcribe",
        SARVAM_BASE_WSS, STT_PATH
    )
}

/// Build audio message per AsyncAPI spec:
/// {"audio": {"data": <base64>, "sample_rate": "16000", "encoding": "audio/wav"}}
fn audio_message(pcm: &[u8]) -> String {
    serde_json::json!({
        "audio": {
            "data":        BASE64.encode(pcm),
            "sample_rate": "16000",
            "encoding":    "audio/wav",
        }
    })
    .to_string()
}

/// Flush signal per AsyncAPI spec: {"type": "flush"}
fn flush_message() -> String {
    serde_json::json!({ "type": "flush" }).to_string()
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("SARVAM_API_KEY")
        .expect("SARVAM_API_KEY env var not set");

    let wav_path = std::env::args().nth(1)
        .unwrap_or_else(|| "tests/test.wav".to_string());

    let url = build_url();

    println!("=== Sarvam WebSocket standalone test ===\n");
    println!("URL: {}", url);
    println!("WAV: {}\n", wav_path);

    // --- Read WAV ---
    let mut reader = hound::WavReader::open(&wav_path)?;
    let spec = reader.spec();
    println!("WAV: {}Hz  {}ch  {}bit", spec.sample_rate, spec.channels, spec.bits_per_sample);
    let samples: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
    let pcm_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

    // Use first 10 seconds max
    let max_bytes = 16_000 * 2 * 10;
    let pcm_bytes = &pcm_bytes[..pcm_bytes.len().min(max_bytes)];
    println!("Using {:.1}s of audio\n", pcm_bytes.len() as f32 / (16_000.0 * 2.0));

    // --- Build request ---
    let request = Request::builder()
        .uri(&url)
        .header("Host", "api.sarvam.ai")
        .header("api-subscription-key", &api_key)
        .header("Connection", "Upgrade")
        .header("Upgrade", "websocket")
        .header("Sec-WebSocket-Version", "13")
        .header(
            "Sec-WebSocket-Key",
            tokio_tungstenite::tungstenite::handshake::client::generate_key(),
        )
        .body(())?;

    // --- Connect ---
    let (ws_stream, response) = tokio_tungstenite::connect_async(request).await?;
    println!("Connected! HTTP status: {}\n", response.status());

    let (mut sink, mut stream) = ws_stream.split();

    // --- Send audio in 512-sample chunks ---
    let chunk_bytes = 512 * 2;
    let chunks: Vec<&[u8]> = pcm_bytes.chunks(chunk_bytes).collect();
    println!("Sending {} chunks ({} bytes each)...", chunks.len(), chunk_bytes);

    for chunk in &chunks {
        let msg = audio_message(chunk);
        sink.send(Message::Text(msg.into())).await?;
        tokio::time::sleep(Duration::from_millis(8)).await;
    }
    println!("Sent all audio chunks");

    // --- Send flush ---
    sink.send(Message::Text(flush_message().into())).await?;
    println!("Sent flush signal\n");

    // --- Read responses for up to 8 seconds ---
    println!("Waiting for responses (8s timeout)...\n");
    let timeout = tokio::time::sleep(Duration::from_secs(8));
    tokio::pin!(timeout);

    loop {
        tokio::select! {
            msg = stream.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        println!("Raw: {}", text);

                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                            match v["type"].as_str().unwrap_or("?") {
                                "data" => {
                                    let t = v["data"]["transcript"].as_str().unwrap_or("");
                                    let l = v["data"]["language_code"].as_str().unwrap_or("?");
                                    if t.is_empty() {
                                        println!("  → Empty transcript");
                                    } else {
                                        println!("  ✓ Transcript: \"{}\"  lang={}", t, l);
                                    }
                                }
                                "events" => {
                                    let s = v["data"]["signal_type"].as_str().unwrap_or("?");
                                    println!("  → VAD event: {}", s);
                                }
                                "error" => {
                                    println!("  ✗ Server error: {}", v["data"]);
                                }
                                other => {
                                    println!("  → type={}", other);
                                }
                            }
                        }
                        println!();
                    }
                    Some(Ok(Message::Close(f))) => {
                        println!("Server closed: {:?}", f);
                        break;
                    }
                    Some(Ok(_)) => {}
                    Some(Err(e)) => { eprintln!("Error: {}", e); break; }
                    None        => { println!("Stream ended"); break; }
                }
            }
            _ = &mut timeout => {
                println!("Timeout — closing");
                break;
            }
        }
    }

    let _ = sink.close().await;
    println!("\n=== test complete ===");
    Ok(())
}
