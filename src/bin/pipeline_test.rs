//! Pipeline integration test.
//!
//! Topology:
//!   transport.input() → PrintProcessor → transport.output()
//!
//! Audio is fed via BaseTransport::push_audio_frame() (realistic path).
//! Every frame that passes through PrintProcessor is logged with timestamp.
//!
//! Run: docker run --rm rustvani-pipeline-test

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use rustvani::{
    system_clock, AudioRawData, Frame, FrameDirection, FrameHandler, FrameProcessor,
    PipelineParams, PipelineTask, Result, VadParams, VadProcessor,
};
use rustvani::transport::{BaseTransport, TransportParams};

const CHUNK_BYTES: usize = 512 * 2; // 512 samples × 2 bytes (i16 LE)
const WAV_PATH:   &str   = "/app/test.wav";

// ---------------------------------------------------------------------------
// PrintProcessor
// ---------------------------------------------------------------------------

struct PrintHandler;

#[async_trait]
impl FrameHandler for PrintHandler {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        println!(
            "[print] ts={:.6}  dir={:?}  frame={}  id={}",
            ts,
            direction,
            frame.name(),
            frame.id,
        );

        processor.push_frame(frame, direction).await
    }
}

// ---------------------------------------------------------------------------
// WAV reader — identical to vad_test
// ---------------------------------------------------------------------------

fn read_wav_pcm(path: &str) -> std::result::Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| format!("Failed to open {}: {}", path, e))?;
    let spec = reader.spec();
    println!(
        "[wav]  sample_rate={}  channels={}  bits={}",
        spec.sample_rate, spec.channels, spec.bits_per_sample
    );
    assert_eq!(spec.sample_rate, 16_000, "Need 16 kHz");
    assert_eq!(spec.channels, 1, "Need mono");

    let samples: Vec<i16> = reader
        .samples::<i16>()
        .map(|s| s.expect("WAV read error"))
        .collect();

    println!(
        "[wav]  total_samples={}  duration={:.1}s",
        samples.len(),
        samples.len() as f32 / 16_000.0
    );

    Ok(samples.iter().flat_map(|s| s.to_le_bytes()).collect())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
    .init();

    println!("=== rustvani pipeline integration test ===\n");

    let pcm_bytes = read_wav_pcm(WAV_PATH)?;

    // ---- Transport ----
    let params = TransportParams {
        audio_in_enabled:         true,
        audio_in_sample_rate:     Some(16_000),
        audio_in_channels:        1,
        audio_in_passthrough:     true,
        audio_in_stream_on_start: true,
        ..TransportParams::default()
    };
    let transport = Arc::new(BaseTransport::new("Test", params));

    // ---- VadProcessor ----
    let vad = VadProcessor::new(16_000, VadParams::default())?.into_processor();

    // ---- PrintProcessor ----
    let printer = FrameProcessor::new("PrintProcessor", Box::new(PrintHandler), false);

    // ---- Pipeline ----
    let task = PipelineTask::new(
        vec![transport.input(), vad, printer, transport.output()],
        PipelineParams::default(),
    );

    // Clone sender before run() consumes the receiver.
    let push_tx = task.push_sender();

    // ---- Audio feeder task ----
    // Pushes all PCM chunks through transport (realistic path), then sends End.
    let transport_for_feeder = transport.clone();
    let feeder = tokio::spawn(async move {
        let chunks: Vec<&[u8]> = pcm_bytes.chunks(CHUNK_BYTES)
            .filter(|c| c.len() == CHUNK_BYTES)
            .collect();

        println!("[feeder] pushing {} chunks via transport.push_audio_frame()", chunks.len());

        for chunk in chunks {
            let data = AudioRawData::new(chunk.to_vec(), 16_000, 1);
            transport_for_feeder.push_audio_frame(data).await;
        }

        println!("[feeder] all audio pushed — waiting 300ms for audio task to drain");
        tokio::time::sleep(Duration::from_millis(300)).await;

        println!("[feeder] sending Frame::end()");
        let _ = push_tx.send((Frame::end(), FrameDirection::Downstream)).await;
    });

    // ---- Run pipeline (blocks until End reaches TaskSink) ----
    task.run(system_clock(), None).await?;
    let _ = feeder.await;

    println!("\n=== pipeline test complete ===");
    Ok(())
}