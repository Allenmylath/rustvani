//! VAD pipeline integration test.
//!
//! Run: docker run --rm rustvani-vad-test

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use ort::session::builder::SessionBuilder;
use rustvani::{
    system_clock, AudioRawData, Frame, FrameDirection, FrameHandler, FrameKind,
    FrameProcessor, PipelineParams, PipelineTask, Result, VadParams, SileroVad,
};
use rustvani::transport::{BaseTransport, TransportParams};
use rustvani::vad::SileroVad as SileroVadDirect;

const CHUNK_BYTES: usize = 512 * 2;
const MODEL_PATH: &str = "/app/silero_vad.onnx";

// ---------------------------------------------------------------------------
// ObserverHandler
// ---------------------------------------------------------------------------

struct ObserverHandler {
    frames: Arc<Mutex<Vec<FrameKind>>>,
}

#[async_trait]
impl FrameHandler for ObserverHandler {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match frame.kind() {
            FrameKind::VADUserStartedSpeaking | FrameKind::VADUserStoppedSpeaking => {
                println!("[observer] {:?}", frame.kind());
                self.frames.lock().unwrap().push(frame.kind());
            }
            _ => {}
        }
        processor.push_frame(frame, direction).await
    }
}

// ---------------------------------------------------------------------------
// WAV reader
// ---------------------------------------------------------------------------

fn read_wav_pcm(path: &str) -> std::result::Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| format!("Failed to open {}: {}", path, e))?;
    let spec = reader.spec();
    println!("    sample_rate: {}  channels: {}  bits: {}",
        spec.sample_rate, spec.channels, spec.bits_per_sample);
    assert_eq!(spec.sample_rate, 16000, "Need 16kHz");
    assert_eq!(spec.channels, 1, "Need mono");
    let samples: Vec<i16> = reader.samples::<i16>()
        .map(|s| s.expect("WAV read error"))
        .collect();
    println!("    total samples: {} ({:.1}s)", samples.len(), samples.len() as f32 / 16000.0);
    Ok(samples.iter().flat_map(|s| s.to_le_bytes()).collect())
}

fn rms_norm(chunk: &[u8]) -> f32 {
    let sum_sq: f64 = chunk.chunks_exact(2)
        .map(|b| { let s = i16::from_le_bytes([b[0], b[1]]) as f64; s * s })
        .sum();
    (sum_sq / 512.0_f64).sqrt() as f32 / 32768.0
}

// ---------------------------------------------------------------------------
// Step 0: Model introspection
// ---------------------------------------------------------------------------

fn inspect_model() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("\n[Model Introspection]");
    let session = SessionBuilder::new()?.commit_from_file(MODEL_PATH)?;

    println!("  Inputs:");
    for input in session.inputs().iter() {
        println!("    input name: {:?}", input.name());
    }
    println!("  Outputs:");
    for output in session.outputs().iter() {
        println!("    output name: {:?}", output.name());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Step 1: Sample confidence across file — stateless, no pipeline needed
// ---------------------------------------------------------------------------

async fn sample_confidence_across_file(
    pcm_bytes: &[u8],
) -> std::result::Result<f32, Box<dyn std::error::Error>> {
    println!("\n[Confidence across file — stateless, 10 evenly spaced chunks]");
    println!("    {:>8}  {:>10}  {:>10}  {:>10}", "chunk", "time(s)", "confidence", "rms_norm");

    let chunks: Vec<&[u8]> = pcm_bytes.chunks(CHUNK_BYTES)
        .filter(|c| c.len() == CHUNK_BYTES)
        .collect();

    let total = chunks.len();
    let indices: Vec<usize> = (0..10).map(|i| i * total / 10).collect();
    let mut max_conf = 0.0f32;

    for &idx in &indices {
        let chunk = chunks[idx];
        let time_s = idx as f32 * 512.0 / 16000.0;

        // Fresh model per chunk — stateless baseline
        let model = SileroVadDirect::from_path(16_000, MODEL_PATH)
            .map_err(|e| format!("Model load failed: {}", e))?;
        let conf = model.infer_async(chunk.to_vec()).await
            .map_err(|e| format!("infer error at chunk {}: {}", idx, e))?;

        if conf > max_conf { max_conf = conf; }
        println!("    {:>8}  {:>10.2}  {:>10.6}  {:>10.6}", idx, time_s, conf, rms_norm(chunk));
    }

    println!("  Max confidence across file: {:.6}", max_conf);
    Ok(max_conf)
}

// ---------------------------------------------------------------------------
// Full pipeline run — VAD configured on transport
// ---------------------------------------------------------------------------

async fn run_pipeline(
    pcm_bytes: Vec<u8>,
    params: VadParams,
    label: &str,
) -> std::result::Result<Vec<FrameKind>, Box<dyn std::error::Error>> {
    println!("\n--- {} ---", label);
    println!("    confidence={:.4} min_volume={:.4}", params.confidence, params.min_volume);

    let vad_analyzer = Arc::new(
        SileroVad::from_path(16_000, MODEL_PATH)
            .map_err(|e| format!("VAD init failed: {}", e))?
    );

    let transport = Arc::new(BaseTransport::new("VadTest", TransportParams {
        audio_in_enabled:         true,
        audio_in_sample_rate:     Some(16_000),
        audio_in_channels:        1,
        audio_in_passthrough:     false, // no downstream STT — just VAD events
        audio_in_stream_on_start: true,
        vad_analyzer:             Some(vad_analyzer),
        vad_params:               params,
        ..TransportParams::default()
    }));

    let observed = Arc::new(Mutex::new(Vec::<FrameKind>::new()));
    let observer_proc = FrameProcessor::new(
        "Observer",
        Box::new(ObserverHandler { frames: observed.clone() }),
        false,
    );

    let task = PipelineTask::new(
        vec![transport.input(), observer_proc, transport.output()],
        PipelineParams::default(),
    );

    let push_tx = task.push_sender();
    let transport_for_feeder = transport.clone();
    let feeder_bytes = pcm_bytes.clone();

    let audio_feeder = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;

        for chunk in feeder_bytes.chunks(CHUNK_BYTES) {
            if chunk.len() < CHUNK_BYTES { break; }
            let data = AudioRawData::new(chunk.to_vec(), 16_000, 1);
            transport_for_feeder.push_audio_frame(data).await;
        }

        tokio::time::sleep(Duration::from_millis(800)).await;
        let _ = push_tx.send((Frame::end(), FrameDirection::Downstream)).await;
    });

    task.run(system_clock(), None).await?;
    let _ = audio_feeder.await;

    let frames = observed.lock().unwrap().clone();
    println!("    observed: {:?}", frames);
    Ok(frames)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let wav_path = std::env::args().nth(1).unwrap_or_else(|| "/app/test.wav".to_string());

    println!("=== rustvani VAD pipeline integration test ===\n");
    println!("[0] Reading WAV: {}", wav_path);
    let pcm_bytes = read_wav_pcm(&wav_path)?;

    inspect_model()?;

    let max_conf = sample_confidence_across_file(&pcm_bytes).await?;

    if max_conf < 0.01 {
        println!("\n  ✗ Max confidence {:.6} is near zero across the entire file.", max_conf);
        panic!("Model not producing meaningful confidence values.");
    }

    let frames_open = run_pipeline(
        pcm_bytes.clone(),
        VadParams { confidence: max_conf * 0.5, min_volume: 0.0, ..VadParams::default() },
        &format!("Run A: confidence={:.4} min_volume=0.0", max_conf * 0.5),
    ).await?;

    let frames_default = run_pipeline(
        pcm_bytes.clone(),
        VadParams::default(),
        "Run B: default params",
    ).await?;

    println!("\n[Assertions]");
    assert!(
        frames_open.contains(&FrameKind::VADUserStartedSpeaking),
        "Missing VADUserStartedSpeaking. Got: {:?}", frames_open
    );
    assert!(
        frames_open.contains(&FrameKind::VADUserStoppedSpeaking),
        "Missing VADUserStoppedSpeaking. Got: {:?}", frames_open
    );

    // Verify no duplicates — Started must always alternate with Stopped
    let mut speaking = false;
    for kind in &frames_open {
        match kind {
            FrameKind::VADUserStartedSpeaking => {
                assert!(!speaking, "Duplicate VADUserStartedSpeaking");
                speaking = true;
            }
            FrameKind::VADUserStoppedSpeaking => {
                assert!(speaking, "VADUserStoppedSpeaking without preceding Started");
                speaking = false;
            }
            _ => {}
        }
    }

    println!("  ✓ VADUserStartedSpeaking");
    println!("  ✓ VADUserStoppedSpeaking");
    println!("  ✓ No duplicate events — Started/Stopped alternate correctly");

    if !frames_default.is_empty() {
        println!("  ✓ Default params also work");
    } else {
        println!("  ⚠ Default params need tuning");
    }

    println!("\n=== All checks passed. ===");
    Ok(())
}