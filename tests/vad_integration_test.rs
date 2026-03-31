//! VAD integration test — synthetic audio through a real pipeline.
//!
//! Requires the Silero ONNX model at src/vad/data/silero_vad.onnx.
//!
//! Verifies:
//! - Audio chunked at 320 bytes (10ms @ 16kHz) accumulates correctly into
//!   Silero windows (1024 bytes = 512 frames)
//! - Sustained loud audio → VADUserStartedSpeaking emitted
//! - Sustained silence   → VADUserStoppedSpeaking emitted
//! - No ErrorFrame emitted during the run

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tokio::time::sleep;

use rustvani::{
    clock::system_clock,
    error::Result,
    frames::{
        AudioRawData, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor,
        FrameKind, SystemFrame,
    },
    pipeline::{PipelineParams, PipelineTask},
    transport::{BaseTransport, TransportParams},
    vad::{VadParams, VadProcessor},
};

const CHUNK_BYTES: usize = 320;

fn loud_chunk() -> AudioRawData {
    let sample: i16 = i16::MAX;
    let bytes: Vec<u8> = sample.to_le_bytes().iter().cycle().take(CHUNK_BYTES).cloned().collect();
    AudioRawData::new(bytes, 16_000, 1)
}

fn silent_chunk() -> AudioRawData {
    AudioRawData::new(vec![0u8; CHUNK_BYTES], 16_000, 1)
}

#[derive(Clone)]
struct FrameCollector {
    frames: Arc<Mutex<Vec<FrameKind>>>,
    errors: Arc<Mutex<Vec<String>>>,
}

impl FrameCollector {
    fn new() -> Self {
        Self {
            frames: Arc::new(Mutex::new(Vec::new())),
            errors: Arc::new(Mutex::new(Vec::new())),
        }
    }
    fn contains(&self, kind: FrameKind) -> bool {
        self.frames.lock().unwrap().contains(&kind)
    }
    fn error_count(&self) -> usize {
        self.errors.lock().unwrap().len()
    }
    fn all_kinds(&self) -> Vec<FrameKind> {
        self.frames.lock().unwrap().clone()
    }
}

#[async_trait]
impl rustvani::frames::FrameHandler for FrameCollector {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        self.frames.lock().unwrap().push(frame.kind());
        if let FrameInner::System(SystemFrame::Error(ref e)) = frame.inner {
            self.errors.lock().unwrap().push(e.error.clone());
        }
        processor.push_frame(frame, direction).await
    }
}

#[tokio::test]
async fn test_vad_responds_to_synthetic_audio() {
    let params = TransportParams {
        audio_in_enabled:         true,
        audio_in_passthrough:     true,
        audio_in_sample_rate:     Some(16_000),
        audio_in_stream_on_start: true,
        ..TransportParams::default()
    };
    let transport = BaseTransport::new("VadTest", params);

    let vad = VadProcessor::new(16_000, VadParams::default())
        .expect("Failed to load Silero ONNX model")
        .into_processor();

    let collector = FrameCollector::new();
    let collector_proc = FrameProcessor::new("Collector", Box::new(collector.clone()), false);

    let task = PipelineTask::new(
        vec![transport.input(), vad, collector_proc, transport.output()],
        PipelineParams::default(),
    );

    let task_handle = tokio::spawn(async move {
        task.run(system_clock(), None).await.expect("pipeline failed");
    });

    sleep(Duration::from_millis(100)).await;

    println!("Phase 1: pushing loud audio");
    for i in 0..60 {
        let sent = transport.push_audio_frame(loud_chunk()).await;
        assert!(sent, "push_audio_frame failed on chunk {}", i);
        if i % 10 == 0 { sleep(Duration::from_millis(5)).await; }
    }

    sleep(Duration::from_millis(2000)).await;
    println!("After loud phase — frames: {:?}", collector.all_kinds());
    assert!(
        collector.contains(FrameKind::VADUserStartedSpeaking),
        "VADUserStartedSpeaking not emitted. Frames: {:?}", collector.all_kinds()
    );

    println!("Phase 2: pushing silence");
    for i in 0..50 {
        transport.push_audio_frame(silent_chunk()).await;
        if i % 10 == 0 { sleep(Duration::from_millis(5)).await; }
    }

    sleep(Duration::from_millis(2000)).await;
    println!("After silence phase — frames: {:?}", collector.all_kinds());
    assert!(
        collector.contains(FrameKind::VADUserStoppedSpeaking),
        "VADUserStoppedSpeaking not emitted. Frames: {:?}", collector.all_kinds()
    );

    assert_eq!(collector.error_count(), 0, "Errors: {:?}", collector.errors.lock().unwrap());

    transport.input().queue_frame(Frame::end(), FrameDirection::Downstream, None).await.unwrap();
    tokio::time::timeout(Duration::from_secs(5), task_handle)
        .await.expect("pipeline did not shut down").expect("pipeline panicked");

    println!("VAD integration test passed.");
}
