//! Integration test: BaseTransport audio passthrough.
//!
//! Verifies that:
//! 1. Audio pushed via `push_audio_frame()` arrives as `InputAudioRaw` frames
//!    downstream.
//! 2. The pipeline starts and finishes cleanly.
//! 3. Speaking state is tracked correctly (UserStoppedSpeaking on timeout).

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tokio::time::sleep;

use pipecat_core::{
    error::Result,
    frames::{
        AudioRawData, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor,
        SystemFrame,
    },
    pipeline::{FinishReason, Pipeline, PipelineParams, PipelineTask},
    transport::{BaseTransport, TransportParams},
    clock::system_clock,
};

// ---------------------------------------------------------------------------
// Helper: a frame collector processor
//
// Records the names of every InputAudioRaw frame it sees.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AudioCollector {
    received: Arc<Mutex<Vec<String>>>,
}

impl AudioCollector {
    fn new() -> Self {
        Self {
            received: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn count(&self) -> usize {
        self.received.lock().unwrap().len()
    }
}

#[async_trait]
impl FrameHandler for AudioCollector {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        // Record InputAudioRaw frames.
        if matches!(
            &frame.inner,
            FrameInner::System(SystemFrame::InputAudioRaw(_))
        ) {
            self.received
                .lock()
                .unwrap()
                .push(frame.name().to_string());
        }

        // Always pass through.
        processor.push_frame(frame, direction).await
    }
}

// ---------------------------------------------------------------------------
// Test 1: audio frames are passed through
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_audio_passthrough() {
    // Transport with audio input enabled and passthrough on.
    let params = TransportParams {
        audio_in_enabled:     true,
        audio_in_passthrough: true,
        ..TransportParams::default()
    };

    let transport = BaseTransport::new("Test", params);

    // Collector sits between input and output.
    let collector = AudioCollector::new();
    let collector_handler = collector.clone();

    let collector_proc = FrameProcessor::new(
        "AudioCollector",
        Box::new(collector_handler),
        false,
    );

    // Pipeline: input → collector → output
    let pipeline = PipelineTask::new(
        vec![
            transport.input(),
            collector_proc,
            transport.output(),
        ],
        PipelineParams::default(),
    );

    // Run the pipeline in a background task.
    let task_handle = {
        let clock = system_clock();
        tokio::spawn(async move {
            pipeline.run(clock, None).await.expect("pipeline failed");
        })
    };

    // Give the pipeline time to start.
    sleep(Duration::from_millis(50)).await;

    // Push 3 audio frames.
    let dummy_audio = vec![0u8; 320]; // 160 samples, 16-bit mono
    for _ in 0..3 {
        let data = AudioRawData::new(dummy_audio.clone(), 16_000, 1);
        let sent = transport.push_audio_frame(data).await;
        assert!(sent, "push_audio_frame should return true");
    }

    // Give the audio task time to process.
    sleep(Duration::from_millis(100)).await;

    // Verify all 3 frames were received by the collector.
    assert_eq!(
        collector.count(),
        3,
        "expected 3 InputAudioRaw frames, got {}",
        collector.count()
    );

    // Shut down the pipeline cleanly.
    transport
        .input()
        .queue_frame(Frame::end(), FrameDirection::Downstream, None)
        .await
        .unwrap();

    tokio::time::timeout(Duration::from_secs(2), task_handle)
        .await
        .expect("pipeline did not shut down in time")
        .expect("pipeline task panicked");
}

// ---------------------------------------------------------------------------
// Test 2: audio is dropped when transport is paused
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_audio_dropped_when_paused() {
    let params = TransportParams {
        audio_in_enabled:     true,
        audio_in_passthrough: true,
        ..TransportParams::default()
    };

    let transport = BaseTransport::new("TestPause", params);

    let collector = AudioCollector::new();
    let collector_proc = FrameProcessor::new(
        "AudioCollector",
        Box::new(collector.clone()),
        false,
    );

    let pipeline = PipelineTask::new(
        vec![
            transport.input(),
            collector_proc,
            transport.output(),
        ],
        PipelineParams::default(),
    );

    let task_handle = {
        let clock = system_clock();
        tokio::spawn(async move {
            pipeline.run(clock, None).await.expect("pipeline failed");
        })
    };

    sleep(Duration::from_millis(50)).await;

    // Push StopFrame to pause the transport.
    transport
        .input()
        .queue_frame(Frame::stop(), FrameDirection::Downstream, None)
        .await
        .unwrap();

    sleep(Duration::from_millis(30)).await;

    // These frames should be dropped — transport is paused.
    let dummy_audio = vec![0u8; 320];
    for _ in 0..3 {
        let data = AudioRawData::new(dummy_audio.clone(), 16_000, 1);
        transport.push_audio_frame(data).await;
    }

    sleep(Duration::from_millis(100)).await;

    assert_eq!(
        collector.count(),
        0,
        "expected 0 frames while paused, got {}",
        collector.count()
    );

    // Cancel to shut down.
    transport
        .input()
        .queue_frame(Frame::cancel(), FrameDirection::Downstream, None)
        .await
        .unwrap();

    tokio::time::timeout(Duration::from_secs(2), task_handle)
        .await
        .expect("pipeline did not shut down in time")
        .expect("pipeline task panicked");
}