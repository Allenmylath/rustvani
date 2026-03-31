//! Transport integration tests.
//!
//! Tests BaseTransport, BaseInputTransport and BaseOutputTransport behaviour
//! without requiring the Silero model.
//!
//! Covers:
//! - TransportParams defaults
//! - Audio passthrough end-to-end
//! - Audio dropped when input disabled
//! - Audio dropped when paused (StopFrame)
//! - Concurrent audio push via cloned sender
//! - BotStartedSpeaking emitted on first OutputAudioRaw
//! - BotStoppedSpeaking emitted on interruption
//! - Clean shutdown on EndFrame and CancelFrame

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tokio::time::sleep;

use rustvani::{
    clock::system_clock,
    error::Result,
    frames::{
        AudioRawData, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor,
        SystemFrame,
    },
    pipeline::{PipelineParams, PipelineTask},
    transport::{BaseTransport, TransportParams},
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Dummy PCM: 160 samples of silence, 16-bit mono @ 16kHz = 320 bytes
fn silent_audio() -> AudioRawData {
    AudioRawData::new(vec![0u8; 320], 16_000, 1)
}

/// Dummy PCM: 160 samples of mid-amplitude audio
fn loud_audio() -> AudioRawData {
    let sample: i16 = 16384;
    let bytes: Vec<u8> = sample
        .to_le_bytes()
        .iter()
        .cycle()
        .take(320)
        .cloned()
        .collect();
    AudioRawData::new(bytes, 16_000, 1)
}

/// Collects frame names seen downstream.
#[derive(Clone)]
struct FrameCollector {
    frames: Arc<Mutex<Vec<&'static str>>>,
}

impl FrameCollector {
    fn new() -> Self {
        Self {
            frames: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn names(&self) -> Vec<&'static str> {
        self.frames.lock().unwrap().clone()
    }

    fn count_of(&self, name: &str) -> usize {
        self.frames
            .lock()
            .unwrap()
            .iter()
            .filter(|&&n| n == name)
            .count()
    }

    fn contains(&self, name: &str) -> bool {
        self.frames.lock().unwrap().iter().any(|&n| n == name)
    }
}

#[async_trait]
impl FrameHandler for FrameCollector {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        self.frames.lock().unwrap().push(frame.name());
        processor.push_frame(frame, direction).await
    }
}

/// Build a pipeline with input → collector → output and return
/// (task_handle, transport, collector).
async fn build_pipeline(
    params: TransportParams,
) -> (
    tokio::task::JoinHandle<()>,
    BaseTransport,
    FrameCollector,
) {
    let transport = BaseTransport::new("Test", params);
    let collector = FrameCollector::new();

    let collector_proc = FrameProcessor::new(
        "Collector",
        Box::new(collector.clone()),
        false,
    );

    let task = PipelineTask::new(
        vec![
            transport.input(),
            collector_proc,
            transport.output(),
        ],
        PipelineParams::default(),
    );

    let handle = tokio::spawn(async move {
        task.run(system_clock(), None)
            .await
            .expect("pipeline failed");
    });

    // Let the pipeline start before pushing frames.
    sleep(Duration::from_millis(50)).await;

    (handle, transport, collector)
}

/// Shut down the pipeline and wait for it.
async fn shutdown(transport: &BaseTransport, handle: tokio::task::JoinHandle<()>) {
    transport
        .input()
        .queue_frame(Frame::end(), FrameDirection::Downstream, None)
        .await
        .unwrap();

    tokio::time::timeout(Duration::from_secs(2), handle)
        .await
        .expect("pipeline did not shut down in time")
        .expect("pipeline task panicked");
}

// ---------------------------------------------------------------------------
// TransportParams
// ---------------------------------------------------------------------------

#[test]
fn test_transport_params_defaults() {
    let p = TransportParams::default();
    assert!(!p.audio_in_enabled,     "audio_in_enabled should default false");
    assert!(p.audio_in_passthrough,  "audio_in_passthrough should default true");
    assert!(p.audio_in_stream_on_start, "audio_in_stream_on_start should default true");
    assert_eq!(p.audio_in_channels, 1);
    assert!(!p.audio_out_enabled,    "audio_out_enabled should default false");
    assert_eq!(p.audio_out_channels, 1);
    assert_eq!(p.audio_out_bitrate,  96_000);
    assert!(!p.vad_enabled,          "vad_enabled should default false");
}

#[test]
fn test_transport_params_vad_enabled() {
    let p = TransportParams {
        vad_enabled: true,
        ..TransportParams::default()
    };
    assert!(p.vad_enabled);
}

// ---------------------------------------------------------------------------
// Test 1: audio passthrough — frames arrive at collector
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_audio_passthrough_through_pipeline() {
    let params = TransportParams {
        audio_in_enabled:     true,
        audio_in_passthrough: true,
        ..TransportParams::default()
    };

    let (handle, transport, collector) = build_pipeline(params).await;

    // Push 5 audio frames
    for _ in 0..5 {
        let sent = transport.push_audio_frame(silent_audio()).await;
        assert!(sent, "push_audio_frame should succeed");
    }

    sleep(Duration::from_millis(100)).await;

    assert_eq!(
        collector.count_of("InputAudioRawFrame"),
        5,
        "all 5 audio frames should reach the collector"
    );

    shutdown(&transport, handle).await;
}

// ---------------------------------------------------------------------------
// Test 2: audio disabled — no frames reach collector
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_audio_not_sent_when_disabled() {
    let params = TransportParams {
        audio_in_enabled: false, // disabled
        ..TransportParams::default()
    };

    let (handle, transport, collector) = build_pipeline(params).await;

    for _ in 0..5 {
        transport.push_audio_frame(silent_audio()).await;
    }

    sleep(Duration::from_millis(100)).await;

    assert_eq!(
        collector.count_of("InputAudioRawFrame"),
        0,
        "no audio should reach collector when audio_in_enabled is false"
    );

    shutdown(&transport, handle).await;
}

// ---------------------------------------------------------------------------
// Test 3: passthrough disabled — transport receives audio but doesn't forward
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_audio_not_forwarded_when_passthrough_disabled() {
    let params = TransportParams {
        audio_in_enabled:     true,
        audio_in_passthrough: false, // no downstream push
        ..TransportParams::default()
    };

    let (handle, transport, collector) = build_pipeline(params).await;

    for _ in 0..5 {
        transport.push_audio_frame(silent_audio()).await;
    }

    sleep(Duration::from_millis(100)).await;

    assert_eq!(
        collector.count_of("InputAudioRawFrame"),
        0,
        "audio should not reach collector when passthrough is disabled"
    );

    shutdown(&transport, handle).await;
}

// ---------------------------------------------------------------------------
// Test 4: audio dropped after StopFrame
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_audio_dropped_after_stop_frame() {
    let params = TransportParams {
        audio_in_enabled:     true,
        audio_in_passthrough: true,
        ..TransportParams::default()
    };

    let (handle, transport, collector) = build_pipeline(params).await;

    // Pause the transport
    transport
        .input()
        .queue_frame(Frame::stop(), FrameDirection::Downstream, None)
        .await
        .unwrap();

    sleep(Duration::from_millis(50)).await;

    // Push audio — should be dropped
    for _ in 0..5 {
        transport.push_audio_frame(silent_audio()).await;
    }

    sleep(Duration::from_millis(100)).await;

    assert_eq!(
        collector.count_of("InputAudioRawFrame"),
        0,
        "audio should be dropped after StopFrame"
    );

    // Cancel to exit (pipeline is paused, EndFrame won't drain)
    transport
        .input()
        .queue_frame(Frame::cancel(), FrameDirection::Downstream, None)
        .await
        .unwrap();

    tokio::time::timeout(Duration::from_secs(2), handle)
        .await
        .expect("pipeline did not shut down")
        .expect("pipeline panicked");
}

// ---------------------------------------------------------------------------
// Test 5: concurrent push via cloned audio_sender
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_concurrent_audio_push_via_sender() {
    let params = TransportParams {
        audio_in_enabled:     true,
        audio_in_passthrough: true,
        ..TransportParams::default()
    };

    let (handle, transport, collector) = build_pipeline(params).await;

    // Clone sender and push from two concurrent tasks
    let sender1 = transport.audio_sender();
    let sender2 = transport.audio_sender();

    let t1 = tokio::spawn(async move {
        for _ in 0..5 {
            sender1.send(silent_audio()).await.unwrap();
        }
    });

    let t2 = tokio::spawn(async move {
        for _ in 0..5 {
            sender2.send(loud_audio()).await.unwrap();
        }
    });

    t1.await.unwrap();
    t2.await.unwrap();

    sleep(Duration::from_millis(150)).await;

    assert_eq!(
        collector.count_of("InputAudioRawFrame"),
        10,
        "all 10 frames from concurrent senders should arrive"
    );

    shutdown(&transport, handle).await;
}

// ---------------------------------------------------------------------------
// Test 6: BotStartedSpeaking emitted on first OutputAudioRaw
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_bot_started_speaking_on_first_audio_out() {
    let params = TransportParams {
        audio_out_enabled: true,
        ..TransportParams::default()
    };

    let (handle, transport, collector) = build_pipeline(params).await;

    // Push OutputAudioRaw directly into the pipeline
    let audio_frame = Frame::output_audio(vec![0u8; 320], 24_000, 1);
    transport
        .output()
        .queue_frame(audio_frame, FrameDirection::Upstream, None)
        .await
        .unwrap();

    sleep(Duration::from_millis(100)).await;

    assert!(
        collector.contains("BotStartedSpeakingFrame"),
        "BotStartedSpeakingFrame should be emitted on first OutputAudioRaw, got: {:?}",
        collector.names()
    );

    shutdown(&transport, handle).await;
}

// ---------------------------------------------------------------------------
// Test 7: BotStoppedSpeaking emitted on InterruptionFrame
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_bot_stopped_speaking_on_interruption() {
    let params = TransportParams {
        audio_out_enabled: true,
        ..TransportParams::default()
    };

    let (handle, transport, collector) = build_pipeline(params).await;

    // Start bot speaking
    let audio_frame = Frame::output_audio(vec![0u8; 320], 24_000, 1);
    transport
        .output()
        .queue_frame(audio_frame, FrameDirection::Upstream, None)
        .await
        .unwrap();

    sleep(Duration::from_millis(50)).await;

    // Send interruption
    transport
        .input()
        .queue_frame(Frame::interruption(), FrameDirection::Downstream, None)
        .await
        .unwrap();

    sleep(Duration::from_millis(100)).await;

    assert!(
        collector.contains("BotStoppedSpeakingFrame"),
        "BotStoppedSpeakingFrame should be emitted on interruption, got: {:?}",
        collector.names()
    );

    shutdown(&transport, handle).await;
}

// ---------------------------------------------------------------------------
// Test 8: clean shutdown on CancelFrame
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_clean_shutdown_on_cancel() {
    let params = TransportParams {
        audio_in_enabled:     true,
        audio_in_passthrough: true,
        ..TransportParams::default()
    };

    let (handle, transport, _collector) = build_pipeline(params).await;

    transport
        .input()
        .queue_frame(Frame::cancel(), FrameDirection::Downstream, None)
        .await
        .unwrap();

    tokio::time::timeout(Duration::from_secs(2), handle)
        .await
        .expect("pipeline did not shut down after CancelFrame")
        .expect("pipeline task panicked");
}