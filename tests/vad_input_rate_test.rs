//! VAD follows the rate of the audio that comes in.
//!
//! Run with:
//!   cargo test --test vad_input_rate_test
//!
//! Audio pushed as `ChannelMessage::AudioFrame` at 48 kHz stereo must reach
//! the VAD as 16 kHz mono windows, at real-time pace: one second of input is
//! about 31 windows of 512 samples, not 93 windows read at the wrong rate.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc;

use rustvani::{
    clock::system_clock,
    error::Result,
    frames::{Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor, SystemFrame},
    pipeline::{PipelineParams, PipelineTask},
    transport::{ChannelMessage, ChannelTransport, TransportParams},
    VadAnalyzer,
};

/// A 16 kHz "model" that records every window it is asked about.
struct RecordingVad {
    windows: Mutex<Vec<usize>>,
}

#[async_trait]
impl VadAnalyzer for RecordingVad {
    fn num_frames_required(&self) -> usize {
        512
    }

    async fn voice_confidence(&self, audio: Vec<u8>) -> f32 {
        self.windows.lock().unwrap().push(audio.len());
        0.0
    }
}

/// Records the format of the audio frames passed downstream.
struct FormatProbe {
    seen: Arc<Mutex<Vec<(u32, u16)>>>,
    count: Arc<AtomicUsize>,
}

#[async_trait]
impl FrameHandler for FormatProbe {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        if let FrameInner::System(SystemFrame::InputAudioRaw(data)) = &frame.inner {
            self.seen.lock().unwrap().push((data.sample_rate, data.num_channels));
            self.count.fetch_add(1, Ordering::Relaxed);
        }
        processor.push_frame(frame, direction).await
    }
}

#[tokio::test]
async fn vad_resamples_48k_stereo_to_its_own_rate() {
    let vad = Arc::new(RecordingVad { windows: Mutex::new(Vec::new()) });

    let (incoming_tx, incoming_rx) = mpsc::channel::<ChannelMessage>(200);
    let (outgoing_tx, _outgoing_rx) = mpsc::channel::<ChannelMessage>(10);
    let transport = ChannelTransport::new(
        "rate-test",
        TransportParams {
            audio_in_enabled: true,
            // What untagged audio would be; the AudioFrames below override it.
            audio_in_sample_rate: Some(16_000),
            audio_in_passthrough: true,
            audio_in_stream_on_start: true,
            vad_analyzer: Some(vad.clone()),
            ..TransportParams::default()
        },
        incoming_rx,
    );

    let seen = Arc::new(Mutex::new(Vec::new()));
    let count = Arc::new(AtomicUsize::new(0));
    let probe = FrameProcessor::new(
        "probe",
        Box::new(FormatProbe { seen: seen.clone(), count: count.clone() }),
        false,
    );
    let task = PipelineTask::new(vec![transport.input(), probe], PipelineParams::default());
    let push_tx = task.push_sender();

    let pipeline = tokio::spawn(async move { task.run(system_clock(), None).await.ok() });
    let transport_run = tokio::spawn(async move { transport.run(push_tx, outgoing_tx).await });
    tokio::time::sleep(Duration::from_millis(100)).await;

    // One second of 48 kHz stereo, in 10 ms frames of a quiet tone.
    const FRAMES: usize = 100;
    for n in 0..FRAMES {
        let audio: Vec<u8> = (0..480)
            .flat_map(|i| {
                let t = (n * 480 + i) as f32 / 48_000.0;
                let s = ((t * 440.0 * std::f32::consts::TAU).sin() * 3000.0) as i16;
                [s, s]
            })
            .flat_map(|s| s.to_le_bytes())
            .collect();
        incoming_tx
            .send(ChannelMessage::AudioFrame { audio, sample_rate: 48_000, channels: 2 })
            .await
            .unwrap();
    }

    // Wait until every frame has passed the transport's audio task.
    for _ in 0..100 {
        if count.load(Ordering::Relaxed) >= FRAMES {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    drop(incoming_tx);
    let _ = tokio::time::timeout(Duration::from_secs(5), pipeline).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), transport_run).await;

    let windows = vad.windows.lock().unwrap().clone();
    assert!(windows.iter().all(|&len| len == 512 * 2), "window sizes: {windows:?}");
    // 16 000 / 512 = 31.25 windows per second, less the resampler's fill.
    assert!((29..=31).contains(&windows.len()), "got {} VAD windows", windows.len());

    // Downstream (STT) still gets the audio as it came in.
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), FRAMES);
    assert!(seen.iter().all(|&f| f == (48_000, 2)), "formats: {seen:?}");
}
