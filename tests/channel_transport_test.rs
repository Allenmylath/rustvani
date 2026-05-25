//! ChannelTransport integration test.
//!
//! Run with:
//!   cargo test --test channel_transport_test
//!
//! What happens:
//!   1. A ChannelTransport is created with mpsc channels for input and output.
//!   2. A tiny pipeline is built: transport.input() → EchoProcessor → transport.output().
//!   3. Audio and text frames are pushed into the incoming channel.
//!   4. The EchoProcessor converts InputAudioRaw → OutputAudioRaw and
//!      RaviClientMessage → RaviServerResponse.
//!   5. The transport forwards the results to the outgoing channel.
//!   6. We assert that what comes out matches what went in.

use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc;

use rustvani::{
    clock::system_clock,
    error::Result,
    frames::{
        Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor, SystemFrame,
    },
    pipeline::{PipelineParams, PipelineTask},
    transport::{ChannelMessage, ChannelTransport, TransportParams},
};

// ---------------------------------------------------------------------------
// EchoProcessor — mirrors audio and RAVI messages back to the output side
// ---------------------------------------------------------------------------

struct EchoProcessor;

#[async_trait]
impl FrameHandler for EchoProcessor {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match &frame.inner {
            // Echo audio: InputAudioRaw → OutputAudioRaw
            FrameInner::System(SystemFrame::InputAudioRaw(data)) => {
                let echo = Frame::output_audio_raw(data.clone());
                processor.push_frame(echo, direction).await?;
            }

            // Echo RAVI: RaviClientMessage → RaviServerResponse
            FrameInner::System(SystemFrame::RaviClientMessage {
                msg_id,
                msg_type,
                data,
            }) => {
                let payload = format!(
                    r#"{{"echo_id":"{}","echo_type":"{}","echo_data":{}}}"#,
                    msg_id,
                    msg_type,
                    data.as_deref().unwrap_or("null")
                );
                let response = Frame::ravi_server_response(msg_id, payload);
                processor.push_frame(response, direction).await?;
            }

            // Pass everything else through
            _ => {
                processor.push_frame(frame, direction).await?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn transport_params() -> TransportParams {
    TransportParams {
        audio_in_enabled: true,
        audio_in_sample_rate: Some(16_000),
        audio_in_channels: 1,
        audio_in_passthrough: true,
        audio_in_stream_on_start: true,
        audio_out_enabled: true,
        audio_out_sample_rate: Some(16_000),
        audio_out_channels: 1,
        audio_out_10ms_chunks: 4,
        ..TransportParams::default()
    }
}

fn build_pipeline(
    transport: &ChannelTransport,
) -> (PipelineTask, mpsc::Sender<(Frame, FrameDirection)>) {
    let echo = FrameProcessor::new("Echo", Box::new(EchoProcessor), false);
    let task = PipelineTask::new(
        vec![transport.input(), echo, transport.output()],
        PipelineParams::default(),
    );
    let push_tx = task.push_sender();
    (task, push_tx)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_audio_round_trip() {
    // Channels
    let (incoming_tx, incoming_rx) = mpsc::channel::<ChannelMessage>(10);
    let (outgoing_tx, mut outgoing_rx) = mpsc::channel::<ChannelMessage>(10);

    // Transport + pipeline
    let transport = ChannelTransport::new("audio-test", transport_params(), incoming_rx);
    let (task, push_tx) = build_pipeline(&transport);

    // Run pipeline
    let pipeline_handle = tokio::spawn(async move {
        task.run(system_clock(), None).await.ok();
    });

    // Run transport
    let transport_handle = tokio::spawn(async move {
        transport.run(push_tx, outgoing_tx).await;
    });

    // Send exactly one chunk of audio (1280 bytes @ 16kHz mono, 40ms).
    let fake_audio: Vec<u8> = (0..1280u16).map(|i| (i % 256) as u8).collect();
    incoming_tx
        .send(ChannelMessage::Audio(fake_audio.clone()))
        .await
        .unwrap();

    // Expect the same chunk back
    let received = tokio::time::timeout(Duration::from_secs(3), outgoing_rx.recv())
        .await
        .expect("timeout waiting for audio echo")
        .expect("outgoing channel closed");

    assert!(
        matches!(received, ChannelMessage::Audio(ref bytes) if bytes == &fake_audio),
        "expected audio echo, got {:?}",
        received
    );

    // Close incoming channel → transport sends EndFrame → pipeline shuts down
    drop(incoming_tx);

    let _ = tokio::time::timeout(Duration::from_secs(3), pipeline_handle).await;
    let _ = tokio::time::timeout(Duration::from_secs(3), transport_handle).await;
}

#[tokio::test]
async fn test_text_round_trip() {
    // Channels
    let (incoming_tx, incoming_rx) = mpsc::channel::<ChannelMessage>(10);
    let (outgoing_tx, mut outgoing_rx) = mpsc::channel::<ChannelMessage>(10);

    // Transport + pipeline
    let transport = ChannelTransport::new("text-test", transport_params(), incoming_rx);
    let (task, push_tx) = build_pipeline(&transport);

    // Run pipeline
    let pipeline_handle = tokio::spawn(async move {
        task.run(system_clock(), None).await.ok();
    });

    // Run transport
    let transport_handle = tokio::spawn(async move {
        transport.run(push_tx, outgoing_tx).await;
    });

    // Send a RAVI client message as JSON text.
    let ravi_json = r#"{"label":"ravi","id":"msg-42","type":"action","data":"{\"key\":\"value\"}"}"#;
    incoming_tx
        .send(ChannelMessage::Text(ravi_json.to_string()))
        .await
        .unwrap();

    // Expect the echoed server response back.
    let received = tokio::time::timeout(Duration::from_secs(3), outgoing_rx.recv())
        .await
        .expect("timeout waiting for text echo")
        .expect("outgoing channel closed");

    match received {
        ChannelMessage::Text(text) => {
            assert!(text.contains("msg-42"), "response should echo the message id");
            assert!(text.contains("action"), "response should echo the message type");
        }
        other => panic!("expected text response, got {:?}", other),
    }

    // Graceful shutdown via dropping the incoming channel.
    drop(incoming_tx);

    let _ = tokio::time::timeout(Duration::from_secs(3), pipeline_handle).await;
    let _ = tokio::time::timeout(Duration::from_secs(3), transport_handle).await;
}

#[tokio::test]
async fn test_interruption() {
    // Channels
    let (incoming_tx, incoming_rx) = mpsc::channel::<ChannelMessage>(10);
    let (outgoing_tx, mut outgoing_rx) = mpsc::channel::<ChannelMessage>(10);

    // Transport + pipeline
    let transport = ChannelTransport::new("interrupt-test", transport_params(), incoming_rx);
    let (task, push_tx) = build_pipeline(&transport);

    // Run pipeline
    let pipeline_handle = tokio::spawn(async move {
        task.run(system_clock(), None).await.ok();
    });

    // Run transport
    let transport_handle = tokio::spawn(async move {
        transport.run(push_tx, outgoing_tx).await;
    });

    // Send an interruption.
    incoming_tx
        .send(ChannelMessage::Interruption)
        .await
        .unwrap();

    // Expect the interruption marker back.
    let received = tokio::time::timeout(Duration::from_secs(3), outgoing_rx.recv())
        .await
        .expect("timeout waiting for interruption")
        .expect("outgoing channel closed");

    assert!(
        matches!(received, ChannelMessage::Interruption),
        "expected interruption, got {:?}",
        received
    );

    drop(incoming_tx);

    let _ = tokio::time::timeout(Duration::from_secs(3), pipeline_handle).await;
    let _ = tokio::time::timeout(Duration::from_secs(3), transport_handle).await;
}
