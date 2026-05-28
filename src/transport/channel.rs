use std::sync::Arc;

use tokio::sync::mpsc;

use crate::frames::{AudioRawData, Frame, FrameDirection, FrameProcessor};
use crate::transport::{BaseTransport, OutputMessage, TransportParams};
use crate::transport::incoming::dispatch_text_message;

// ---------------------------------------------------------------------------
// ChannelMessage
// ---------------------------------------------------------------------------

/// Messages flowing through the channel transport.
///
/// This is the bidirectional counterpart to [`OutputMessage`]: it is used
/// for *both* incoming data (outside world → pipeline) and outgoing data
/// (pipeline → outside world).
#[derive(Debug, Clone)]
pub enum ChannelMessage {
    /// Raw PCM audio bytes.
    Audio(Vec<u8>),
    /// Serialised JSON text frame — used by RAVI for protocol messages.
    Text(String),
    /// Interruption signal (legacy).
    Interruption,
    /// Client VAD: user started speaking. Carries wall-clock timestamp in seconds.
    ClientVadStart(f64),
    /// Client VAD: user stopped speaking. Carries wall-clock timestamp in seconds.
    ClientVadStop(f64),
}

// ---------------------------------------------------------------------------
// ChannelTransport
// ---------------------------------------------------------------------------

pub struct ChannelTransport {
    base: Arc<BaseTransport>,
    incoming_rx: std::sync::Mutex<Option<mpsc::Receiver<ChannelMessage>>>,
    audio_out_rx: std::sync::Mutex<Option<mpsc::Receiver<OutputMessage>>>,
    sample_rate: u32,
    channels: u16,
}

const AUDIO_OUT_CHANNEL_CAP: usize = 150;

impl ChannelTransport {
    pub fn new(
        name: &str,
        params: TransportParams,
        incoming_rx: mpsc::Receiver<ChannelMessage>,
    ) -> Self {
        let base = Arc::new(BaseTransport::new(name, params.clone()));

        let (audio_out_tx, audio_out_rx) = mpsc::channel::<OutputMessage>(AUDIO_OUT_CHANNEL_CAP);
        base.set_audio_out_tx(audio_out_tx);

        let sample_rate = params.audio_in_sample_rate.unwrap_or(16_000);
        let channels = params.audio_in_channels.max(1);

        Self {
            base,
            incoming_rx: std::sync::Mutex::new(Some(incoming_rx)),
            audio_out_rx: std::sync::Mutex::new(Some(audio_out_rx)),
            sample_rate,
            channels,
        }
    }

    /// The input FrameProcessor — place first in the pipeline.
    pub fn input(&self) -> FrameProcessor {
        self.base.input()
    }

    /// The output FrameProcessor — place last in the pipeline.
    pub fn output(&self) -> FrameProcessor {
        self.base.output()
    }

    /// Drive the channel transport until either direction closes.
    ///
    /// Arm 1 — `incoming_rx.recv()`: incoming messages from the outside world.
    ///
    ///   - `Audio(bytes)`  → raw PCM audio → pipeline via `push_audio_frame`.
    ///   - `Text(json)`    → parsed as RAVI or legacy interruption → pushed
    ///                       into the pipeline via `push_tx`.
    ///   - `Interruption`  → `InterruptionFrame` downstream.
    ///
    /// Arm 2 — `audio_out_rx.recv()`: outgoing pipeline messages.
    ///
    ///   - `Audio(bytes)`  → `ChannelMessage::Audio`.
    ///   - `Text(json)`    → `ChannelMessage::Text` (RAVI protocol messages).
    ///   - `Interruption`  → drains stale audio, then sends
    ///                       `ChannelMessage::Interruption`.
    pub async fn run(
        &self,
        push_tx: mpsc::Sender<(Frame, FrameDirection)>,
        outgoing_tx: mpsc::Sender<ChannelMessage>,
    ) {
        let mut incoming_rx = self
            .incoming_rx
            .lock()
            .unwrap()
            .take()
            .expect("run called more than once on the same ChannelTransport");

        let mut audio_out_rx = self
            .audio_out_rx
            .lock()
            .unwrap()
            .take()
            .expect("run called more than once on the same ChannelTransport");

        let base = self.base.clone();
        let sample_rate = self.sample_rate;
        let channels = self.channels;

        loop {
            tokio::select! {
                // ----------------------------------------------------------------
                // Arm 1: incoming messages → pipeline
                // ----------------------------------------------------------------
                msg = incoming_rx.recv() => {
                    match msg {
                        Some(ChannelMessage::Audio(bytes)) => {
                            let data = AudioRawData::new(bytes, sample_rate, channels);
                            base.push_audio_frame(data).await;
                        }

                        Some(ChannelMessage::Text(text)) => {
                            dispatch_text_message(&text, &push_tx).await;
                        }

                        Some(ChannelMessage::Interruption) => {
                            let _ = push_tx
                                .send((Frame::interruption(), FrameDirection::Downstream))
                                .await;
                        }

                        Some(ChannelMessage::ClientVadStart(ts)) => {
                            let _ = push_tx
                                .send((Frame::client_vad_user_started_speaking(ts), FrameDirection::Downstream))
                                .await;
                        }

                        Some(ChannelMessage::ClientVadStop(ts)) => {
                            let _ = push_tx
                                .send((Frame::client_vad_user_stopped_speaking(ts), FrameDirection::Downstream))
                                .await;
                        }

                        None => {
                            log::debug!("ChannelTransport: incoming channel closed");
                            break;
                        }
                    }
                }

                // ----------------------------------------------------------------
                // Arm 2: outgoing pipeline messages → outside world
                // ----------------------------------------------------------------
                output_msg = audio_out_rx.recv() => {
                    match output_msg {
                        Some(OutputMessage::Audio(bytes)) => {
                            if outgoing_tx.send(ChannelMessage::Audio(bytes)).await.is_err() {
                                log::warn!("ChannelTransport: outgoing channel closed");
                                break;
                            }
                        }

                        Some(OutputMessage::Text(json)) => {
                            if outgoing_tx.send(ChannelMessage::Text(json)).await.is_err() {
                                log::warn!("ChannelTransport: outgoing channel closed");
                                break;
                            }
                        }

                        Some(OutputMessage::Interruption) => {
                            // Drain stale audio chunks queued before the marker.
                            while let Ok(queued) = audio_out_rx.try_recv() {
                                match queued {
                                    OutputMessage::Audio(_) => {}    // discard
                                    OutputMessage::Interruption => break,
                                    OutputMessage::Text(_) => {}     // discard
                                }
                            }

                            if outgoing_tx.send(ChannelMessage::Interruption).await.is_err() {
                                log::warn!("ChannelTransport: outgoing channel closed");
                                break;
                            }
                            log::debug!("ChannelTransport: sent interruption");
                        }

                        None => {
                            log::debug!("ChannelTransport: audio out channel closed");
                            break;
                        }
                    }
                }
            }
        }

        let _ = push_tx
            .send((Frame::end(), FrameDirection::Downstream))
            .await;
    }
}

