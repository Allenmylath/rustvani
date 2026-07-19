use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket};
use tokio::sync::mpsc;

use crate::frames::{AudioRawData, Frame, FrameDirection, FrameInner, FrameProcessor, SystemFrame};
use crate::serializers::{FrameSerializer, SerializedInput, SerializedOutput};
use crate::transport::{BaseTransport, TransportParams};
use crate::transport::incoming::dispatch_text_message;
use crate::transport::output::OutputMessage;

// ---------------------------------------------------------------------------
// WebSocketParams
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct WebSocketParams {
    pub transport: TransportParams,
}

impl Default for WebSocketParams {
    fn default() -> Self {
        Self {
            transport: TransportParams {
                audio_in_enabled:         true,
                audio_in_sample_rate:     Some(16_000),
                audio_in_channels:        1,
                audio_in_passthrough:     true,
                audio_in_stream_on_start: true,
                ..TransportParams::default()
            },
        }
    }
}

// ---------------------------------------------------------------------------
// WebSocketTransport
// ---------------------------------------------------------------------------

pub struct WebSocketTransport {
    base:         Arc<BaseTransport>,
    audio_out_rx: std::sync::Mutex<Option<mpsc::Receiver<OutputMessage>>>,
    /// Optional wire-protocol serializer (e.g. Twilio). When set, all socket
    /// I/O is routed through it instead of the raw-PCM path.
    serializer:   std::sync::Mutex<Option<Box<dyn FrameSerializer>>>,
    audio_in_sample_rate:  u32,
    audio_out_sample_rate: u32,
}

const AUDIO_OUT_CHANNEL_CAP: usize = 150;

impl WebSocketTransport {
    pub fn new(name: &str, params: WebSocketParams) -> Self {
        let audio_in_sample_rate = params.transport.audio_in_sample_rate.unwrap_or(16_000);
        let audio_out_sample_rate =
            params.transport.audio_out_sample_rate.unwrap_or(audio_in_sample_rate);

        let base = Arc::new(BaseTransport::new(name, params.transport));

        let (audio_out_tx, audio_out_rx) = mpsc::channel::<OutputMessage>(AUDIO_OUT_CHANNEL_CAP);
        base.set_audio_out_tx(audio_out_tx);

        Self {
            base,
            audio_out_rx: std::sync::Mutex::new(Some(audio_out_rx)),
            serializer: std::sync::Mutex::new(None),
            audio_in_sample_rate,
            audio_out_sample_rate,
        }
    }

    pub fn input(&self) -> FrameProcessor {
        self.base.input()
    }

    pub fn output(&self) -> FrameProcessor {
        self.base.output()
    }

    /// Install a wire-protocol serializer. Call before [`run_socket`](Self::run_socket).
    ///
    /// With a serializer set, incoming provider messages are decoded via
    /// [`FrameSerializer::deserialize`] and outgoing audio/interruptions are
    /// encoded via [`FrameSerializer::serialize`]; without one, the transport
    /// uses its raw 16 kHz-PCM path.
    pub fn set_serializer(&self, serializer: Box<dyn FrameSerializer>) {
        *self.serializer.lock().unwrap() = Some(serializer);
    }

    /// Drive the WebSocket connection until it closes.
    ///
    /// Arm 1 — `socket.recv()`: incoming messages from the client.
    ///
    ///   - With a serializer: every Text/Binary message is passed to
    ///     `deserialize`; audio frames go to the input transport, other frames
    ///     (e.g. DTMF) are pushed downstream.
    ///   - Without a serializer: Binary → raw 16 kHz PCM; Text → RAVI/control
    ///     dispatch.
    ///
    /// Arm 2 — `audio_out_rx.recv()`: outgoing pipeline messages.
    ///
    ///   - With a serializer: `Audio`/`Interruption` are re-framed and
    ///     `serialize`d (Twilio `media`/`clear` events).
    ///   - Without a serializer: `Audio` → binary frame, `Interruption` → JSON
    ///     clear marker.
    ///
    /// On close, a serializer's `serialize(EndFrame)` is invoked to drive
    /// provider-side teardown (e.g. Twilio auto hang-up).
    pub async fn run_socket(
        &self,
        mut socket: WebSocket,
        push_tx: mpsc::Sender<(Frame, FrameDirection)>,
    ) {
        let mut audio_out_rx = self
            .audio_out_rx
            .lock()
            .unwrap()
            .take()
            .expect("run_socket called more than once on the same WebSocketTransport");

        let mut serializer = self.serializer.lock().unwrap().take();

        let base = self.base.clone();

        if let Some(ser) = serializer.as_mut() {
            ser.setup(self.audio_in_sample_rate, self.audio_out_sample_rate).await;
        }

        loop {
            tokio::select! {
                // ----------------------------------------------------------------
                // Arm 1: incoming messages → pipeline
                // ----------------------------------------------------------------
                msg = socket.recv() => {
                    match msg {
                        Some(Ok(Message::Binary(bytes))) => {
                            if let Some(ser) = serializer.as_mut() {
                                let input = SerializedInput::Binary(bytes.to_vec());
                                if let Some(frame) = ser.deserialize(&input).await {
                                    Self::dispatch_incoming_frame(&base, &push_tx, frame).await;
                                }
                            } else {
                                let data = AudioRawData::new(bytes.to_vec(), 16_000, 1);
                                base.push_audio_frame(data).await;
                            }
                        }

                        Some(Ok(Message::Text(text))) => {
                            if let Some(ser) = serializer.as_mut() {
                                let input = SerializedInput::Text(text.to_string());
                                if let Some(frame) = ser.deserialize(&input).await {
                                    Self::dispatch_incoming_frame(&base, &push_tx, frame).await;
                                }
                            } else {
                                dispatch_text_message(&text, &push_tx).await;
                            }
                        }

                        Some(Ok(Message::Close(_))) | None => {
                            log::debug!("WebSocketTransport: client closed connection");
                            break;
                        }

                        Some(Ok(_)) => {} // ping / pong — ignore

                        Some(Err(e)) => {
                            log::warn!("WebSocketTransport: socket error: {}", e);
                            break;
                        }
                    }
                }

                // ----------------------------------------------------------------
                // Arm 2: outgoing pipeline messages → client
                // ----------------------------------------------------------------
                output_msg = audio_out_rx.recv() => {
                    match output_msg {
                        Some(OutputMessage::Audio(bytes)) => {
                            if let Some(ser) = serializer.as_mut() {
                                let frame =
                                    Frame::output_audio(bytes, self.audio_out_sample_rate, 1);
                                if let Some(out) = ser.serialize(&frame).await {
                                    if Self::send_serialized(&mut socket, out).await.is_err() {
                                        log::warn!("WebSocketTransport: failed to send audio");
                                        break;
                                    }
                                }
                            } else if socket.send(Message::Binary(bytes.into())).await.is_err() {
                                log::warn!("WebSocketTransport: failed to send audio");
                                break;
                            }
                        }

                        Some(OutputMessage::Text(json)) => {
                            // RAVI protocol messages (bot-ready, transcriptions, etc.)
                            if socket.send(Message::Text(json.into())).await.is_err() {
                                log::warn!("WebSocketTransport: failed to send RAVI text message");
                                break;
                            }
                        }

                        Some(OutputMessage::Interruption) => {
                            // Drain stale audio chunks queued before the marker.
                            while let Ok(queued) = audio_out_rx.try_recv() {
                                match queued {
                                    OutputMessage::Audio(_) => {}    // discard
                                    OutputMessage::Interruption => break,
                                    OutputMessage::Text(_) => {}     // discard (shouldn't queue during interruption)
                                }
                            }

                            let send_result = if let Some(ser) = serializer.as_mut() {
                                match ser.serialize(&Frame::interruption()).await {
                                    Some(out) => Self::send_serialized(&mut socket, out).await,
                                    None => Ok(()),
                                }
                            } else {
                                socket
                                    .send(Message::Text(r#"{"type":"interruption"}"#.into()))
                                    .await
                            };

                            if send_result.is_err() {
                                log::warn!("WebSocketTransport: failed to send interruption");
                                break;
                            }
                            log::debug!("WebSocketTransport: sent interruption to client");
                        }

                        None => break, // pipeline shut down
                    }
                }
            }
        }

        // Give the serializer a chance to tear down the provider session
        // (e.g. Twilio auto hang-up on EndFrame). Any serialized output here is
        // best-effort — the socket may already be closing.
        if let Some(ser) = serializer.as_mut() {
            if let Some(out) = ser.serialize(&Frame::end()).await {
                let _ = Self::send_serialized(&mut socket, out).await;
            }
        }

        let _ = push_tx
            .send((Frame::end(), FrameDirection::Downstream))
            .await;
    }

    /// Route a deserialized incoming frame: audio to the input transport,
    /// everything else (DTMF, control) downstream into the pipeline.
    async fn dispatch_incoming_frame(
        base: &BaseTransport,
        push_tx: &mpsc::Sender<(Frame, FrameDirection)>,
        frame: Frame,
    ) {
        match frame.inner {
            FrameInner::System(SystemFrame::InputAudioRaw(data)) => {
                base.push_audio_frame(data).await;
            }
            _ => {
                let _ = push_tx.send((frame, FrameDirection::Downstream)).await;
            }
        }
    }

    /// Send a serializer's output over the socket as the matching frame type.
    async fn send_serialized(
        socket: &mut WebSocket,
        out: SerializedOutput,
    ) -> Result<(), axum::Error> {
        match out {
            SerializedOutput::Text(t) => socket.send(Message::Text(t.into())).await,
            SerializedOutput::Binary(b) => socket.send(Message::Binary(b.into())).await,
        }
    }
}
