use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use log;
use tokio::sync::mpsc;

use crate::error::Result;
use crate::frames::{
    DataFrame, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor, SystemFrame,
};

use super::params::TransportParams;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

struct OutputTransportState {
    #[allow(dead_code)]
    params: TransportParams,

    /// True while OutputAudioRaw frames are being sent.
    bot_speaking: AtomicBool,

    /// Set by the concrete transport after construction.
    audio_out_tx: Mutex<Option<mpsc::Sender<Vec<u8>>>>,

    /// Mute gate — set true on InterruptionFrame so run_socket drops
    /// any stale audio bytes already sitting in the channel.
    /// Cleared automatically on the first OutputAudioRaw of the next
    /// bot turn so audio flows again without any manual reset.
    muted: Arc<AtomicBool>,

    /// Accumulates incoming TTS audio bytes between chunk boundaries.
    /// Drained in `chunk_size`-byte pieces so each mpsc send is exactly
    /// one 10 ms frame.  Remainder carries over to the next TTS blob.
    audio_buffer: Mutex<Vec<u8>>,

    /// Bytes per 10 ms output chunk.
    /// Computed dynamically from the first OutputAudioRaw frame:
    ///   chunk_size = (sample_rate / 100) × channels × 2   (16-bit PCM)
    /// Stored so we can log it once and clear the buffer on interruption.
    chunk_size: AtomicU32,
}

// ---------------------------------------------------------------------------
// BaseOutputTransport
// ---------------------------------------------------------------------------

pub struct BaseOutputTransport {
    state: Arc<OutputTransportState>,
}

impl BaseOutputTransport {
    pub fn new(params: TransportParams) -> Self {
        Self {
            state: Arc::new(OutputTransportState {
                params,
                bot_speaking: AtomicBool::new(false),
                audio_out_tx: Mutex::new(None),
                muted: Arc::new(AtomicBool::new(false)),
                audio_buffer: Mutex::new(Vec::with_capacity(8192)),
                chunk_size: AtomicU32::new(0),
            }),
        }
    }

    /// Wire up the audio output channel.
    /// Called once by the concrete transport before the pipeline starts.
    pub fn set_audio_out_tx(&self, tx: mpsc::Sender<Vec<u8>>) {
        *self.state.audio_out_tx.lock().unwrap() = Some(tx);
    }

    pub fn is_bot_speaking(&self) -> bool {
        self.state.bot_speaking.load(Ordering::Relaxed)
    }

    /// Clone the mute gate Arc so run_socket can share it.
    /// When this returns true, run_socket should silently drop audio chunks.
    pub fn mute_gate(&self) -> Arc<AtomicBool> {
        self.state.muted.clone()
    }
}

// ---------------------------------------------------------------------------
// FrameHandler impl
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameHandler for BaseOutputTransport {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match &frame.inner {
            // ---- Audio output: chunk into 10 ms pieces ----
            FrameInner::Data(DataFrame::OutputAudioRaw(audio)) => {
                // New bot audio arriving — clear the mute gate so run_socket
                // starts sending again.
                self.state.muted.store(false, Ordering::Relaxed);

                // Compute chunk size from this frame's sample rate.
                // Formula: (sample_rate / 100) × channels × 2  (16-bit PCM)
                // At 24 kHz mono → 240 samples × 2 bytes = 480 bytes per 10 ms
                // At 22050 Hz mono → 220 samples × 2 bytes = 440 bytes per 10 ms
                let channels = audio.num_channels.max(1) as u32;
                let new_chunk_size = (audio.sample_rate / 100) * channels * 2;

                if new_chunk_size == 0 {
                    log::warn!(
                        "BaseOutputTransport: invalid sample_rate={} — skipping frame",
                        audio.sample_rate
                    );
                    processor.push_frame(frame, direction).await?;
                    return Ok(());
                }

                // Log once when chunk size is first set or changes.
                let prev = self.state.chunk_size.swap(new_chunk_size, Ordering::Relaxed);
                if prev != new_chunk_size {
                    log::info!(
                        "BaseOutputTransport: 10ms chunk_size={}B (sr={}, ch={})",
                        new_chunk_size,
                        audio.sample_rate,
                        channels,
                    );
                }

                let chunk_size = new_chunk_size as usize;

                // Signal bot started speaking.
                if !self.state.bot_speaking.swap(true, Ordering::Relaxed) {
                    log::debug!("BaseOutputTransport: bot started speaking");
                    processor.broadcast_frame(Frame::bot_started_speaking()).await?;
                }

                // Append to buffer and extract 10 ms chunks.
                let chunks: Vec<Vec<u8>> = {
                    let mut buf = self.state.audio_buffer.lock().unwrap();
                    buf.extend_from_slice(&audio.audio);

                    let mut out = Vec::with_capacity(buf.len() / chunk_size + 1);
                    while buf.len() >= chunk_size {
                        out.push(buf.drain(..chunk_size).collect());
                    }
                    out
                    // Lock released here — held only for memory ops.
                };

                // Send each chunk through the channel.
                // Check the mute gate between sends so we bail fast if an
                // interruption was set concurrently.
                let tx = self.state.audio_out_tx.lock().unwrap().clone();
                if let Some(tx) = tx {
                    for chunk in chunks {
                        if self.state.muted.load(Ordering::Relaxed) {
                            self.state.audio_buffer.lock().unwrap().clear();
                            break;
                        }
                        if tx.try_send(chunk).is_err() {
                            log::warn!("BaseOutputTransport: audio_out channel full — dropping chunk");
                        }
                    }
                }

                processor.push_frame(frame, direction).await?;
            }

            // ---- Interruption: mute immediately + clear buffer + stop bot ----
            FrameInner::System(SystemFrame::Interruption) => {
                // Set mute gate BEFORE anything else so run_socket drops
                // in-flight chunks as fast as possible.
                self.state.muted.store(true, Ordering::Relaxed);

                // Discard any partial audio sitting in the chunking buffer.
                self.state.audio_buffer.lock().unwrap().clear();

                if self.state.bot_speaking.swap(false, Ordering::Relaxed) {
                    log::debug!("BaseOutputTransport: bot stopped speaking (interruption)");
                    processor.broadcast_frame(Frame::bot_stopped_speaking()).await?;
                }
                processor.push_frame(frame, direction).await?;
            }

            // ---- End / Cancel: clear speaking state + buffer ----
            FrameInner::Control(_) | FrameInner::System(SystemFrame::Cancel { .. }) => {
                self.state.audio_buffer.lock().unwrap().clear();

                if self.state.bot_speaking.swap(false, Ordering::Relaxed) {
                    log::debug!("BaseOutputTransport: bot stopped speaking (end/cancel)");
                    processor
                        .push_frame(Frame::bot_stopped_speaking(), FrameDirection::Upstream)
                        .await?;
                }
                processor.push_frame(frame, direction).await?;
            }

            // ---- Everything else: pass through ----
            _ => {
                processor.push_frame(frame, direction).await?;
            }
        }

        Ok(())
    }
}