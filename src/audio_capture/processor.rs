use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use chrono::Utc;
use uuid::Uuid;

use crate::error::Result;
use crate::frames::{
    ControlFrame, DataFrame, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor, SystemFrame,
};
use super::collector::AudioCaptureCollector;
use super::encoder::{downmix_to_mono, resample_pcm};
use super::segment::PendingAudioSegment;

/// All recorded audio is normalised to this rate (mono) before buffering, so
/// the user and bot tracks share one sample grid and can be byte-synchronised.
const TARGET_RATE: u32 = 16_000;
const BYTES_PER_SAMPLE: usize = 2; // 16-bit mono

/// Wall-clock gap (seconds) above which we inject silence into a track. Kept
/// safely above normal frame jitter so a momentary scheduling hiccup does not
/// fragment an utterance.
const SILENCE_GAP_THRESHOLD_SECS: f64 = 0.2;

// ---------------------------------------------------------------------------
// Internal state — two continuous, time-aligned tracks (pipecat-style)
// ---------------------------------------------------------------------------

struct State {
    /// User track and bot track, mono @ `TARGET_RATE`. Kept the same length as
    /// audio flows: whichever side is silent is padded so the two never drift.
    user_buf: Vec<u8>,
    bot_buf:  Vec<u8>,

    user_speaking: bool,
    bot_speaking:  bool,

    /// Monotonic time each buffer was last written — used to fill real-world
    /// silence (latency between turns, muted mic, slow tool calls).
    last_user_update: Option<Instant>,
    last_bot_update:  Option<Instant>,

    finalized: bool,
}

impl State {
    fn new() -> Self {
        Self {
            user_buf: Vec::new(),
            bot_buf:  Vec::new(),
            user_speaking: false,
            bot_speaking:  false,
            last_user_update: None,
            last_bot_update:  None,
            finalized: false,
        }
    }

    /// Append user audio, filling real silence gaps and keeping the bot track
    /// aligned (unless the bot is mid-utterance — a barge-in).
    fn push_user_audio(&mut self, resampled: &[u8], now: Instant) {
        fill_silence_gap(&mut self.user_buf, self.last_user_update, now, resampled.len());
        self.last_user_update = Some(now);
        if !self.bot_speaking {
            let target = self.user_buf.len();
            pad_silence_to(&mut self.bot_buf, target);
            self.last_bot_update = Some(now);
        }
        self.user_buf.extend_from_slice(resampled);
    }

    /// Append bot audio, mirror of `push_user_audio`.
    fn push_bot_audio(&mut self, resampled: &[u8], now: Instant) {
        fill_silence_gap(&mut self.bot_buf, self.last_bot_update, now, resampled.len());
        self.last_bot_update = Some(now);
        if !self.user_speaking {
            let target = self.bot_buf.len();
            pad_silence_to(&mut self.user_buf, target);
            self.last_user_update = Some(now);
        }
        self.bot_buf.extend_from_slice(resampled);
    }
}

// ---------------------------------------------------------------------------
// AudioCaptureProcessor
// ---------------------------------------------------------------------------

/// Records the whole session as two continuous, time-synchronised tracks (user
/// and bot) and emits them at session end for mixing into one recording.
///
/// Position this processor **after TTS and before the output transport**. There
/// it sees `InputAudioRaw` (user PCM) and `OutputAudioRaw` (bot PCM) flowing
/// downstream, plus the VAD / Bot speaking frames it uses to track who holds the
/// floor.
///
/// ## Why two synchronised tracks instead of per-turn segments
///
/// Each incoming audio frame is normalised to mono @ `TARGET_RATE` and appended
/// to its own track. Before appending, the *other* track is padded with silence
/// up to the current write position (unless that side is actively speaking — a
/// barge-in, where overlap is real). Real-world gaps (turn latency, idle time)
/// are filled by comparing wall-clock elapsed time against the audio actually
/// written. The result is two equal-length tracks on a shared timeline: when one
/// side speaks the other is silent, so the final mono mix never overlaps voices
/// except during genuine barge-in. (This mirrors pipecat's `AudioBufferProcessor`.)
///
/// The `turn_id` cells are still updated so transcript entries recorded by the
/// aggregators stay populated; the audio itself is now one mixed recording.
pub struct AudioCaptureProcessor {
    collector:           Arc<dyn AudioCaptureCollector>,
    active_user_turn_id: Arc<Mutex<Option<Uuid>>>,
    active_bot_turn_id:  Arc<Mutex<Option<Uuid>>>,
    state:               Mutex<State>,
}

impl AudioCaptureProcessor {
    pub fn new(
        collector: Arc<dyn AudioCaptureCollector>,
        active_user_turn_id: Arc<Mutex<Option<Uuid>>>,
        active_bot_turn_id:  Arc<Mutex<Option<Uuid>>>,
    ) -> FrameProcessor {
        FrameProcessor::new(
            "AudioCaptureProcessor",
            Box::new(Self {
                collector,
                active_user_turn_id,
                active_bot_turn_id,
                state: Mutex::new(State::new()),
            }),
            false,
        )
    }

    /// Emit the two finished tracks (aligned to equal length) at session end.
    fn finalize(&self) {
        let (user, bot) = {
            let mut s = self.state.lock().unwrap();
            if s.finalized {
                return;
            }
            s.finalized = true;
            let target = s.user_buf.len().max(s.bot_buf.len());
            pad_silence_to(&mut s.user_buf, target);
            pad_silence_to(&mut s.bot_buf, target);
            (std::mem::take(&mut s.user_buf), std::mem::take(&mut s.bot_buf))
        };

        // Same timestamp on both tracks — they describe the same timeline, so
        // storage overlays (mixes) them rather than placing them sequentially.
        let started_at = Utc::now();
        if user.iter().any(|&b| b != 0) {
            self.collector.record_segment(PendingAudioSegment {
                segment_id:   Uuid::new_v4(),
                turn_id:      None,
                speaker:      "user",
                pcm:          user,
                sample_rate:  TARGET_RATE,
                num_channels: 1,
                started_at,
                interrupted:  false,
            });
        }
        if bot.iter().any(|&b| b != 0) {
            self.collector.record_segment(PendingAudioSegment {
                segment_id:   Uuid::new_v4(),
                turn_id:      None,
                speaker:      "bot",
                pcm:          bot,
                sample_rate:  TARGET_RATE,
                num_channels: 1,
                started_at,
                interrupted:  false,
            });
        }
    }
}

#[async_trait]
impl FrameHandler for AudioCaptureProcessor {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match &frame.inner {
            // ---- Who holds the floor ----
            FrameInner::System(SystemFrame::VADUserStartedSpeaking { .. }) => {
                *self.active_user_turn_id.lock().unwrap() = Some(Uuid::new_v4());
                self.state.lock().unwrap().user_speaking = true;
                processor.push_frame(frame, direction).await?;
            }
            FrameInner::System(SystemFrame::VADUserStoppedSpeaking { .. }) => {
                self.state.lock().unwrap().user_speaking = false;
                processor.push_frame(frame, direction).await?;
            }
            FrameInner::System(SystemFrame::BotStartedSpeaking) => {
                *self.active_bot_turn_id.lock().unwrap() = Some(Uuid::new_v4());
                self.state.lock().unwrap().bot_speaking = true;
                processor.push_frame(frame, direction).await?;
            }
            FrameInner::System(SystemFrame::BotStoppedSpeaking) => {
                self.state.lock().unwrap().bot_speaking = false;
                processor.push_frame(frame, direction).await?;
            }

            // ---- User audio ----
            FrameInner::System(SystemFrame::InputAudioRaw(audio)) => {
                let resampled = to_mono_target(&audio.audio, audio.sample_rate, audio.num_channels);
                if !resampled.is_empty() {
                    self.state.lock().unwrap().push_user_audio(&resampled, Instant::now());
                }
                processor.push_frame(frame, direction).await?;
            }

            // ---- Bot audio ----
            FrameInner::Data(DataFrame::OutputAudioRaw(audio)) => {
                let resampled = to_mono_target(&audio.audio, audio.sample_rate, audio.num_channels);
                if !resampled.is_empty() {
                    self.state.lock().unwrap().push_bot_audio(&resampled, Instant::now());
                }
                processor.push_frame(frame, direction).await?;
            }

            // ---- Session end — emit the two tracks ----
            FrameInner::System(SystemFrame::Stop { .. } | SystemFrame::Cancel { .. }) => {
                self.finalize();
                processor.push_frame(frame, direction).await?;
            }
            FrameInner::Control(ControlFrame::End { .. }) => {
                self.finalize();
                processor.push_frame(frame, direction).await?;
            }

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

/// Normalise an incoming PCM frame to mono @ `TARGET_RATE`.
fn to_mono_target(pcm: &[u8], sample_rate: u32, num_channels: u16) -> Vec<u8> {
    if pcm.is_empty() || sample_rate == 0 {
        return Vec::new();
    }
    let mono = downmix_to_mono(pcm, num_channels);
    resample_pcm(&mono, sample_rate, TARGET_RATE)
}

/// Pad `buf` with silence until it reaches `target_len` bytes. No-op if already
/// at or beyond the target.
fn pad_silence_to(buf: &mut Vec<u8>, target_len: usize) {
    if buf.len() < target_len {
        buf.resize(target_len, 0);
    }
}

/// Insert silence into `buf` for any wall-clock gap since it was last written
/// that exceeds the duration of the incoming frame — covers turn latency, a
/// muted mic, or idle time between bot utterances so turns stay temporally
/// separated instead of butting together.
fn fill_silence_gap(buf: &mut Vec<u8>, last_update: Option<Instant>, now: Instant, frame_bytes: usize) {
    let Some(last) = last_update else { return };
    let elapsed = now.duration_since(last).as_secs_f64();
    let frame_duration = frame_bytes as f64 / (TARGET_RATE as f64 * BYTES_PER_SAMPLE as f64);
    let gap = elapsed - frame_duration;
    if gap > SILENCE_GAP_THRESHOLD_SECS {
        let mut silence_bytes = (gap * TARGET_RATE as f64 * BYTES_PER_SAMPLE as f64) as usize;
        silence_bytes -= silence_bytes % BYTES_PER_SAMPLE; // keep 16-bit alignment
        if silence_bytes > 0 {
            buf.extend(std::iter::repeat(0u8).take(silence_bytes));
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_silence_extends_to_target() {
        let mut b = vec![1u8, 2, 3, 4];
        pad_silence_to(&mut b, 8);
        assert_eq!(b, vec![1, 2, 3, 4, 0, 0, 0, 0]);
    }

    #[test]
    fn pad_silence_noop_when_already_long_enough() {
        let mut b = vec![1u8, 2, 3, 4];
        pad_silence_to(&mut b, 2);
        assert_eq!(b, vec![1, 2, 3, 4]);
    }

    #[test]
    fn no_gap_filled_on_first_write() {
        let mut b = Vec::new();
        fill_silence_gap(&mut b, None, Instant::now(), 320);
        assert!(b.is_empty(), "first write must not inject silence");
    }

    #[test]
    fn large_wall_clock_gap_injects_aligned_silence() {
        let mut b = Vec::new();
        let last = Instant::now();
        // Pretend ~1s elapsed for a 10ms (320-byte) frame → ~990ms of silence.
        let now = last + std::time::Duration::from_millis(1000);
        fill_silence_gap(&mut b, Some(last), now, 320);
        // ~0.99s * 16000 * 2 ≈ 31680 bytes, and 16-bit aligned.
        assert!(b.len() > 30_000 && b.len() < 32_000, "got {}", b.len());
        assert_eq!(b.len() % BYTES_PER_SAMPLE, 0);
        assert!(b.iter().all(|&x| x == 0));
    }

    #[test]
    fn small_jitter_does_not_inject_silence() {
        let mut b = Vec::new();
        let last = Instant::now();
        let now = last + std::time::Duration::from_millis(15); // ~10ms frame + jitter
        fill_silence_gap(&mut b, Some(last), now, 320);
        assert!(b.is_empty(), "sub-threshold jitter must not inject silence");
    }
}
