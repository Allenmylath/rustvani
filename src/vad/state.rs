//! VAD state machine.
//!
//! `VadState` and `StateMachine` are a direct port of the state transitions
//! in Python's `VADAnalyzer._run_analyzer()`. No logic changed, only language.
//!
//! # Volume calculation
//!
//! Python's `calculate_audio_volume` uses EBU R128 loudness (pyloudnorm),
//! normalised from -20..80 LUFS → 0.0..1.0. We approximate this with dBFS:
//!
//!   rms_linear = RMS(samples) / 32768          # 0.0–1.0
//!   db = 20 * log10(rms_linear).clamp(-60, 0)  # -60..0 dBFS
//!   volume = (db + 60) / 60                    # 0.0..1.0
//!
//! This maps silence (~-60 dBFS) → 0.0 and full-scale → 1.0.
//! Normal conversational speech (RMS ~500–3000 / 32768) maps to roughly
//! 0.3–0.6 on this scale, which is compatible with VAD_MIN_VOLUME = 0.6.

use super::params::VadParams;

// ---------------------------------------------------------------------------
// VadState
// ---------------------------------------------------------------------------

/// Voice Activity Detection states.
///
/// Mirrors Python's `VADState` enum exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadState {
    /// No voice activity detected.
    Quiet,
    /// Voice activity beginning — waiting to confirm speech start.
    Starting,
    /// Active voice confirmed.
    Speaking,
    /// Voice ending — waiting to confirm silence.
    Stopping,
}

impl Default for VadState {
    fn default() -> Self {
        Self::Quiet
    }
}

// ---------------------------------------------------------------------------
// Audio helpers
// ---------------------------------------------------------------------------

/// Calculate audio volume as a normalised 0.0–1.0 value using dBFS.
///
/// Approximates Python's EBU R128-based `calculate_audio_volume`:
///   - Silence (~-60 dBFS or below) → ~0.0
///   - Full-scale → 1.0
///   - Normal conversational speech → ~0.3–0.6
///
/// This makes `VAD_MIN_VOLUME = 0.6` semantically compatible with the
/// Python implementation without the cost of a full loudness meter.
pub fn calculate_audio_volume(audio: &[u8]) -> f32 {
    let samples: Vec<i16> = audio
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();

    if samples.is_empty() {
        return 0.0;
    }

    // Compute RMS normalised to 0.0–1.0
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64).powi(2)).sum();
    let rms = (sum_sq / samples.len() as f64).sqrt() as f32 / 32768.0;

    if rms < 1e-9 {
        return 0.0;
    }

    // Convert to dBFS, clamp to -60..0, then normalise to 0.0..1.0
    let db = (20.0 * rms.log10()).clamp(-60.0, 0.0);
    (db + 60.0) / 60.0
}

/// Exponential smoothing.
///
/// Equivalent to Python's `exp_smoothing(value, prev, factor)`:
///   return prev + factor * (value - prev)
/// which is algebraically identical to our original form.
#[inline]
pub fn exp_smoothing(current: f32, prev: f32, factor: f32) -> f32 {
    current * factor + prev * (1.0 - factor)
}

// ---------------------------------------------------------------------------
// StateMachine
// ---------------------------------------------------------------------------

const SMOOTHING_FACTOR: f32 = 0.2;

/// Stateful VAD state machine.
///
/// One instance per audio stream. Holds the accumulated audio buffer,
/// frame counts, volume history, and current state.
pub struct StateMachine {
    params: VadParams,

    /// Number of PCM frames required per inference call.
    /// 512 @ 16 kHz, 256 @ 8 kHz.
    frames_required: usize,

    /// Byte length of one inference window.
    bytes_required: usize,

    /// Frames needed to confirm STARTING → SPEAKING.
    start_frames: usize,

    /// Frames needed to confirm STOPPING → QUIET.
    stop_frames: usize,

    // Counters
    starting_count: usize,
    stopping_count: usize,

    // Volume smoothing
    prev_volume: f32,

    // Accumulation buffer
    buffer: Vec<u8>,

    pub state: VadState,
}

impl StateMachine {
    /// Create a new state machine for the given sample rate and params.
    pub fn new(sample_rate: u32, params: VadParams) -> Self {
        let frames_required: usize = if sample_rate == 16000 { 512 } else { 256 };
        let bytes_required = frames_required * 2; // mono, 16-bit

        let frames_per_sec = frames_required as f32 / sample_rate as f32;
        let start_frames = (params.start_secs / frames_per_sec).round() as usize;
        let stop_frames  = (params.stop_secs  / frames_per_sec).round() as usize;

        Self {
            params,
            frames_required,
            bytes_required,
            start_frames,
            stop_frames,
            starting_count: 0,
            stopping_count: 0,
            prev_volume: 0.0,
            buffer: Vec::with_capacity(bytes_required * 2),
            state: VadState::Quiet,
        }
    }

    /// Feed a PCM chunk into the buffer and take **one** inference window.
    ///
    /// Returns `Some(window)` when a full window is ready, `None` if more data
    /// is needed.
    ///
    /// # Callers must drain in a loop
    ///
    /// One call yields at most one window, so a caller that calls this once per
    /// audio frame keeps up only while frames are no wider than
    /// `frames_required` (512 samples at 16 kHz, 256 at 8 kHz). Feed a wider
    /// frame — an upsampling input path such as the Twilio serializer emits
    /// 960-sample frames at 16 kHz — and the excess stays buffered, growing
    /// without bound, so the VAD drifts ever further behind the live audio.
    ///
    /// Feed the chunk once, then keep pulling until `None`:
    ///
    /// ```ignore
    /// let mut pending: &[u8] = &frame.audio;
    /// while let Some(window) = { let w = machine.next_window(pending); pending = &[]; w } {
    ///     // ... analyse window ...
    /// }
    /// ```
    pub fn next_window(&mut self, chunk: &[u8]) -> Option<Vec<u8>> {
        self.buffer.extend_from_slice(chunk);
        if self.buffer.len() >= self.bytes_required {
            let window: Vec<u8> = self.buffer.drain(..self.bytes_required).collect();
            Some(window)
        } else {
            None
        }
    }

    /// Advance the state machine with a confidence value from the model.
    ///
    /// Mirrors Python's `_run_analyzer()` state transition logic exactly.
    /// Returns the new `VadState`.
    pub fn advance(&mut self, confidence: f32, audio_window: &[u8]) -> VadState {
        let volume = exp_smoothing(
            calculate_audio_volume(audio_window),
            self.prev_volume,
            SMOOTHING_FACTOR,
        );
        self.prev_volume = volume;

        let speaking = confidence >= self.params.confidence
            && volume >= self.params.min_volume;

        // ---- State transitions — exact Python logic ----
        if speaking {
            match self.state {
                VadState::Quiet => {
                    self.state = VadState::Starting;
                    self.starting_count = 1;
                }
                VadState::Starting => {
                    self.starting_count += 1;
                }
                VadState::Stopping => {
                    self.state = VadState::Speaking;
                    self.stopping_count = 0;
                }
                VadState::Speaking => {}
            }
        } else {
            match self.state {
                VadState::Starting => {
                    self.state = VadState::Quiet;
                    self.starting_count = 0;
                }
                VadState::Speaking => {
                    self.state = VadState::Stopping;
                    self.stopping_count = 1;
                }
                VadState::Stopping => {
                    self.stopping_count += 1;
                }
                VadState::Quiet => {}
            }
        }

        // ---- Threshold checks ----
        if self.state == VadState::Starting
            && self.starting_count >= self.start_frames
        {
            self.state = VadState::Speaking;
            self.starting_count = 0;
        }

        if self.state == VadState::Stopping
            && self.stopping_count >= self.stop_frames
        {
            self.state = VadState::Quiet;
            self.stopping_count = 0;
        }

        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Windows produced when a caller feeds `frame_samples` per frame and
    /// drains until `next_window` returns `None`.
    fn windows_when_drained(frame_samples: usize, n_frames: usize) -> usize {
        let mut m = StateMachine::new(16_000, VadParams::default());
        let chunk = vec![0u8; frame_samples * 2];
        let mut windows = 0;
        for _ in 0..n_frames {
            let mut pending: &[u8] = &chunk;
            while {
                let w = m.next_window(pending);
                pending = &[];
                w
            }
            .is_some()
            {
                windows += 1;
            }
        }
        windows
    }

    /// Windows produced by the old one-call-per-frame pattern.
    fn windows_when_called_once(frame_samples: usize, n_frames: usize) -> usize {
        let mut m = StateMachine::new(16_000, VadParams::default());
        let chunk = vec![0u8; frame_samples * 2];
        (0..n_frames).filter(|_| m.next_window(&chunk).is_some()).count()
    }

    /// A drain loop analyses everything it is fed, whatever the frame size.
    /// 960-sample frames are what the Twilio serializer emits at 16 kHz.
    #[test]
    fn drain_loop_keeps_up_with_wide_frames() {
        for frame in [160usize, 320, 512, 960, 4800] {
            let n = 200;
            let analysed = windows_when_drained(frame, n) * 512;
            let fed = frame * n;
            assert!(
                fed - analysed < 512,
                "frame={frame}: fed {fed} samples, analysed {analysed} — \
                 a drain loop must leave less than one window unconsumed"
            );
        }
    }

    /// The regression this guards: one call per frame silently drops 47% of a
    /// 960-sample frame, so the VAD can never catch up.
    #[test]
    fn single_call_per_frame_starves_on_wide_frames() {
        let analysed = windows_when_called_once(960, 200) * 512;
        let fed = 960 * 200;
        assert!(
            (analysed as f32) / (fed as f32) < 0.55,
            "expected the old pattern to fall behind, got {analysed}/{fed}"
        );
    }

    /// For frames at or under one window the drain loop changes nothing, so
    /// existing transports (browser 320, WebRTC 160) are unaffected.
    #[test]
    fn drain_loop_is_a_no_op_for_narrow_frames() {
        for frame in [160usize, 320, 512] {
            assert_eq!(
                windows_when_drained(frame, 200),
                windows_when_called_once(frame, 200),
                "frame={frame}: drain loop must be behaviour-identical here"
            );
        }
    }
}