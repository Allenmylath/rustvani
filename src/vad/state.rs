//! VAD state machine.
//!
//! `VadState` and `StateMachine` are a direct port of the state transitions
//! in Python's `VADAnalyzer._run_analyzer()`. No logic changed, only language.

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
// Audio helpers — ported from Python's audio_utils.py
// ---------------------------------------------------------------------------

/// Calculate RMS volume of raw 16-bit PCM audio, normalised to 0–1.
///
/// Equivalent to Python's `calculate_audio_volume()`.
pub fn calculate_audio_volume(audio: &[u8]) -> f32 {
    let samples: Vec<i16> = audio
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();

    if samples.is_empty() {
        return 0.0;
    }

    let sum_sq: f64 = samples.iter().map(|&s| (s as f64).powi(2)).sum();
    let rms = (sum_sq / samples.len() as f64).sqrt() as f32;
    rms / 32768.0
}

/// Exponential smoothing.
///
/// Equivalent to Python's `exp_smoothing(value, prev, factor)`.
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
///
/// Call `process_chunk()` with each raw PCM chunk; it returns the new state.
pub struct StateMachine {
    params: VadParams,

    /// Number of PCM frames required per inference call.
    /// 512 @ 16 kHz, 256 @ 8 kHz.
    frames_required: usize,

    /// Byte length of one inference window.
    /// `frames_required * num_channels * 2`
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

    // Accumulation buffer — collects bytes until a full window is ready.
    buffer: Vec<u8>,

    pub state: VadState,
}

impl StateMachine {
    /// Create a new state machine for the given sample rate and params.
    ///
    /// `sample_rate` must be 8000 or 16000.
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

    /// Feed a PCM chunk and a confidence score (from the Silero model).
    ///
    /// Returns `(new_state, Option<window_bytes>)`.
    /// `window_bytes` is `Some` when a full inference window is available
    /// and should be passed to `SileroVad::infer()`.
    /// When `None`, the chunk was buffered and no inference is needed yet.
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
        // Volume check — same as Python
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