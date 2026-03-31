//! VAD unit tests.
//!
//! Tests the state machine, audio helpers and params without
//! requiring the Silero model file or tract compilation.

use rustvani::vad::{
    VadParams, VadState,
    VAD_CONFIDENCE, VAD_MIN_VOLUME, VAD_START_SECS, VAD_STOP_SECS,
};
use rustvani::vad::state::{
    StateMachine, calculate_audio_volume, exp_smoothing,
};

// ---------------------------------------------------------------------------
// VadParams defaults match Python exactly
// ---------------------------------------------------------------------------

#[test]
fn test_vad_params_defaults() {
    let p = VadParams::default();
    assert_eq!(p.confidence, VAD_CONFIDENCE, "confidence should be 0.7");
    assert_eq!(p.start_secs, VAD_START_SECS, "start_secs should be 0.2");
    assert_eq!(p.stop_secs,  VAD_STOP_SECS,  "stop_secs should be 0.2");
    assert_eq!(p.min_volume, VAD_MIN_VOLUME,  "min_volume should be 0.6");
}

#[test]
fn test_vad_constants_values() {
    assert_eq!(VAD_CONFIDENCE, 0.7);
    assert_eq!(VAD_START_SECS, 0.2);
    assert_eq!(VAD_STOP_SECS,  0.2);
    assert_eq!(VAD_MIN_VOLUME, 0.6);
}

// ---------------------------------------------------------------------------
// Audio helpers
// ---------------------------------------------------------------------------

#[test]
fn test_calculate_audio_volume_silence() {
    // All zeros = silence = volume 0.0
    let silence = vec![0u8; 1024];
    let vol = calculate_audio_volume(&silence);
    assert_eq!(vol, 0.0, "silence should give volume 0.0");
}

#[test]
fn test_calculate_audio_volume_max() {
    // i16::MAX = 32767 → after normalisation ≈ 1.0
    let sample: i16 = i16::MAX;
    let bytes = sample.to_le_bytes();
    // 512 identical samples of max amplitude
    let audio: Vec<u8> = bytes.iter().cycle().take(1024).cloned().collect();
    let vol = calculate_audio_volume(&audio);
    assert!(
        (vol - 1.0).abs() < 0.001,
        "max amplitude should give volume ≈ 1.0, got {}",
        vol
    );
}

#[test]
fn test_calculate_audio_volume_nonzero() {
    // A non-trivial signal should produce a volume between 0 and 1
    let sample: i16 = 16384; // half max
    let bytes = sample.to_le_bytes();
    let audio: Vec<u8> = bytes.iter().cycle().take(1024).cloned().collect();
    let vol = calculate_audio_volume(&audio);
    assert!(vol > 0.0 && vol < 1.0, "half amplitude volume should be in (0, 1), got {}", vol);
}

#[test]
fn test_exp_smoothing_full_weight_on_current() {
    // factor=1.0 → result == current entirely
    let result = exp_smoothing(0.8, 0.2, 1.0);
    assert!((result - 0.8).abs() < 1e-6);
}

#[test]
fn test_exp_smoothing_full_weight_on_prev() {
    // factor=0.0 → result == prev entirely
    let result = exp_smoothing(0.8, 0.2, 0.0);
    assert!((result - 0.2).abs() < 1e-6);
}

#[test]
fn test_exp_smoothing_half() {
    // factor=0.5 → average of both
    let result = exp_smoothing(1.0, 0.0, 0.5);
    assert!((result - 0.5).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// StateMachine — state transitions
//
// We drive the machine directly with confidence + high-volume audio
// so we can test transitions without the Silero model.
// ---------------------------------------------------------------------------

/// Build a silent PCM window at the right byte length for 16kHz.
/// Used to supply the volume calculation inside `advance()`.
fn loud_window_16k() -> Vec<u8> {
    // i16 value that gives volume > VAD_MIN_VOLUME (0.6)
    // 32767 * 0.7 ≈ 22937 → well above threshold
    let sample: i16 = 22937;
    let bytes = sample.to_le_bytes();
    // 512 frames * 2 bytes = 1024 bytes for 16kHz mono
    bytes.iter().cycle().take(1024).cloned().collect()
}

fn silent_window_16k() -> Vec<u8> {
    // Near-zero samples → volume ≈ 0 → below min_volume
    vec![1u8, 0u8].iter().cycle().take(1024).cloned().collect()
}

fn make_machine() -> StateMachine {
    StateMachine::new(16_000, VadParams::default())
}

#[test]
fn test_initial_state_is_quiet() {
    let m = make_machine();
    assert_eq!(m.state, VadState::Quiet);
}

#[test]
fn test_quiet_to_starting_on_speech() {
    let mut m = make_machine();
    let window = loud_window_16k();
    // High confidence + loud audio → QUIET → STARTING
    let state = m.advance(0.9, &window);
    assert_eq!(state, VadState::Starting, "should move to Starting on first speech");
}

#[test]
fn test_starting_to_speaking_after_start_frames() {
    let mut m = make_machine();
    let window = loud_window_16k();

    // At 16kHz: frames_per_sec = 512/16000 = 0.032s
    // start_frames = round(0.2 / 0.032) = round(6.25) = 6
    // So we need 6 consecutive advances with high confidence
    let mut final_state = VadState::Quiet;
    for _ in 0..10 {
        final_state = m.advance(0.9, &window);
    }
    assert_eq!(
        final_state,
        VadState::Speaking,
        "should reach Speaking after enough start frames"
    );
}

#[test]
fn test_starting_falls_back_to_quiet_on_silence() {
    let mut m = make_machine();
    let loud   = loud_window_16k();
    let silent = silent_window_16k();

    // One speech frame → STARTING
    m.advance(0.9, &loud);
    assert_eq!(m.state, VadState::Starting);

    // One silence frame → back to QUIET
    let state = m.advance(0.0, &silent);
    assert_eq!(state, VadState::Quiet, "Starting + silence should revert to Quiet");
}

#[test]
fn test_speaking_to_stopping_on_silence() {
    let mut m = make_machine();
    let loud   = loud_window_16k();
    let silent = silent_window_16k();

    // Get to Speaking
    for _ in 0..10 {
        m.advance(0.9, &loud);
    }
    assert_eq!(m.state, VadState::Speaking);

    // First silence frame → STOPPING
    let state = m.advance(0.0, &silent);
    assert_eq!(state, VadState::Stopping, "Speaking + silence should move to Stopping");
}

#[test]
fn test_stopping_to_quiet_after_stop_frames() {
    let mut m = make_machine();
    let loud   = loud_window_16k();
    let silent = silent_window_16k();

    // Get to Speaking
    for _ in 0..10 {
        m.advance(0.9, &loud);
    }

    // Hold silence until QUIET
    let mut final_state = VadState::Speaking;
    for _ in 0..10 {
        final_state = m.advance(0.0, &silent);
    }
    assert_eq!(
        final_state,
        VadState::Quiet,
        "should reach Quiet after enough stop frames"
    );
}

#[test]
fn test_stopping_recovers_to_speaking_on_speech() {
    let mut m = make_machine();
    let loud   = loud_window_16k();
    let silent = silent_window_16k();

    // Get to Speaking
    for _ in 0..10 {
        m.advance(0.9, &loud);
    }

    // One silence → STOPPING
    m.advance(0.0, &silent);
    assert_eq!(m.state, VadState::Stopping);

    // Speech again → back to SPEAKING directly
    let state = m.advance(0.9, &loud);
    assert_eq!(
        state,
        VadState::Speaking,
        "Stopping + speech should recover to Speaking"
    );
}

// ---------------------------------------------------------------------------
// StateMachine buffer accumulation
// ---------------------------------------------------------------------------

#[test]
fn test_next_window_accumulates_until_full() {
    let mut m = make_machine();

    // 16kHz needs 1024 bytes (512 frames * 2 bytes)
    // Feed 512 bytes — not enough yet
    let half = vec![0u8; 512];
    let result = m.next_window(&half);
    assert!(result.is_none(), "half window should not trigger inference");

    // Feed another 512 bytes — now we have a full window
    let result = m.next_window(&half);
    assert!(result.is_some(), "full window should be returned");
    assert_eq!(result.unwrap().len(), 1024);
}

#[test]
fn test_next_window_returns_exact_window_size() {
    let mut m = make_machine();
    let full = vec![0u8; 1024];
    let result = m.next_window(&full);
    assert!(result.is_some());
    assert_eq!(result.unwrap().len(), 1024, "window should be exactly 1024 bytes at 16kHz");
}