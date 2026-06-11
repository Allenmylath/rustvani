# Speech Enhancement (Audio Front-End)

**File:** `src/audio_process/`
**Feature:** always available (no feature flag required)

Every audio frame is cleaned in-process before it reaches the STT provider. The chain is pure Rust, on by default, and adds zero latency beyond RNNoise's 10 ms frame buffer.

```
raw mic audio
  → High-pass filter   (DC offset, rumble, handling noise < 90 Hz)
  → RNNoise            (neural noise suppression, auto-resampled 16k ↔ 48k)
  → AGC                (normalize speech to −20 dBFS RMS)
  → Soft limiter       (no hard clipping, ever)
  → STT
```

---

## Why each stage matters for STT

| Stage | Problem it fixes |
|---|---|
| High-pass filter | Cheap mics add DC bias; handling and wind add rumble below the speech band. Removing it also helps RNNoise. |
| RNNoise | Background noise (streets, fans, crowds) that causes mistranscription and hallucinated tokens. |
| AGC | Quiet speakers get mistranscribed; loud speakers clip. Both are normalized to a level STT models like. |
| Soft limiter | Hard digital clipping (flat-topped waveforms) hurts STT more than mild compression. The limiter makes it impossible by construction. |

---

## Zero-config usage (Sarvam STT)

Both stages are on by default — there is nothing to wire up:

```rust
let stt = SarvamSttHandler::new(SarvamSttConfig {
    api_key: std::env::var("SARVAM_API_KEY").unwrap(),
    noise_reduction: true,   // RNNoise             (default: true)
    agc:             true,   // HPF + AGC + limiter (default: true)
    ..Default::default()
}).into_processor();
```

Inside the handler the order is: `pre_filter` (high-pass) → `RNNoiseFilter` → `post_filter` (AGC + limiter) → turn gate → Sarvam. The denoiser tail flushed on `VADUserStoppedSpeaking` also passes through the limiter.

---

## Standalone usage

### RNNoiseFilter

```rust
use rustvani::audio_process::noisefilter::RNNoiseFilter;

let mut nf = RNNoiseFilter::new(16_000);  // any rate; auto-resamples to 48 kHz
let clean  = nf.filter(&noisy_pcm_i16);   // may return fewer samples (10 ms framing)
let tail   = nf.flush();                  // drain at end of utterance
nf.reset();                               // clean slate for next utterance
```

### AudioEnhancer (HPF + AGC + limiter)

```rust
use rustvani::audio_process::agc::AudioEnhancer;

let mut enh = AudioEnhancer::new(16_000);
let pcm = enh.pre_filter(&raw_pcm);    // high-pass — run BEFORE the denoiser
let out = enh.post_filter(&denoised);  // AGC + limiter — run AFTER the denoiser
enh.reset();                           // clears filter state; keeps adapted gain
let db = enh.gain_db();                // current AGC gain, for diagnostics
```

Both `pre_filter` and `post_filter` are zero-latency: output length always equals input length. Like the noise filter, the chain operates in i16-range floats (−32 768 … 32 767).

**Ordering rule:** high-pass goes *before* the denoiser (cleans its input); AGC goes *after* (so silence and noise are not amplified before suppression).

---

## Tuning (`AgcConfig`)

```rust
use rustvani::audio_process::agc::{AudioEnhancer, AgcConfig};

let enh = AudioEnhancer::with_config(16_000, AgcConfig {
    target_rms: 4_000.0,   // hotter output level
    max_gain:   15.8,      // cap boost at +24 dB
    ..Default::default()
});
```

| Field | Default | Meaning |
|---|---|---|
| `highpass_hz` | `90.0` | High-pass cutoff (Hz). Speech fundamentals start ~85 Hz. |
| `target_rms` | `3277.0` | Target speech RMS in i16 range (≈ −20 dBFS). |
| `noise_gate_rms` | `165.0` | RMS below this (≈ −46 dBFS) is silence — gain adaptation holds, so the noise floor is never pumped up between words. |
| `max_gain` | `31.6` | Maximum boost, linear (+30 dB). |
| `min_gain` | `0.125` | Maximum cut, linear (−18 dB). |
| `attack_ms` | `10.0` | How fast gain *drops* when input gets loud. |
| `release_ms` | `400.0` | How fast gain *rises* for quiet speakers. Slow, to avoid breathing artefacts. |
| `limiter_knee` | `22937.0` | Limiting starts here (≈ −3 dBFS); peaks compress smoothly toward full scale via tanh. |

---

## Behavior notes

- **Gain memory across utterances.** `reset()` clears biquad filter state but intentionally keeps the adapted AGC gain — the same speaker is likely to continue at the same level, so re-learning from 0 dB every utterance would clip or duck the first words. In the Sarvam handler, `reset()` is called automatically after each transcript.
- **Attack vs release asymmetry.** Gain reductions track fast (10 ms — catches sudden shouts before they clip); gain increases track slow (400 ms — a cough or a pause doesn't make the floor breathe).
- **Silence handling.** Chunks below `noise_gate_rms` hold the current gain instead of adapting toward `max_gain`. Without this, every inter-word gap would crank the gain up and amplify the noise floor.
- **The limiter is not a clamp.** Below the knee the signal is untouched; above it, peaks asymptotically approach full scale — the output can never produce the flat-topped clipping signature that degrades STT.

---

## Resampling

`StreamResampler` (in `src/audio_process/resamplers/`) provides streaming sample-rate conversion via `rubato`. The noise filter uses it transparently when the source rate ≠ 48 kHz; it is also usable standalone for any rate pair.
