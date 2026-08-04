# Speech Enhancement (Audio Front-End)

**File:** `src/audio_process/`
**Feature:** always available (no feature flag required)

Every audio frame is cleaned in-process before it reaches the STT provider. The chain is pure Rust, on by default, and adds roughly one denoiser frame (~10 ms) of latency.

```
raw mic audio
  → High-pass filter        (DC offset, rumble, handling noise < 90 Hz)
  → RNNoise or hush-vani    (neural noise suppression, resampled transparently)
  → AGC                     (normalize speech to −20 dBFS RMS)
  → Soft limiter            (no hard clipping, ever)
  → STT
```

---

## Why each stage matters for STT

| Stage | Problem it fixes |
|---|---|
| High-pass filter | Cheap mics add DC bias; handling and wind add rumble below the speech band. Removing it also helps the denoiser. |
| Denoiser | Background noise (streets, fans, crowds) that causes mistranscription and hallucinated tokens. |
| AGC | Quiet speakers get mistranscribed; loud speakers clip. Both are normalized to a level STT models like. |
| Soft limiter | Hard digital clipping (flat-topped waveforms) hurts STT more than mild compression. The limiter makes it impossible by construction. |

---

## Zero-config usage (Sarvam STT)

Both stages are on by default — there is nothing to wire up:

```rust
use rustvani::{NoiseBackend, SarvamSttConfig, SarvamSttHandler};

let stt = SarvamSttHandler::new(SarvamSttConfig {
    api_key: std::env::var("SARVAM_API_KEY").unwrap(),
    noise_reduction: true,                    // denoiser on         (default: true)
    noise_backend:   NoiseBackend::Rnnoise,   // which one           (default: Rnnoise)
    agc:             true,                    // HPF + AGC + limiter (default: true)
    ..SarvamSttConfig::default()
})
.into_processor();
```

Inside the handler the order is: `pre_filter` (high-pass) → denoiser → `post_filter` (AGC + limiter) → turn gate → Sarvam. The denoiser tail flushed on `VADUserStoppedSpeaking` also passes through the limiter.

---

## Choosing a noise backend

Both backends implement the `StreamingDenoiser` trait, so the STT path selects one at runtime behind a single `Box<dyn>`:

```rust
pub trait StreamingDenoiser: Send {
    fn filter(&mut self, audio: &[i16]) -> Vec<i16>;
    fn flush(&mut self) -> Vec<i16>;
    fn reset(&mut self);
}
```

The contract is length-preserving *over an utterance*, not per call: any single `filter` may return fewer (or zero) samples because of internal framing, so call `flush()` at end-of-utterance to drain the tail and `reset()` before the next one.

| | `NoiseBackend::Rnnoise` *(default)* | `NoiseBackend::HushVani` |
|---|---|---|
| Crate | `nnnoiseless` | `hush-vani` |
| Model | RNNoise | DeepFilterNet3-style |
| Nature | True streaming, 10 ms frames | Batch API, wrapped in a sliding window |
| Suppression | Good | Stronger, especially on non-stationary noise |
| Native rate | 48 kHz (resampled in/out) | 16 kHz (resampled in/out) |
| Added latency | ~10 ms (one frame) | ~10 ms (160 samples held back per call) |
| Cost | Very cheap | ~100× real-time, plus context recompute |

Neither requires a feature flag — both crates are regular dependencies. Switching is one field.

### How hush-vani is made streaming

`hush-vani` only exposes a **batch** `Hush::enhance`, and its GRUs start from zero on every call. Feeding it the live 20 ms chunks directly would produce cold-start artefacts on every chunk. `HushVaniFilter` (`src/audio_process/hushfilter/`) works around this with a sliding window:

- All input for the current utterance is buffered at 16 kHz (hush-vani's native rate). Each `filter` call re-runs `enhance` over a window of up to **200 ms of prior audio** (`CONTEXT_SAMPLES = 20 × 160`) plus the newly-arrived samples.
- The context region re-primes the GRUs to roughly the state a true streaming decode would hold. That region of the output is then discarded — only the new samples are emitted.
- `enhance` returns output that **lags its input by 160 samples**. For an absolute input index `m` in a window starting at `win_start`, `clean(input[m]) == out[m - win_start + 160]`. The last frame of each call therefore has no lagged output yet, so it is held back and drained in `flush()` via zero padding.

Across a full utterance, total output length ≈ total input length. Recomputing the context every call is affordable because hush-vani runs at roughly 100× real-time.

Sample rates other than 16 kHz are resampled in and out transparently via `StreamResampler`; samples are normalised to `f32` in `[-1, 1]` for hush-vani and denormalised back to i16 on the way out.

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

### HushVaniFilter

```rust
use rustvani::audio_process::hushfilter::HushVaniFilter;

let mut hf = HushVaniFilter::new(16_000)?;  // Result — model init can fail
let clean  = hf.filter(&noisy_pcm_i16);     // holds back the last 160 samples
let tail   = hf.flush();                    // drain at end of utterance
hf.reset();                                 // discard buffered audio + window
```

Unlike `RNNoiseFilter::new`, this returns a `Result<_, hush_vani::Error>` because the model is constructed eagerly. Inside `SarvamSttHandler::new`, a failure here logs an error and **falls back to RNNoise** rather than failing the pipeline — so `NoiseBackend::HushVani` is always safe to set.

Both types implement `StreamingDenoiser` (`rustvani::audio_process::StreamingDenoiser`), so you can hold either behind `Box<dyn StreamingDenoiser>`.

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

`StreamResampler` (in `src/audio_process/resamplers/`) provides streaming sample-rate conversion via `rubato`. Both denoisers use it transparently when the source rate differs from their native rate — 48 kHz for RNNoise, 16 kHz for hush-vani. It is also usable standalone for any rate pair.
