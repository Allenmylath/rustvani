//! Pure Rust Silero VAD backend — zero external dependencies.
//!
//! Drop-in alternative to the ort-based `SileroVadOrt`.
//! Loads weights from a flat binary file (extracted from the ONNX model).
//!
//! Default path: `silero_vad_16k.bin` in the rustvani cache directory
//! (`~/.rustvani/cache/` on Unix, `%LOCALAPPDATA%\rustvani\cache` on Windows).
//!
//! Usage:
//! ```ignore
//! let vad = SileroVadNative::new(16000)?;
//! let confidence = vad.infer_async(audio_bytes).await?;
//! ```

use std::sync::{Arc, Mutex};

use super::analyzer::VadAnalyzer;

pub const DEFAULT_WEIGHTS_PATH: &str =
    concat!(env!("RUSTVANI_CACHE_DIR"), "/silero_vad_16k.bin");

// ─── Engine constants ─────────────────────────────────────────────────

const H: usize = 128;
const GATES: usize = 4 * H;
const CONTEXT_16K: usize = 64;
const COMBINED: usize = 2 * H;

// ─── Engine internals ─────────────────────────────────────────────────

struct InferState {
    h: Vec<f32>,
    c: Vec<f32>,
    context: Vec<f32>,
}

impl InferState {
    fn new(context_size: usize) -> Self {
        Self {
            h: vec![0.0; H],
            c: vec![0.0; H],
            context: vec![0.0; context_size],
        }
    }
}

struct Conv1dW {
    weight: Vec<f32>,
    bias: Vec<f32>,
    out_ch: usize,
    in_ch: usize,
}

struct Engine {
    stft_basis: Vec<f32>,
    enc: [Conv1dW; 4],
    lstm_w: Vec<f32>,
    lstm_bias: Vec<f32>,
    dec_weight: Vec<f32>,
    dec_bias: f32,
}

impl Engine {
    fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        if floats.len() < 309_633 {
            return Err(format!(
                "Weight file too small: {} floats, expected >= 309633",
                floats.len()
            ));
        }

        let mut o = 0usize;
        let mut take = |n: usize| -> Vec<f32> {
            let v = floats[o..o + n].to_vec();
            o += n;
            v
        };

        let stft_basis = take(258 * 256);
        let e0w = take(128 * 129 * 3); let e0b = take(128);
        let e1w = take(64 * 128 * 3);  let e1b = take(64);
        let e2w = take(64 * 64 * 3);   let e2b = take(64);
        let e3w = take(128 * 64 * 3);  let e3b = take(128);
        let wih = take(512 * 128);
        let whh = take(512 * 128);
        let bih = take(512);
        let bhh = take(512);
        let dw = take(128);
        let db = floats[o];

        let mut bias = vec![0.0f32; GATES];
        for i in 0..GATES { bias[i] = bih[i] + bhh[i]; }

        let mut lstm_w = vec![0.0f32; GATES * COMBINED];
        for g in 0..GATES {
            let dst = g * COMBINED;
            lstm_w[dst..dst + H].copy_from_slice(&wih[g * H..(g + 1) * H]);
            lstm_w[dst + H..dst + COMBINED].copy_from_slice(&whh[g * H..(g + 1) * H]);
        }

        Ok(Self {
            stft_basis,
            enc: [
                Conv1dW { weight: e0w, bias: e0b, out_ch: 128, in_ch: 129 },
                Conv1dW { weight: e1w, bias: e1b, out_ch: 64,  in_ch: 128 },
                Conv1dW { weight: e2w, bias: e2b, out_ch: 64,  in_ch: 64 },
                Conv1dW { weight: e3w, bias: e3b, out_ch: 128, in_ch: 64 },
            ],
            lstm_w,
            lstm_bias: bias,
            dec_weight: dw,
            dec_bias: db,
        })
    }

    /// Infer on 512 f32 samples. Updates state in-place.
    fn infer(&self, samples: &[f32], st: &mut InferState) -> f32 {
        debug_assert_eq!(samples.len(), 512);

        // Prepend context
        let mut input = Vec::with_capacity(CONTEXT_16K + 512);
        input.extend_from_slice(&st.context);
        input.extend_from_slice(samples);
        st.context.copy_from_slice(&input[input.len() - CONTEXT_16K..]);

        // STFT
        let padded = reflect_pad_right(&input, 64);
        let stft_len = (padded.len() - 256) / 128 + 1;
        let mag = self.stft_magnitude(&padded, stft_len);

        // Encoder
        let strides = [1usize, 2, 2, 1];
        let mut x = mag;
        let mut ch = 129usize;
        let mut len = stft_len;
        for (i, e) in self.enc.iter().enumerate() {
            let new_len = (len + 2 - 3) / strides[i] + 1;
            x = conv1d_k3_pad1_relu(&x, ch, len, e, strides[i], new_len);
            ch = e.out_ch;
            len = new_len;
        }

        // Decoder
        let mut prob_sum = 0.0f32;
        for t in 0..len {
            let mut frame = [0.0f32; H];
            for c in 0..ch { frame[c] = x[c * len + t]; }
            self.lstm_cell(&frame, &mut st.h, &mut st.c);
            let logit = self.dec_bias + dot_relu(&self.dec_weight, &st.h);
            prob_sum += sigmoid(logit);
        }
        prob_sum / len as f32
    }

    fn stft_magnitude(&self, padded: &[f32], out_len: usize) -> Vec<f32> {
        let mut mag = vec![0.0f32; 129 * out_len];
        for t in 0..out_len {
            let x_off = t * 128;
            let x_slice = &padded[x_off..x_off + 256];
            for f in 0..129 {
                let re = dot(&self.stft_basis[f * 256..(f + 1) * 256], x_slice);
                let im = dot(&self.stft_basis[(f + 129) * 256..(f + 130) * 256], x_slice);
                mag[f * out_len + t] = re.mul_add(re, im * im).sqrt();
            }
        }
        mag
    }

    fn lstm_cell(&self, input: &[f32], h: &mut Vec<f32>, c: &mut Vec<f32>) {
        let mut xh = [0.0f32; COMBINED];
        xh[..H].copy_from_slice(input);
        xh[H..].copy_from_slice(h);

        let mut gates = [0.0f32; GATES];
        let mut g = 0;
        while g + 4 <= GATES {
            let r0 = g * COMBINED;
            let r1 = (g + 1) * COMBINED;
            let r2 = (g + 2) * COMBINED;
            let r3 = (g + 3) * COMBINED;
            gates[g]     = dot(&self.lstm_w[r0..r0 + COMBINED], &xh) + self.lstm_bias[g];
            gates[g + 1] = dot(&self.lstm_w[r1..r1 + COMBINED], &xh) + self.lstm_bias[g + 1];
            gates[g + 2] = dot(&self.lstm_w[r2..r2 + COMBINED], &xh) + self.lstm_bias[g + 2];
            gates[g + 3] = dot(&self.lstm_w[r3..r3 + COMBINED], &xh) + self.lstm_bias[g + 3];
            g += 4;
        }

        for i in 0..H {
            let ig = sigmoid(gates[i]);
            let fg = sigmoid(gates[H + i]);
            let gg = gates[2 * H + i].tanh();
            let og = sigmoid(gates[3 * H + i]);
            c[i] = fg * c[i] + ig * gg;
            h[i] = og * c[i].tanh();
        }
    }
}

// ─── SIMD dot products ────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_avx2(a, b) };
        }
    }
    dot_scalar(a, b)
}

#[inline]
fn dot_relu(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_relu_avx2(a, b) };
        }
    }
    let mut s = 0.0f32;
    for i in 0..a.len() { s += a[i] * b[i].max(0.0); }
    s
}

fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let chunks = n / 4;
    for i in 0..chunks {
        let j = i * 4;
        unsafe {
            s0 += *a.get_unchecked(j)     * *b.get_unchecked(j);
            s1 += *a.get_unchecked(j + 1) * *b.get_unchecked(j + 1);
            s2 += *a.get_unchecked(j + 2) * *b.get_unchecked(j + 2);
            s3 += *a.get_unchecked(j + 3) * *b.get_unchecked(j + 3);
        }
    }
    for i in (chunks * 4)..n {
        unsafe { s0 += *a.get_unchecked(i) * *b.get_unchecked(i); }
    }
    s0 + s1 + s2 + s3
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let chunks32 = n / 32;
    for i in 0..chunks32 {
        let j = i * 32;
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(j)),      _mm256_loadu_ps(bp.add(j)),      acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(j + 8)),  _mm256_loadu_ps(bp.add(j + 8)),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(j + 16)), _mm256_loadu_ps(bp.add(j + 16)), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(j + 24)), _mm256_loadu_ps(bp.add(j + 24)), acc3);
    }
    let done = chunks32 * 32;
    let chunks8 = (n - done) / 8;
    for i in 0..chunks8 {
        let j = done + i * 8;
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(j)), _mm256_loadu_ps(bp.add(j)), acc0);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);
    let hi = _mm256_extractf128_ps::<1>(acc0);
    let lo = _mm256_castps256_ps128(acc0);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);
    let tail = done + chunks8 * 8;
    for i in tail..n { total += *ap.add(i) * *bp.add(i); }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_relu_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let zero = _mm256_setzero_ps();
    let mut acc = _mm256_setzero_ps();
    let chunks = n / 8;
    for i in 0..chunks {
        let j = i * 8;
        let vb = _mm256_max_ps(_mm256_loadu_ps(bp.add(j)), zero);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(j)), vb, acc);
    }
    let hi = _mm256_extractf128_ps::<1>(acc);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);
    for i in (chunks * 8)..n { total += *ap.add(i) * (*bp.add(i)).max(0.0); }
    total
}

// ─── Small ops ────────────────────────────────────────────────────────

#[inline(always)]
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

fn reflect_pad_right(input: &[f32], pad: usize) -> Vec<f32> {
    let n = input.len();
    let mut out = Vec::with_capacity(n + pad);
    out.extend_from_slice(input);
    for i in 0..pad { out.push(input[n - 2 - i]); }
    out
}

fn conv1d_k3_pad1_relu(
    input: &[f32], in_ch: usize, in_len: usize,
    e: &Conv1dW, stride: usize, ol: usize,
) -> Vec<f32> {
    let pl = in_len + 2;
    let mut padded = vec![0.0f32; in_ch * pl];
    for ci in 0..in_ch {
        let src = ci * in_len;
        let dst = ci * pl + 1;
        padded[dst..dst + in_len].copy_from_slice(&input[src..src + in_len]);
    }
    let out_ch = e.out_ch;
    let mut output = vec![0.0f32; out_ch * ol];
    for co in 0..out_ch {
        let b = e.bias[co];
        for t in 0..ol {
            let ps = t * stride;
            let mut sum = b;
            for ci in 0..in_ch {
                let wb = (co * in_ch + ci) * 3;
                let xb = ci * pl + ps;
                unsafe {
                    sum += *e.weight.get_unchecked(wb)     * *padded.get_unchecked(xb);
                    sum += *e.weight.get_unchecked(wb + 1) * *padded.get_unchecked(xb + 1);
                    sum += *e.weight.get_unchecked(wb + 2) * *padded.get_unchecked(xb + 2);
                }
            }
            output[co * ol + t] = sum.max(0.0);
        }
    }
    output
}

// ─── Public API ───────────────────────────────────────────────────────

struct SileroNativeInner {
    engine: Engine,
    state: InferState,
    num_samples: usize,
}

/// Pure Rust Silero VAD — no onnxruntime dependency.
///
/// Currently supports 16kHz only. For 8kHz, use `SileroVadOrt`.
#[derive(Clone)]
pub struct SileroVadNative {
    inner: Arc<Mutex<SileroNativeInner>>,
}

impl SileroVadNative {
    /// Load from the default weights path (rustvani cache directory).
    pub fn new(sample_rate: u32) -> Result<Self, String> {
        Self::from_path(sample_rate, DEFAULT_WEIGHTS_PATH)
    }

    /// Load from a custom weights path.
    pub fn from_path(sample_rate: u32, path: &str) -> Result<Self, String> {
        if sample_rate != 16000 {
            return Err(format!(
                "SileroVadNative only supports 16000 Hz, got {}. Use SileroVadOrt for 8kHz.",
                sample_rate
            ));
        }
        let data = std::fs::read(path)
            .map_err(|e| format!("Failed to read weights from {}: {}", path, e))?;
        let engine = Engine::from_bytes(&data)?;
        log::info!("SileroVadNative: loaded weights from {} (sr=16000)", path);
        Ok(Self {
            inner: Arc::new(Mutex::new(SileroNativeInner {
                engine,
                state: InferState::new(CONTEXT_16K),
                num_samples: 512,
            })),
        })
    }

    /// Load from raw bytes (e.g. `include_bytes!`).
    pub fn from_bytes(sample_rate: u32, data: &[u8]) -> Result<Self, String> {
        if sample_rate != 16000 {
            return Err(format!(
                "SileroVadNative only supports 16000 Hz, got {}.",
                sample_rate
            ));
        }
        let engine = Engine::from_bytes(data)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(SileroNativeInner {
                engine,
                state: InferState::new(CONTEXT_16K),
                num_samples: 512,
            })),
        })
    }

    /// Run inference on i16 LE mono PCM bytes (1024 bytes = 512 samples).
    pub fn infer(&self, audio_bytes: &[u8]) -> Result<f32, String> {
        let mut guard = self.inner.lock().unwrap();
        let expected = guard.num_samples * 2;
        if audio_bytes.len() != expected {
            return Err(format!(
                "Audio length mismatch: expected {} bytes, got {}",
                expected,
                audio_bytes.len()
            ));
        }

        let samples: Vec<f32> = audio_bytes
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect();

        let inner = &mut *guard;
	Ok(inner.engine.infer(&samples, &mut inner.state))
    }

    /// Async wrapper — runs inference on a blocking thread.
    pub async fn infer_async(&self, audio_bytes: Vec<u8>) -> Result<f32, String> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut guard = inner.lock().unwrap();
            let samples: Vec<f32> = audio_bytes
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                .collect();
            let inner = &mut *guard;
	    Ok(inner.engine.infer(&samples, &mut inner.state))
        })
        .await
        .map_err(|e| format!("spawn_blocking error: {}", e))?
    }
}

// ─── VadAnalyzer impl ─────────────────────────────────────────────────

#[async_trait::async_trait]
impl VadAnalyzer for SileroVadNative {
    fn num_frames_required(&self) -> usize {
        self.inner.lock().unwrap().num_samples
    }

    async fn voice_confidence(&self, audio: Vec<u8>) -> f32 {
        match self.infer_async(audio).await {
            Ok(c) => c,
            Err(e) => {
                log::error!("SileroVadNative: inference error: {}", e);
                0.0
            }
        }
    }
}
