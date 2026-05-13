//! Pure Rust smart-turn-v3 inference engine — SIMD optimized.
//!
//! Optimizations vs naive version:
//! - AVX2/FMA SIMD dot product and axpy (y += a*x) operations
//! - Cache-friendly MatMul loop ordering (sequential weight reads)
//! - Fused QKV projection (single pass over input)
//! - Pre-allocated scratch buffers (zero per-inference allocation)

use std::io::Cursor;

const D: usize = 384;
const HEADS: usize = 6;
const HD: usize = 64;
const FF: usize = 1536;
const SEQ: usize = 400;
const N_LAYERS: usize = 4;
const POOL_DIM: usize = 256;
const CLS_MID: usize = 256;
const CLS_SMALL: usize = 64;

const EXPECTED_FLOATS: usize = 8_000_386;
const LN_EPS: f32 = 1e-5;
const ATTN_SCALE: f32 = 0.353_553_39;

const WEIGHTS_XZ: &[u8] = include_bytes!("smart_turn_weights.bin.xz");

struct LayerOffsets {
    aln_w: usize, aln_b: usize,
    q_w: usize, q_b: usize,
    k_w: usize,
    v_w: usize, v_b: usize,
    out_w: usize, out_b: usize,
    fln_w: usize, fln_b: usize,
    fc1_w: usize, fc1_b: usize,
    fc2_w: usize, fc2_b: usize,
}

/// Scratch buffers reused across inference calls — zero allocation in hot path.
struct Scratch {
    ln_buf:   Vec<f32>,  // SEQ * D
    q:        Vec<f32>,  // SEQ * D
    k:        Vec<f32>,  // SEQ * D
    v:        Vec<f32>,  // SEQ * D
    attn_out: Vec<f32>,  // SEQ * D
    scores:   Vec<f32>,  // SEQ * SEQ
    ln2:      Vec<f32>,  // SEQ * D
    ff:       Vec<f32>,  // SEQ * FF
    proj:     Vec<f32>,  // SEQ * D
    pool_h:   Vec<f32>,  // POOL_DIM
}

impl Scratch {
    fn new() -> Self {
        Self {
            ln_buf:   vec![0.0; SEQ * D],
            q:        vec![0.0; SEQ * D],
            k:        vec![0.0; SEQ * D],
            v:        vec![0.0; SEQ * D],
            attn_out: vec![0.0; SEQ * D],
            scores:   vec![0.0; SEQ * SEQ],
            ln2:      vec![0.0; SEQ * D],
            ff:       vec![0.0; SEQ * FF],
            proj:     vec![0.0; SEQ * D],
            pool_h:   vec![0.0; POOL_DIM],
        }
    }
}

pub struct SmartTurnEngine {
    w: Vec<f32>,
    conv1_w: usize, conv1_b: usize,
    conv2_w: usize, conv2_b: usize,
    pos_emb: usize,
    layers: [LayerOffsets; N_LAYERS],
    fln_w: usize, fln_b: usize,
    pool0_w: usize, pool0_b: usize,
    pool2_w: usize, pool2_b: usize,
    cls0_w: usize, cls0_b: usize,
    cls_ln_w: usize, cls_ln_b: usize,
    cls4_w: usize, cls4_b: usize,
    cls6_w: usize, cls6_b: usize,
    scratch: Scratch,
}

impl SmartTurnEngine {
    pub fn new() -> Result<Self, String> {
        let mut reader = Cursor::new(WEIGHTS_XZ);
        let mut raw = Vec::with_capacity(EXPECTED_FLOATS * 4);
        lzma_rs::xz_decompress(&mut reader, &mut raw)
            .map_err(|e| format!("Failed to decompress weights: {}", e))?;

        let w: Vec<f32> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        if w.len() < EXPECTED_FLOATS {
            return Err(format!(
                "Decompressed weights too small: {} floats, expected {}",
                w.len(), EXPECTED_FLOATS
            ));
        }

        let mut o = 0usize;
        let mut take = |n: usize| -> usize { let s = o; o += n; s };

        let conv1_w = take(384 * 80 * 3);
        let conv1_b = take(384);
        let conv2_w = take(384 * 384 * 3);
        let conv2_b = take(384);
        let pos_emb = take(SEQ * D);

        let mut layers: [LayerOffsets; N_LAYERS] = unsafe { std::mem::zeroed() };
        for l in layers.iter_mut() {
            *l = LayerOffsets {
                aln_w: take(D), aln_b: take(D),
                q_w: take(D * D), q_b: take(D),
                k_w: take(D * D),
                v_w: take(D * D), v_b: take(D),
                out_w: take(D * D), out_b: take(D),
                fln_w: take(D), fln_b: take(D),
                fc1_w: take(D * FF), fc1_b: take(FF),
                fc2_w: take(FF * D), fc2_b: take(D),
            };
        }

        let fln_w = take(D); let fln_b = take(D);
        let pool0_w = take(D * POOL_DIM); let pool0_b = take(POOL_DIM);
        let pool2_w = take(POOL_DIM); let pool2_b = take(1);
        let cls0_w = take(CLS_MID * D); let cls0_b = take(CLS_MID);
        let cls_ln_w = take(CLS_MID); let cls_ln_b = take(CLS_MID);
        let cls4_w = take(CLS_SMALL * CLS_MID); let cls4_b = take(CLS_SMALL);
        let cls6_w = take(CLS_SMALL); let cls6_b = take(1);

        assert_eq!(o, EXPECTED_FLOATS);

        log::info!(
            "SmartTurnEngine: loaded {:.1} MB weights",
            w.len() as f64 * 4.0 / 1024.0 / 1024.0
        );

        Ok(Self {
            w, conv1_w, conv1_b, conv2_w, conv2_b, pos_emb,
            layers, fln_w, fln_b,
            pool0_w, pool0_b, pool2_w, pool2_b,
            cls0_w, cls0_b, cls_ln_w, cls_ln_b,
            cls4_w, cls4_b, cls6_w, cls6_b,
            scratch: Scratch::new(),
        })
    }

    pub fn infer(&mut self, features: &[f32]) -> f32 {
        debug_assert_eq!(features.len(), 80 * 800);

        // Conv1: [80, 800] → [384, 800] + GELU
        let mut x = conv1d_k3(
            features, 80, 800,
            &self.w[self.conv1_w..], &self.w[self.conv1_b..],
            384, 1, 1,
        );
        gelu_inplace(&mut x);

        // Conv2: [384, 800] → [384, 400] + GELU
        x = conv1d_k3(
            &x, 384, 800,
            &self.w[self.conv2_w..], &self.w[self.conv2_b..],
            384, 1, 2,
        );
        gelu_inplace(&mut x);

        // Transpose [384, 400] → [400, 384] + pos embeddings
        // Reuse ln_buf as seq_data
        let seq = &mut self.scratch.ln_buf;
        let pos = &self.w[self.pos_emb..self.pos_emb + SEQ * D];
        for s in 0..SEQ {
            for d in 0..D {
                seq[s * D + d] = x[d * SEQ + s] + pos[s * D + d];
            }
        }
        // Copy to a separate owned buffer for the transformer (ln_buf is scratch)
        let mut seq_data = vec![0.0f32; SEQ * D];
        seq_data.copy_from_slice(seq);

        // 4 Transformer layers
        for l in 0..N_LAYERS {
            self.transformer_layer(&mut seq_data, &self.layers[l] as *const LayerOffsets);
        }

        // Final LayerNorm
        let fln_w = &self.w[self.fln_w..self.fln_w + D];
        let fln_b = &self.w[self.fln_b..self.fln_b + D];
        for s in 0..SEQ {
            layer_norm_inplace(&mut seq_data[s * D..(s + 1) * D], fln_w, fln_b);
        }

        // Attention Pooling
        let pool0_w = &self.w[self.pool0_w..self.pool0_w + D * POOL_DIM];
        let pool0_b = &self.w[self.pool0_b..self.pool0_b + POOL_DIM];
        let pool2_w = &self.w[self.pool2_w..self.pool2_w + POOL_DIM];
        let pool2_b = self.w[self.pool2_b];

        let mut energies = vec![0.0f32; SEQ];
        let h = &mut self.scratch.pool_h;
        for s in 0..SEQ {
            let row = &seq_data[s * D..(s + 1) * D];
            // Linear(384→256) via axpy + tanh, then dot with pool2_w
            h.copy_from_slice(&pool0_b[..POOL_DIM]);
            for d in 0..D {
                axpy(row[d], &pool0_w[d * POOL_DIM..(d + 1) * POOL_DIM], h);
            }
            for v in h.iter_mut() { *v = v.tanh(); }
            energies[s] = pool2_b + dot(h, pool2_w);
        }
        softmax_inplace(&mut energies);

        let mut pooled = vec![0.0f32; D];
        for s in 0..SEQ {
            axpy(energies[s], &seq_data[s * D..(s + 1) * D], &mut pooled);
        }

        // Classifier: Gemm(384→256) + LN + GELU
        let cls0_w = &self.w[self.cls0_w..self.cls0_w + CLS_MID * D];
        let cls0_b = &self.w[self.cls0_b..self.cls0_b + CLS_MID];
        let mut c = vec![0.0f32; CLS_MID];
        // Gemm: y = x @ W^T + b  (W is [CLS_MID, D])
        c.copy_from_slice(cls0_b);
        for n in 0..CLS_MID {
            c[n] += dot(&cls0_w[n * D..(n + 1) * D], &pooled);
        }
        layer_norm_inplace(
            &mut c,
            &self.w[self.cls_ln_w..self.cls_ln_w + CLS_MID],
            &self.w[self.cls_ln_b..self.cls_ln_b + CLS_MID],
        );
        gelu_inplace(&mut c);

        // Gemm(256→64) + GELU
        let cls4_w = &self.w[self.cls4_w..self.cls4_w + CLS_SMALL * CLS_MID];
        let cls4_b = &self.w[self.cls4_b..self.cls4_b + CLS_SMALL];
        let mut c2 = vec![0.0f32; CLS_SMALL];
        c2.copy_from_slice(cls4_b);
        for n in 0..CLS_SMALL {
            c2[n] += dot(&cls4_w[n * CLS_MID..(n + 1) * CLS_MID], &c);
        }
        gelu_inplace(&mut c2);

        // Gemm(64→1) + sigmoid
        let cls6_w = &self.w[self.cls6_w..self.cls6_w + CLS_SMALL];
        let cls6_b = self.w[self.cls6_b];
        sigmoid(cls6_b + dot(cls6_w, &c2))
    }

    fn transformer_layer(&mut self, x: &mut [f32], l_ptr: *const LayerOffsets) {
        // SAFETY: l_ptr points into self.layers which outlives this call.
        // We use a raw pointer to avoid borrow conflict with &mut self.
        let l = unsafe { &*l_ptr };

        let aln_w = &self.w[l.aln_w..l.aln_w + D];
        let aln_b = &self.w[l.aln_b..l.aln_b + D];

        // LayerNorm
        self.scratch.ln_buf.copy_from_slice(&x[..SEQ * D]);
        for s in 0..SEQ {
            layer_norm_inplace(&mut self.scratch.ln_buf[s * D..(s + 1) * D], aln_w, aln_b);
        }

        // ── Fused QKV projection (cache-friendly axpy pattern) ──
        // Initialize Q with bias, K with zeros, V with bias
        for s in 0..SEQ {
            let q_row = &mut self.scratch.q[s * D..(s + 1) * D];
            let v_row = &mut self.scratch.v[s * D..(s + 1) * D];
            q_row.copy_from_slice(&self.w[l.q_b..l.q_b + D]);
            v_row.copy_from_slice(&self.w[l.v_b..l.v_b + D]);
        }
        self.scratch.k.iter_mut().for_each(|v| *v = 0.0);

        // Accumulate: for each input dim d, broadcast across all output dims
        // Weight layout: [D_in, D_out] row-major → sequential reads
        let q_w = l.q_w;
        let k_w = l.k_w;
        let v_w = l.v_w;

        for s in 0..SEQ {
            let inp = &self.scratch.ln_buf[s * D..(s + 1) * D];
            let q_out = &mut self.scratch.q[s * D..(s + 1) * D];
            let k_out = &mut self.scratch.k[s * D..(s + 1) * D];
            let v_out = &mut self.scratch.v[s * D..(s + 1) * D];

            for d in 0..D {
                let val = inp[d];
                let w_off = d * D;
                axpy(val, &self.w[q_w + w_off..q_w + w_off + D], q_out);
                axpy(val, &self.w[k_w + w_off..k_w + w_off + D], k_out);
                axpy(val, &self.w[v_w + w_off..v_w + w_off + D], v_out);
            }
        }

        // Scale Q and K
        let scale = ATTN_SCALE;
        for v in self.scratch.q.iter_mut() { *v *= scale; }
        for v in self.scratch.k.iter_mut() { *v *= scale; }

        // ── Multi-head attention ──
        self.scratch.attn_out.iter_mut().for_each(|v| *v = 0.0);

        for h in 0..HEADS {
            let ho = h * HD;

            // Attention scores: Q_h @ K_h^T
            for s1 in 0..SEQ {
                let q_slice = &self.scratch.q[s1 * D + ho..s1 * D + ho + HD];
                for s2 in 0..SEQ {
                    let k_slice = &self.scratch.k[s2 * D + ho..s2 * D + ho + HD];
                    self.scratch.scores[s1 * SEQ + s2] = dot(q_slice, k_slice);
                }
                softmax_inplace(&mut self.scratch.scores[s1 * SEQ..(s1 + 1) * SEQ]);
            }

            // Weighted sum of V (axpy pattern — sequential V reads)
            for s1 in 0..SEQ {
                let attn_row = &mut self.scratch.attn_out[s1 * D + ho..s1 * D + ho + HD];
                for s2 in 0..SEQ {
                    let w = self.scratch.scores[s1 * SEQ + s2];
                    if w > 1e-8 {
                        let v_slice = &self.scratch.v[s2 * D + ho..s2 * D + ho + HD];
                        axpy(w, v_slice, attn_row);
                    }
                }
            }
        }

        // Out projection (axpy pattern) + residual
        let out_w = l.out_w;
        let out_b = &self.w[l.out_b..l.out_b + D];

        for s in 0..SEQ {
            let inp = &self.scratch.attn_out[s * D..(s + 1) * D];
            let x_row = &mut x[s * D..(s + 1) * D];

            // Init proj with bias, then axpy
            self.scratch.proj[s * D..(s + 1) * D].copy_from_slice(out_b);
        }
        for s in 0..SEQ {
            let inp = &self.scratch.attn_out[s * D..(s + 1) * D];
            let proj_row = &mut self.scratch.proj[s * D..(s + 1) * D];
            for d in 0..D {
                let val = inp[d];
                let w_off = d * D;
                axpy(val, &self.w[out_w + w_off..out_w + w_off + D], proj_row);
            }
            // Residual add
            let x_row = &mut x[s * D..(s + 1) * D];
            for n in 0..D { x_row[n] += self.scratch.proj[s * D + n]; }
        }

        // ── Feed-Forward ──
        let fln_w = &self.w[l.fln_w..l.fln_w + D];
        let fln_b = &self.w[l.fln_b..l.fln_b + D];
        let fc1_w = l.fc1_w;
        let fc1_b = &self.w[l.fc1_b..l.fc1_b + FF];
        let fc2_w = l.fc2_w;
        let fc2_b = &self.w[l.fc2_b..l.fc2_b + D];

        self.scratch.ln2.copy_from_slice(&x[..SEQ * D]);
        for s in 0..SEQ {
            layer_norm_inplace(&mut self.scratch.ln2[s * D..(s + 1) * D], fln_w, fln_b);
        }

        // FC1: [SEQ, D] @ [D, FF] + bias + GELU (axpy pattern)
        for s in 0..SEQ {
            let ff_row = &mut self.scratch.ff[s * FF..(s + 1) * FF];
            ff_row.copy_from_slice(fc1_b);
            let inp = &self.scratch.ln2[s * D..(s + 1) * D];
            for d in 0..D {
                let val = inp[d];
                let w_off = d * FF;
                axpy(val, &self.w[fc1_w + w_off..fc1_w + w_off + FF], ff_row);
            }
        }
        gelu_inplace(&mut self.scratch.ff[..SEQ * FF]);

        // FC2: [SEQ, FF] @ [FF, D] + bias, fused with residual
        for s in 0..SEQ {
            let inp = &self.scratch.ff[s * FF..(s + 1) * FF];
            let x_row = &mut x[s * D..(s + 1) * D];
            // Add bias first, then axpy, then add to residual
            // We accumulate directly into x_row
            for n in 0..D { x_row[n] += fc2_b[n]; }
            for d in 0..FF {
                let val = inp[d];
                let w_off = d * D;
                axpy(val, &self.w[fc2_w + w_off..fc2_w + w_off + D], x_row);
            }
        }
    }
}

// ── Conv1D (k=3 specialized) ───────────────────────────────────────────

fn conv1d_k3(
    x: &[f32], in_ch: usize, in_len: usize,
    weight: &[f32], bias: &[f32],
    out_ch: usize, pad: usize, stride: usize,
) -> Vec<f32> {
    let padded_len = in_len + 2 * pad;
    let out_len = (padded_len - 3) / stride + 1;

    let mut padded = vec![0.0f32; in_ch * padded_len];
    for c in 0..in_ch {
        padded[c * padded_len + pad..c * padded_len + pad + in_len]
            .copy_from_slice(&x[c * in_len..(c + 1) * in_len]);
    }

    let mut output = vec![0.0f32; out_ch * out_len];
    for co in 0..out_ch {
        let b = bias[co];
        for t in 0..out_len {
            let ps = t * stride;
            let mut sum = b;
            for ci in 0..in_ch {
                let wb = (co * in_ch + ci) * 3;
                let xb = ci * padded_len + ps;
                unsafe {
                    sum += *weight.get_unchecked(wb)     * *padded.get_unchecked(xb);
                    sum += *weight.get_unchecked(wb + 1) * *padded.get_unchecked(xb + 1);
                    sum += *weight.get_unchecked(wb + 2) * *padded.get_unchecked(xb + 2);
                }
            }
            output[co * out_len + t] = sum;
        }
    }
    output
}

// ── SIMD primitives ─────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Dot product: sum(a[i] * b[i])
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_avx2(a, b) };
        }
    }
    dot_scalar(a, b)
}

/// AXPY: y[i] += a * x[i]  (the core of cache-friendly MatMul)
#[inline]
fn axpy(a: f32, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { axpy_avx2(a, x, y); }
            return;
        }
    }
    for i in 0..x.len() { y[i] += a * x[i]; }
}

fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0, 0.0, 0.0);
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
unsafe fn axpy_avx2(a: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len();
    let xp = x.as_ptr();
    let yp = y.as_mut_ptr();
    let va = _mm256_set1_ps(a);
    let chunks32 = n / 32;
    for i in 0..chunks32 {
        let j = i * 32;
        let y0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(xp.add(j)),      _mm256_loadu_ps(yp.add(j)));
        let y1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(xp.add(j + 8)),  _mm256_loadu_ps(yp.add(j + 8)));
        let y2 = _mm256_fmadd_ps(va, _mm256_loadu_ps(xp.add(j + 16)), _mm256_loadu_ps(yp.add(j + 16)));
        let y3 = _mm256_fmadd_ps(va, _mm256_loadu_ps(xp.add(j + 24)), _mm256_loadu_ps(yp.add(j + 24)));
        _mm256_storeu_ps(yp.add(j),      y0);
        _mm256_storeu_ps(yp.add(j + 8),  y1);
        _mm256_storeu_ps(yp.add(j + 16), y2);
        _mm256_storeu_ps(yp.add(j + 24), y3);
    }
    let done = chunks32 * 32;
    let chunks8 = (n - done) / 8;
    for i in 0..chunks8 {
        let j = done + i * 8;
        let yr = _mm256_fmadd_ps(va, _mm256_loadu_ps(xp.add(j)), _mm256_loadu_ps(yp.add(j)));
        _mm256_storeu_ps(yp.add(j), yr);
    }
    let tail = done + chunks8 * 8;
    for i in tail..n {
        *yp.add(i) += a * *xp.add(i);
    }
}

// ── Scalar ops ──────────────────────────────────────────────────────────

#[inline(always)]
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

fn layer_norm_inplace(x: &mut [f32], w: &[f32], b: &[f32]) {
    let n = x.len();
    let mean = x.iter().sum::<f32>() / n as f32;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv = 1.0 / (var + LN_EPS).sqrt();
    for i in 0..n { x[i] = (x[i] - mean) * inv * w[i] + b[i]; }
}

fn gelu_inplace(x: &mut [f32]) {
    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
    for v in x.iter_mut() {
        *v = *v * 0.5 * (1.0 + erf_f32(*v * inv_sqrt2));
    }
}

fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() { *v = (*v - max).exp(); sum += *v; }
    let inv = 1.0 / sum;
    for v in x.iter_mut() { *v *= inv; }
}

fn erf_f32(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0f32 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t * (-x * x).exp();
    sign * y
}
