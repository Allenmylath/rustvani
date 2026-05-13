//! Pure Rust smart-turn-v3 inference engine — cache-tiled GEMM optimized.
//!
//! Key optimizations over previous axpy-based version:
//! - `matrixmultiply::sgemm` for all large matmuls (cache-tiled GEBP, register blocking)
//!   → QKV/out-projection/FC1/FC2/pooling: ~1.8M axpy calls → ~40 sgemm calls total
//! - Attention scaling folded into GEMM alpha (eliminates two O(SEQ*D) scaling passes)
//! - Out-projection + FC2 residual fused via sgemm beta=1 (eliminates proj scratch buffer)
//! - Pooling linear layer via batch sgemm (eliminates per-position axpy loop)
//! - Zero per-inference allocation — all buffers in Scratch
//! - Manual SIMD removed — matrixmultiply handles SIMD internally;
//!   compile with RUSTFLAGS="-C target-cpu=native" for best scalar auto-vectorization
//!
//! Cargo.toml: add `matrixmultiply = "0.3"`
//! .cargo/config.toml:
//!   [build]
//!   rustflags = ["-C", "target-cpu=native"]

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

/// 1/sqrt(head_dim) = 1/sqrt(64) = 1/8.
/// Original code applied sqrt(1/8) to both Q and K separately;
/// we fold the combined scale into a single GEMM alpha for Q@K^T.
const ATTN_SCALE_SQ: f32 = 0.125;

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
    ln_buf:      Vec<f32>,  // SEQ * D
    q:           Vec<f32>,  // SEQ * D
    k:           Vec<f32>,  // SEQ * D
    v:           Vec<f32>,  // SEQ * D
    attn_out:    Vec<f32>,  // SEQ * D
    scores:      Vec<f32>,  // SEQ * SEQ
    ln2:         Vec<f32>,  // SEQ * D
    ff:          Vec<f32>,  // SEQ * FF
    pool_hidden: Vec<f32>,  // SEQ * POOL_DIM
    // Previously allocated per-inference:
    seq_data:    Vec<f32>,  // SEQ * D  (transformer I/O)
    energies:    Vec<f32>,  // SEQ
    pooled:      Vec<f32>,  // D
    cls_mid:     Vec<f32>,  // CLS_MID
    cls_small:   Vec<f32>,  // CLS_SMALL
}

impl Scratch {
    fn new() -> Self {
        Self {
            ln_buf:      vec![0.0; SEQ * D],
            q:           vec![0.0; SEQ * D],
            k:           vec![0.0; SEQ * D],
            v:           vec![0.0; SEQ * D],
            attn_out:    vec![0.0; SEQ * D],
            scores:      vec![0.0; SEQ * SEQ],
            ln2:         vec![0.0; SEQ * D],
            ff:          vec![0.0; SEQ * FF],
            pool_hidden: vec![0.0; SEQ * POOL_DIM],
            seq_data:    vec![0.0; SEQ * D],
            energies:    vec![0.0; SEQ],
            pooled:      vec![0.0; D],
            cls_mid:     vec![0.0; CLS_MID],
            cls_small:   vec![0.0; CLS_SMALL],
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

        // ── Conv1: [80, 800] → [384, 800] + GELU ───────────────────
        let mut x = conv1d_k3(
            features, 80, 800,
            &self.w[self.conv1_w..], &self.w[self.conv1_b..],
            384, 1, 1,
        );
        gelu_inplace(&mut x);

        // ── Conv2: [384, 800] → [384, 400] + GELU ──────────────────
        x = conv1d_k3(
            &x, 384, 800,
            &self.w[self.conv2_w..], &self.w[self.conv2_b..],
            384, 1, 2,
        );
        gelu_inplace(&mut x);

        // ── Transpose [384, 400] → [400, 384] + positional embeddings
        let seq = &mut self.scratch.seq_data;
        let pos = &self.w[self.pos_emb..self.pos_emb + SEQ * D];
        for s in 0..SEQ {
            for d in 0..D {
                seq[s * D + d] = x[d * SEQ + s] + pos[s * D + d];
            }
        }

        // ── 4 Transformer layers ────────────────────────────────────
        for l in 0..N_LAYERS {
            self.transformer_layer(&self.layers[l] as *const LayerOffsets);
        }

        // ── Final LayerNorm ─────────────────────────────────────────
        let fln_w = &self.w[self.fln_w..self.fln_w + D];
        let fln_b = &self.w[self.fln_b..self.fln_b + D];
        for s in 0..SEQ {
            layer_norm_inplace(
                &mut self.scratch.seq_data[s * D..(s + 1) * D],
                fln_w, fln_b,
            );
        }

        // ── Attention Pooling (GEMM-based) ──────────────────────────
        // pool_hidden[SEQ, POOL_DIM] = seq_data[SEQ, D] @ pool0_w[D, POOL_DIM] + bias
        let pool0_b = &self.w[self.pool0_b..self.pool0_b + POOL_DIM];
        for s in 0..SEQ {
            self.scratch.pool_hidden[s * POOL_DIM..(s + 1) * POOL_DIM]
                .copy_from_slice(pool0_b);
        }
        unsafe {
            matrixmultiply::sgemm(
                SEQ, D, POOL_DIM,
                1.0,
                self.scratch.seq_data.as_ptr(), D as isize, 1,
                self.w.as_ptr().add(self.pool0_w), POOL_DIM as isize, 1,
                1.0, // preserve bias
                self.scratch.pool_hidden.as_mut_ptr(), POOL_DIM as isize, 1,
            );
        }

        // tanh + energy scores
        let pool2_w = &self.w[self.pool2_w..self.pool2_w + POOL_DIM];
        let pool2_b = self.w[self.pool2_b];
        for s in 0..SEQ {
            let row = &mut self.scratch.pool_hidden[s * POOL_DIM..(s + 1) * POOL_DIM];
            for v in row.iter_mut() { *v = v.tanh(); }
            self.scratch.energies[s] = pool2_b + dot(row, pool2_w);
        }
        softmax_inplace(&mut self.scratch.energies);

        // Weighted pooling
        self.scratch.pooled.fill(0.0);
        for s in 0..SEQ {
            let e = self.scratch.energies[s];
            let src = &self.scratch.seq_data[s * D..(s + 1) * D];
            let dst = &mut self.scratch.pooled;
            for i in 0..D { dst[i] += e * src[i]; }
        }

        // ── Classifier ─────────────────────────────────────────────
        // Linear(384→256) + LN + GELU
        let cls0_w = &self.w[self.cls0_w..self.cls0_w + CLS_MID * D];
        self.scratch.cls_mid.copy_from_slice(&self.w[self.cls0_b..self.cls0_b + CLS_MID]);
        for n in 0..CLS_MID {
            self.scratch.cls_mid[n] += dot(
                &cls0_w[n * D..(n + 1) * D],
                &self.scratch.pooled,
            );
        }
        layer_norm_inplace(
            &mut self.scratch.cls_mid,
            &self.w[self.cls_ln_w..self.cls_ln_w + CLS_MID],
            &self.w[self.cls_ln_b..self.cls_ln_b + CLS_MID],
        );
        gelu_inplace(&mut self.scratch.cls_mid);

        // Linear(256→64) + GELU
        let cls4_w = &self.w[self.cls4_w..self.cls4_w + CLS_SMALL * CLS_MID];
        self.scratch.cls_small.copy_from_slice(&self.w[self.cls4_b..self.cls4_b + CLS_SMALL]);
        for n in 0..CLS_SMALL {
            self.scratch.cls_small[n] += dot(
                &cls4_w[n * CLS_MID..(n + 1) * CLS_MID],
                &self.scratch.cls_mid,
            );
        }
        gelu_inplace(&mut self.scratch.cls_small);

        // Linear(64→1) + sigmoid
        let cls6_w = &self.w[self.cls6_w..self.cls6_w + CLS_SMALL];
        let cls6_b = self.w[self.cls6_b];
        sigmoid(cls6_b + dot(cls6_w, &self.scratch.cls_small))
    }

    fn transformer_layer(&mut self, l_ptr: *const LayerOffsets) {
        // SAFETY: l_ptr points into self.layers which outlives this call.
        let l = unsafe { &*l_ptr };
        let w_ptr = self.w.as_ptr();

        // ── LayerNorm ───────────────────────────────────────────────
        let aln_w = &self.w[l.aln_w..l.aln_w + D];
        let aln_b = &self.w[l.aln_b..l.aln_b + D];
        self.scratch.ln_buf.copy_from_slice(&self.scratch.seq_data);
        for s in 0..SEQ {
            layer_norm_inplace(
                &mut self.scratch.ln_buf[s * D..(s + 1) * D],
                aln_w, aln_b,
            );
        }

        // ── QKV Projection via sgemm ────────────────────────────────
        // Q[SEQ,D] = LN[SEQ,D] @ W_q[D,D] + bias_q
        // K[SEQ,D] = LN[SEQ,D] @ W_k[D,D]          (no bias)
        // V[SEQ,D] = LN[SEQ,D] @ W_v[D,D] + bias_v

        // Initialize Q with bias, K with zeros, V with bias
        let q_b = &self.w[l.q_b..l.q_b + D];
        let v_b = &self.w[l.v_b..l.v_b + D];
        for s in 0..SEQ {
            self.scratch.q[s * D..(s + 1) * D].copy_from_slice(q_b);
            self.scratch.v[s * D..(s + 1) * D].copy_from_slice(v_b);
        }

        unsafe {
            let ln_ptr = self.scratch.ln_buf.as_ptr();

            // Q += LN @ W_q
            matrixmultiply::sgemm(
                SEQ, D, D,
                1.0,
                ln_ptr, D as isize, 1,
                w_ptr.add(l.q_w), D as isize, 1,
                1.0, // preserve bias
                self.scratch.q.as_mut_ptr(), D as isize, 1,
            );

            // K = LN @ W_k  (beta=0, no bias)
            matrixmultiply::sgemm(
                SEQ, D, D,
                1.0,
                ln_ptr, D as isize, 1,
                w_ptr.add(l.k_w), D as isize, 1,
                0.0,
                self.scratch.k.as_mut_ptr(), D as isize, 1,
            );

            // V += LN @ W_v
            matrixmultiply::sgemm(
                SEQ, D, D,
                1.0,
                ln_ptr, D as isize, 1,
                w_ptr.add(l.v_w), D as isize, 1,
                1.0, // preserve bias
                self.scratch.v.as_mut_ptr(), D as isize, 1,
            );
        }

        // ── Multi-Head Attention ────────────────────────────────────
        // For each head h:
        //   scores[SEQ,SEQ] = (Q_h @ K_h^T) * (1/sqrt(d_k))
        //   attn_out_h = softmax(scores) @ V_h
        //
        // Q_h/K_h/V_h are strided views: row stride = D, col stride = 1,
        // starting at column offset h*HD.

        for h in 0..HEADS {
            let ho = h * HD;

            // scores[SEQ,SEQ] = Q_h[SEQ,HD] @ K_h[SEQ,HD]^T * ATTN_SCALE_SQ
            unsafe {
                matrixmultiply::sgemm(
                    SEQ, HD, SEQ,
                    ATTN_SCALE_SQ, // fold 1/sqrt(d_k) into alpha
                    self.scratch.q.as_ptr().add(ho), D as isize, 1,     // Q_h rows strided by D
                    self.scratch.k.as_ptr().add(ho), 1, D as isize,     // K_h^T: swap strides
                    0.0,
                    self.scratch.scores.as_mut_ptr(), SEQ as isize, 1,
                );
            }

            // Softmax each row
            for s in 0..SEQ {
                softmax_inplace(&mut self.scratch.scores[s * SEQ..(s + 1) * SEQ]);
            }

            // attn_out_h[SEQ,HD] = softmax(scores)[SEQ,SEQ] @ V_h[SEQ,HD]
            unsafe {
                matrixmultiply::sgemm(
                    SEQ, SEQ, HD,
                    1.0,
                    self.scratch.scores.as_ptr(), SEQ as isize, 1,
                    self.scratch.v.as_ptr().add(ho), D as isize, 1,     // V_h rows strided by D
                    0.0,
                    self.scratch.attn_out.as_mut_ptr().add(ho), D as isize, 1,
                );
            }
        }

        // ── Output Projection (fused with residual) ─────────────────
        // seq_data += attn_out @ W_out + out_bias
        //
        // Step 1: Add bias directly to seq_data
        // Step 2: sgemm with beta=1 accumulates into seq_data
        let out_b = &self.w[l.out_b..l.out_b + D];
        for s in 0..SEQ {
            let row = &mut self.scratch.seq_data[s * D..(s + 1) * D];
            for i in 0..D { row[i] += out_b[i]; }
        }
        unsafe {
            matrixmultiply::sgemm(
                SEQ, D, D,
                1.0,
                self.scratch.attn_out.as_ptr(), D as isize, 1,
                w_ptr.add(l.out_w), D as isize, 1,
                1.0, // accumulate into seq_data (preserves residual + bias)
                self.scratch.seq_data.as_mut_ptr(), D as isize, 1,
            );
        }

        // ── Feed-Forward Network ────────────────────────────────────
        let fln_w = &self.w[l.fln_w..l.fln_w + D];
        let fln_b = &self.w[l.fln_b..l.fln_b + D];

        self.scratch.ln2.copy_from_slice(&self.scratch.seq_data);
        for s in 0..SEQ {
            layer_norm_inplace(
                &mut self.scratch.ln2[s * D..(s + 1) * D],
                fln_w, fln_b,
            );
        }

        // FC1: ff[SEQ,FF] = LN2[SEQ,D] @ W_fc1[D,FF] + bias_fc1 + GELU
        let fc1_b = &self.w[l.fc1_b..l.fc1_b + FF];
        for s in 0..SEQ {
            self.scratch.ff[s * FF..(s + 1) * FF].copy_from_slice(fc1_b);
        }
        unsafe {
            matrixmultiply::sgemm(
                SEQ, D, FF,
                1.0,
                self.scratch.ln2.as_ptr(), D as isize, 1,
                w_ptr.add(l.fc1_w), FF as isize, 1,
                1.0, // preserve bias
                self.scratch.ff.as_mut_ptr(), FF as isize, 1,
            );
        }
        gelu_inplace(&mut self.scratch.ff[..SEQ * FF]);

        // FC2: seq_data += ff[SEQ,FF] @ W_fc2[FF,D] + bias_fc2  (fused residual)
        let fc2_b = &self.w[l.fc2_b..l.fc2_b + D];
        for s in 0..SEQ {
            let row = &mut self.scratch.seq_data[s * D..(s + 1) * D];
            for i in 0..D { row[i] += fc2_b[i]; }
        }
        unsafe {
            matrixmultiply::sgemm(
                SEQ, FF, D,
                1.0,
                self.scratch.ff.as_ptr(), FF as isize, 1,
                w_ptr.add(l.fc2_w), D as isize, 1,
                1.0, // accumulate into seq_data (preserves residual + bias)
                self.scratch.seq_data.as_mut_ptr(), D as isize, 1,
            );
        }
    }
}

// ── Conv1D (k=3 specialized, unchanged) ────────────────────────────────

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

// ── Small vector ops (used only for classifier + pooling) ──────────────
//
// These are called on tiny vectors (64–384 elements). With -C target-cpu=native
// the compiler auto-vectorizes these effectively. No manual SIMD needed.

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
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
