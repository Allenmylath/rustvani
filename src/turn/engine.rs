//! Pure Rust smart-turn-v3 inference engine — INT8 weights, GEMM optimized.
//!
//! Architecture unchanged: INT8 transformer weights stay quantized in memory.
//! At inference, each weight matrix is batch-dequantized once into a f32
//! scratch buffer, then multiplied via matrixmultiply::sgemm (cache-tiled
//! GEBP with register blocking).
//!
//! Key wins over previous dequant_axpy version:
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ 1. Batch dequant  — each INT8 weight dequantized ONCE, not SEQ=400 │
//! │    times. FC1 alone: 236M → 590K dequant ops (~400× reduction)     │
//! │ 2. sgemm          — cache-tiled GEMM replaces ~1.8M axpy calls     │
//! │ 3. Fused scaling   — attention 1/sqrt(d_k) folded into GEMM alpha  │
//! │ 4. Fused residual  — out-proj/FC2 use sgemm beta=1 into seq_data   │
//! │ 5. Zero alloc      — all buffers pre-allocated in Scratch           │
//! │ 6. No proj alloc   — eliminated per-position vec![0.0; D] (×400×4) │
//! └──────────────────────────────────────────────────────────────────────┘
//!
//! Memory: ~17 MB (7 MB INT8 + 2 MB f32 + 2.3 MB dequant buf + 6 MB scratch)
//!
//! Cargo.toml: add `matrixmultiply = "0.3"`
//! .cargo/config.toml:
//!   [build]
//!   rustflags = ["-C", "target-cpu=native"]

use std::io::Read;

const D: usize = 384;
const HEADS: usize = 6;
const HD: usize = 64;
const FF: usize = 1536;
const SEQ: usize = 400;
const N_LAYERS: usize = 4;
const POOL_DIM: usize = 256;
const CLS_MID: usize = 256;
const CLS_SMALL: usize = 64;

const LN_EPS: f32 = 1e-5;
/// 1/sqrt(head_dim) = 1/sqrt(64) = 0.125.
/// Original applied sqrt(0.125) to both Q and K; we fold the product into
/// a single GEMM alpha for the Q@K^T score computation.
const ATTN_SCALE_SQ: f32 = 0.125;

/// Return the default weights path resolved at runtime.
pub fn default_weights_path() -> std::path::PathBuf {
    crate::utils::cache::smart_turn_weights_path()
}

// ── Quantized weight block (INT8 in memory) ─────────────────────────────

struct QWeight {
    data: Vec<i8>,
    scale: Vec<f32>,
    zp: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl QWeight {
    #[inline]
    fn row(&self, r: usize) -> &[i8] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
}

// ── Per-transformer-layer INT8 weight storage ───────────────────────────

struct LayerQ {
    aln_w: usize, aln_b: usize,
    q: QWeight, q_b: usize,
    k: QWeight,
    v: QWeight, v_b: usize,
    out: QWeight, out_b: usize,
    fln_w: usize, fln_b: usize,
    fc1: QWeight, fc1_b: usize,
    fc2: QWeight, fc2_b: usize,
}

// ── Scratch buffers (zero allocation in hot path) ───────────────────────

struct Scratch {
    ln_buf:      Vec<f32>,  // SEQ * D
    q:           Vec<f32>,  // SEQ * D
    k:           Vec<f32>,  // SEQ * D
    v:           Vec<f32>,  // SEQ * D
    attn_out:    Vec<f32>,  // SEQ * D
    scores:      Vec<f32>,  // SEQ * SEQ
    ln2:         Vec<f32>,  // SEQ * D
    ff:          Vec<f32>,  // SEQ * FF
    dq_mat:      Vec<f32>,  // max(D*D, D*FF, FF*D) = D*FF — batch dequant buffer
    pool_hidden: Vec<f32>,  // SEQ * POOL_DIM
    seq_data:    Vec<f32>,  // SEQ * D
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
            dq_mat:      vec![0.0; D * FF],  // 384×1536 = largest weight matrix
            pool_hidden: vec![0.0; SEQ * POOL_DIM],
            seq_data:    vec![0.0; SEQ * D],
            energies:    vec![0.0; SEQ],
            pooled:      vec![0.0; D],
            cls_mid:     vec![0.0; CLS_MID],
            cls_small:   vec![0.0; CLS_SMALL],
        }
    }
}

// ── Engine ──────────────────────────────────────────────────────────────

pub struct SmartTurnEngine {
    f32_data: Vec<f32>,
    layers: Vec<LayerQ>,
    conv1_w: usize, conv1_b: usize,
    conv2_w: usize, conv2_b: usize,
    pos_emb: usize,
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
    /// Downloads the weights on first use if not already cached.
    pub fn new(weights_path: Option<&str>) -> Result<Self, String> {
        let path = weights_path
            .map(|p| p.to_string())
            .unwrap_or_else(|| default_weights_path().to_string_lossy().into_owned());
        if weights_path.is_none() {
            crate::utils::cache::ensure_model(
                std::path::Path::new(&path),
                crate::utils::cache::SMART_TURN_URL,
                "smart_turn_weights.bin.gz",
            )?;
        }
        let weights_bytes = std::fs::read(&path)
            .map_err(|e| format!("Failed to read weights from {}: {}", path, e))?;
        let mut decoder = flate2::read::GzDecoder::new(&weights_bytes[..]);
        let mut raw = Vec::new();
        decoder.read_to_end(&mut raw)
            .map_err(|e| format!("Decompress failed: {}", e))?;

        let mut r = BinReader::new(&raw);
        let mut f32_data: Vec<f32> = Vec::new();

        let mut f32_off = |data: &[f32]| -> usize {
            let off = f32_data.len();
            f32_data.extend_from_slice(data);
            off
        };

        // ── Conv (dequant to f32 at init — small weights) ──
        let (c1w, c1s, c1zp) = r.read_quant(384 * 80 * 3, 384);
        let c1b = r.read_f32_vec(384);
        let conv1_w = f32_off(&dequant_ax0(&c1w, &c1s, &c1zp, 384, 80 * 3));
        let conv1_b = f32_off(&c1b);

        let (c2w, c2s, c2zp) = r.read_quant(384 * 384 * 3, 384);
        let c2b = r.read_f32_vec(384);
        let conv2_w = f32_off(&dequant_ax0(&c2w, &c2s, &c2zp, 384, 384 * 3));
        let conv2_b = f32_off(&c2b);

        let pos_emb = f32_off(&r.read_f32_vec(SEQ * D));

        // ── Transformer layers (keep INT8) ──
        let mut layers = Vec::with_capacity(N_LAYERS);
        for _ in 0..N_LAYERS {
            let aln_w = f32_off(&r.read_f32_vec(D));
            let aln_b = f32_off(&r.read_f32_vec(D));

            let q = r.read_qweight(D, D);
            let q_b = f32_off(&r.read_f32_vec(D));
            let k = r.read_qweight(D, D);
            let v = r.read_qweight(D, D);
            let v_b = f32_off(&r.read_f32_vec(D));
            let out = r.read_qweight(D, D);
            let out_b = f32_off(&r.read_f32_vec(D));

            let fln_w = f32_off(&r.read_f32_vec(D));
            let fln_b = f32_off(&r.read_f32_vec(D));

            let fc1 = r.read_qweight(D, FF);
            let fc1_b = f32_off(&r.read_f32_vec(FF));
            let fc2 = r.read_qweight(FF, D);
            let fc2_b = f32_off(&r.read_f32_vec(D));

            layers.push(LayerQ {
                aln_w, aln_b, q, q_b, k, v, v_b, out, out_b,
                fln_w, fln_b, fc1, fc1_b, fc2, fc2_b,
            });
        }

        // ── Final LN ──
        let fln_w = f32_off(&r.read_f32_vec(D));
        let fln_b = f32_off(&r.read_f32_vec(D));

        // ── Pool (dequant to f32 — small) ──
        let (pw, ps, pzp) = r.read_quant(D * POOL_DIM, POOL_DIM);
        let pool0_w = f32_off(&dequant_ax1(&pw, &ps, &pzp, D, POOL_DIM));
        let pool0_b = f32_off(&r.read_f32_vec(POOL_DIM));
        let (pw2, ps2, pzp2) = r.read_quant(POOL_DIM, 1);
        let pool2_w = f32_off(&dequant_ax1(&pw2, &ps2, &pzp2, POOL_DIM, 1));
        let pool2_b = f32_off(&r.read_f32_vec(1));

        // ── Classifier (dequant to f32 — small, axis=0) ──
        let (cw, cs, czp) = r.read_quant(CLS_MID * D, CLS_MID);
        let cls0_w = f32_off(&dequant_ax0(&cw, &cs, &czp, CLS_MID, D));
        let cls0_b = f32_off(&r.read_f32_vec(CLS_MID));
        let cls_ln_w = f32_off(&r.read_f32_vec(CLS_MID));
        let cls_ln_b = f32_off(&r.read_f32_vec(CLS_MID));
        let (cw4, cs4, czp4) = r.read_quant(CLS_SMALL * CLS_MID, CLS_SMALL);
        let cls4_w = f32_off(&dequant_ax0(&cw4, &cs4, &czp4, CLS_SMALL, CLS_MID));
        let cls4_b = f32_off(&r.read_f32_vec(CLS_SMALL));
        let (cw6, cs6, czp6) = r.read_quant(CLS_SMALL, 1);
        let cls6_w = f32_off(&dequant_ax0(&cw6, &cs6, &czp6, 1, CLS_SMALL));
        let cls6_b = f32_off(&r.read_f32_vec(1));

        let i8_total: usize = layers.iter().map(|l|
            l.q.data.len() + l.k.data.len() + l.v.data.len() +
            l.out.data.len() + l.fc1.data.len() + l.fc2.data.len()
        ).sum();

        log::info!(
            "SmartTurnEngine: INT8 {:.1} MB, f32 {:.1} MB, scratch {:.1} MB",
            i8_total as f64 / 1024.0 / 1024.0,
            f32_data.len() as f64 * 4.0 / 1024.0 / 1024.0,
            std::mem::size_of::<Scratch>() as f64 / 1024.0 / 1024.0,
        );

        Ok(Self {
            f32_data, layers,
            conv1_w, conv1_b, conv2_w, conv2_b, pos_emb,
            fln_w, fln_b,
            pool0_w, pool0_b, pool2_w, pool2_b,
            cls0_w, cls0_b, cls_ln_w, cls_ln_b,
            cls4_w, cls4_b, cls6_w, cls6_b,
            scratch: Scratch::new(),
        })
    }

    pub fn infer(&mut self, features: &[f32]) -> f32 {
        debug_assert_eq!(features.len(), 80 * 800);

        // ── Conv1 + GELU ────────────────────────────────────────────
        let mut x = conv1d_k3(
            features, 80, 800,
            &self.f32_data[self.conv1_w..], &self.f32_data[self.conv1_b..],
            384, 1, 1,
        );
        gelu_inplace(&mut x);

        // ── Conv2 + GELU ────────────────────────────────────────────
        x = conv1d_k3(
            &x, 384, 800,
            &self.f32_data[self.conv2_w..], &self.f32_data[self.conv2_b..],
            384, 1, 2,
        );
        gelu_inplace(&mut x);

        // ── Transpose [384, 400] → [400, 384] + pos embeddings ─────
        let pos = &self.f32_data[self.pos_emb..self.pos_emb + SEQ * D];
        let seq = &mut self.scratch.seq_data;
        for s in 0..SEQ {
            for d in 0..D {
                seq[s * D + d] = x[d * SEQ + s] + pos[s * D + d];
            }
        }

        // ── 4 Transformer layers ────────────────────────────────────
        for i in 0..N_LAYERS {
            self.transformer_layer(i);
        }

        // ── Final LayerNorm ─────────────────────────────────────────
        let fln_w = &self.f32_data[self.fln_w..self.fln_w + D];
        let fln_b = &self.f32_data[self.fln_b..self.fln_b + D];
        for s in 0..SEQ {
            layer_norm_inplace(
                &mut self.scratch.seq_data[s * D..(s + 1) * D],
                fln_w, fln_b,
            );
        }

        // ── Attention Pooling (GEMM-based) ──────────────────────────
        // pool_hidden[SEQ, POOL_DIM] = seq_data[SEQ, D] @ pool0_w[D, POOL_DIM] + bias
        let pool0_b = &self.f32_data[self.pool0_b..self.pool0_b + POOL_DIM];
        for s in 0..SEQ {
            self.scratch.pool_hidden[s * POOL_DIM..(s + 1) * POOL_DIM]
                .copy_from_slice(pool0_b);
        }
        unsafe {
            matrixmultiply::sgemm(
                SEQ, D, POOL_DIM,
                1.0,
                self.scratch.seq_data.as_ptr(), D as isize, 1,
                self.f32_data.as_ptr().add(self.pool0_w), POOL_DIM as isize, 1,
                1.0,
                self.scratch.pool_hidden.as_mut_ptr(), POOL_DIM as isize, 1,
            );
        }

        // tanh + energy scores
        let pool2_w = &self.f32_data[self.pool2_w..self.pool2_w + POOL_DIM];
        let pool2_b = self.f32_data[self.pool2_b];
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

        // ── Classifier (pre-dequanted f32) ──────────────────────────
        let cls0_w = &self.f32_data[self.cls0_w..self.cls0_w + CLS_MID * D];
        self.scratch.cls_mid.copy_from_slice(
            &self.f32_data[self.cls0_b..self.cls0_b + CLS_MID],
        );
        for n in 0..CLS_MID {
            self.scratch.cls_mid[n] += dot(
                &cls0_w[n * D..(n + 1) * D],
                &self.scratch.pooled,
            );
        }
        layer_norm_inplace(
            &mut self.scratch.cls_mid,
            &self.f32_data[self.cls_ln_w..self.cls_ln_w + CLS_MID],
            &self.f32_data[self.cls_ln_b..self.cls_ln_b + CLS_MID],
        );
        gelu_inplace(&mut self.scratch.cls_mid);

        let cls4_w = &self.f32_data[self.cls4_w..self.cls4_w + CLS_SMALL * CLS_MID];
        self.scratch.cls_small.copy_from_slice(
            &self.f32_data[self.cls4_b..self.cls4_b + CLS_SMALL],
        );
        for n in 0..CLS_SMALL {
            self.scratch.cls_small[n] += dot(
                &cls4_w[n * CLS_MID..(n + 1) * CLS_MID],
                &self.scratch.cls_mid,
            );
        }
        gelu_inplace(&mut self.scratch.cls_small);

        let cls6_w = &self.f32_data[self.cls6_w..self.cls6_w + CLS_SMALL];
        let cls6_b = self.f32_data[self.cls6_b];
        sigmoid(cls6_b + dot(cls6_w, &self.scratch.cls_small))
    }

    fn transformer_layer(&mut self, layer_idx: usize) {
        // ── LayerNorm ───────────────────────────────────────────────
        {
            let l = &self.layers[layer_idx];
            let aln_w = &self.f32_data[l.aln_w..l.aln_w + D];
            let aln_b = &self.f32_data[l.aln_b..l.aln_b + D];

            self.scratch.ln_buf.copy_from_slice(&self.scratch.seq_data);
            for s in 0..SEQ {
                layer_norm_inplace(
                    &mut self.scratch.ln_buf[s * D..(s + 1) * D],
                    aln_w, aln_b,
                );
            }
        }

        // ── QKV Projection: batch dequant → sgemm ──────────────────
        //
        // Previous: SEQ×D dequant_axpy calls per matrix (each element
        //   dequantized SEQ=400 times). ~460K calls for Q+K+V.
        // Now: 1 dequant + 1 sgemm per matrix. Each INT8 weight
        //   dequantized exactly once.

        // Q = LN @ dequant(W_q) + bias_q
        {
            let q_b = &self.f32_data[self.layers[layer_idx].q_b..self.layers[layer_idx].q_b + D];
            for s in 0..SEQ {
                self.scratch.q[s * D..(s + 1) * D].copy_from_slice(q_b);
            }
        }
        dequant_to_f32(&self.layers[layer_idx].q, &mut self.scratch.dq_mat[..D * D]);
        unsafe {
            matrixmultiply::sgemm(
                SEQ, D, D,
                1.0,
                self.scratch.ln_buf.as_ptr(), D as isize, 1,
                self.scratch.dq_mat.as_ptr(), D as isize, 1,
                1.0,
                self.scratch.q.as_mut_ptr(), D as isize, 1,
            );
        }

        // K = LN @ dequant(W_k)  (no bias)
        dequant_to_f32(&self.layers[layer_idx].k, &mut self.scratch.dq_mat[..D * D]);
        unsafe {
            matrixmultiply::sgemm(
                SEQ, D, D,
                1.0,
                self.scratch.ln_buf.as_ptr(), D as isize, 1,
                self.scratch.dq_mat.as_ptr(), D as isize, 1,
                0.0,
                self.scratch.k.as_mut_ptr(), D as isize, 1,
            );
        }

        // V = LN @ dequant(W_v) + bias_v
        {
            let v_b = &self.f32_data[self.layers[layer_idx].v_b..self.layers[layer_idx].v_b + D];
            for s in 0..SEQ {
                self.scratch.v[s * D..(s + 1) * D].copy_from_slice(v_b);
            }
        }
        dequant_to_f32(&self.layers[layer_idx].v, &mut self.scratch.dq_mat[..D * D]);
        unsafe {
            matrixmultiply::sgemm(
                SEQ, D, D,
                1.0,
                self.scratch.ln_buf.as_ptr(), D as isize, 1,
                self.scratch.dq_mat.as_ptr(), D as isize, 1,
                1.0,
                self.scratch.v.as_mut_ptr(), D as isize, 1,
            );
        }

        // ── Multi-Head Attention (on f32 Q/K/V) ────────────────────
        for h in 0..HEADS {
            let ho = h * HD;

            // scores[SEQ,SEQ] = Q_h[SEQ,HD] @ K_h[SEQ,HD]^T * (1/sqrt(d_k))
            unsafe {
                matrixmultiply::sgemm(
                    SEQ, HD, SEQ,
                    ATTN_SCALE_SQ,
                    self.scratch.q.as_ptr().add(ho), D as isize, 1,
                    self.scratch.k.as_ptr().add(ho), 1, D as isize,
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
                    self.scratch.v.as_ptr().add(ho), D as isize, 1,
                    0.0,
                    self.scratch.attn_out.as_mut_ptr().add(ho), D as isize, 1,
                );
            }
        }

        // ── Output Projection: dequant → sgemm fused with residual ─
        // seq_data += attn_out @ dequant(W_out) + out_bias
        {
            let out_b = &self.f32_data[self.layers[layer_idx].out_b..self.layers[layer_idx].out_b + D];
            for s in 0..SEQ {
                let row = &mut self.scratch.seq_data[s * D..(s + 1) * D];
                for i in 0..D { row[i] += out_b[i]; }
            }
        }
        dequant_to_f32(&self.layers[layer_idx].out, &mut self.scratch.dq_mat[..D * D]);
        unsafe {
            matrixmultiply::sgemm(
                SEQ, D, D,
                1.0,
                self.scratch.attn_out.as_ptr(), D as isize, 1,
                self.scratch.dq_mat.as_ptr(), D as isize, 1,
                1.0, // accumulate into seq_data (residual + bias preserved)
                self.scratch.seq_data.as_mut_ptr(), D as isize, 1,
            );
        }

        // ── Feed-Forward Network ────────────────────────────────────
        {
            let l = &self.layers[layer_idx];
            let fln_w = &self.f32_data[l.fln_w..l.fln_w + D];
            let fln_b = &self.f32_data[l.fln_b..l.fln_b + D];

            self.scratch.ln2.copy_from_slice(&self.scratch.seq_data);
            for s in 0..SEQ {
                layer_norm_inplace(
                    &mut self.scratch.ln2[s * D..(s + 1) * D],
                    fln_w, fln_b,
                );
            }
        }

        // FC1: ff[SEQ,FF] = LN2[SEQ,D] @ dequant(W_fc1)[D,FF] + bias + GELU
        {
            let fc1_b = &self.f32_data[self.layers[layer_idx].fc1_b..self.layers[layer_idx].fc1_b + FF];
            for s in 0..SEQ {
                self.scratch.ff[s * FF..(s + 1) * FF].copy_from_slice(fc1_b);
            }
        }
        dequant_to_f32(&self.layers[layer_idx].fc1, &mut self.scratch.dq_mat[..D * FF]);
        unsafe {
            matrixmultiply::sgemm(
                SEQ, D, FF,
                1.0,
                self.scratch.ln2.as_ptr(), D as isize, 1,
                self.scratch.dq_mat.as_ptr(), FF as isize, 1,
                1.0,
                self.scratch.ff.as_mut_ptr(), FF as isize, 1,
            );
        }
        gelu_inplace(&mut self.scratch.ff[..SEQ * FF]);

        // FC2: seq_data += ff[SEQ,FF] @ dequant(W_fc2)[FF,D] + bias (fused residual)
        {
            let fc2_b = &self.f32_data[self.layers[layer_idx].fc2_b..self.layers[layer_idx].fc2_b + D];
            for s in 0..SEQ {
                let row = &mut self.scratch.seq_data[s * D..(s + 1) * D];
                for i in 0..D { row[i] += fc2_b[i]; }
            }
        }
        dequant_to_f32(&self.layers[layer_idx].fc2, &mut self.scratch.dq_mat[..FF * D]);
        unsafe {
            matrixmultiply::sgemm(
                SEQ, FF, D,
                1.0,
                self.scratch.ff.as_ptr(), FF as isize, 1,
                self.scratch.dq_mat.as_ptr(), D as isize, 1,
                1.0,
                self.scratch.seq_data.as_mut_ptr(), D as isize, 1,
            );
        }
    }
}

// ── Batch dequantize INT8 → f32 ────────────────────────────────────────
//
// Dequantizes an entire weight matrix at once into a pre-allocated buffer.
// With -C target-cpu=native, the f32 arithmetic auto-vectorizes.
//
// Cost: ~200μs for the largest matrix (384×1536). Total per layer: ~1.2ms.
// This replaces ~236M redundant dequant ops (FC1 alone) with ~590K.

fn dequant_to_f32(qw: &QWeight, out: &mut [f32]) {
    let cols = qw.cols;
    let scale = &qw.scale;
    let zp = &qw.zp;
    for r in 0..qw.rows {
        let row = qw.row(r);
        let base = r * cols;
        for c in 0..cols {
            unsafe {
                let w = *row.get_unchecked(c) as f32;
                let z = *zp.get_unchecked(c);
                let s = *scale.get_unchecked(c);
                *out.get_unchecked_mut(base + c) = (w - z) * s;
            }
        }
    }
}

// ── Binary reader (unchanged) ───────────────────────────────────────────

struct BinReader<'a> {
    data: &'a [u8],
    off: usize,
}

impl<'a> BinReader<'a> {
    fn new(data: &'a [u8]) -> Self { Self { data, off: 0 } }

    fn align4(&mut self) { while self.off % 4 != 0 { self.off += 1; } }

    fn read_i8_vec(&mut self, n: usize) -> Vec<i8> {
        let slice = &self.data[self.off..self.off + n];
        let v: Vec<i8> = slice.iter().map(|&b| b as i8).collect();
        self.off += n;
        self.align4();
        v
    }

    fn read_f32_vec(&mut self, n: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            let b = &self.data[self.off..self.off + 4];
            v.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
            self.off += 4;
        }
        v
    }

    fn read_quant(&mut self, n_elements: usize, n_channels: usize) -> (Vec<i8>, Vec<f32>, Vec<f32>) {
        let w = self.read_i8_vec(n_elements);
        let scale = self.read_f32_vec(n_channels);
        let zp_i8 = self.read_i8_vec(n_channels);
        let zp_f32: Vec<f32> = zp_i8.iter().map(|&v| v as f32).collect();
        (w, scale, zp_f32)
    }

    fn read_qweight(&mut self, rows: usize, cols: usize) -> QWeight {
        let (data, scale, zp) = self.read_quant(rows * cols, cols);
        QWeight { data, scale, zp, rows, cols }
    }
}

// ── Dequantize helpers (init-time, unchanged) ───────────────────────────

fn dequant_ax0(w: &[i8], scale: &[f32], zp: &[f32], n_ch: usize, inner: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_ch * inner];
    for ch in 0..n_ch {
        let s = scale[ch];
        let z = zp[ch];
        for i in 0..inner {
            out[ch * inner + i] = (w[ch * inner + i] as f32 - z) * s;
        }
    }
    out
}

fn dequant_ax1(w: &[i8], scale: &[f32], zp: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] = (w[r * cols + c] as f32 - zp[c]) * scale[c];
        }
    }
    out
}

// ── Small vector ops (classifier + pooling) ─────────────────────────────

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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

// ── Scalar ops (unchanged) ──────────────────────────────────────────────

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
