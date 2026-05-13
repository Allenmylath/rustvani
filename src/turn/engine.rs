//! Pure Rust smart-turn-v3 inference engine — INT8 weights, SIMD optimized.
//!
//! Matches Pipecat's 8 MB model size. Transformer layer weights stay INT8
//! in memory (~7 MB) with fused dequant+SIMD in the matmul inner loop.
//! Small weights (conv, pool, classifier) are dequantized to f32 at init.
//!
//! Binary: 7.2 MB gzip-compressed (embedded at compile time).
//! RAM: ~15 MB total (7 MB INT8 + 2 MB f32 + 6 MB scratch).

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
const ATTN_SCALE: f32 = 0.353_553_39;

const WEIGHTS_GZ: &[u8] = include_bytes!("smart_turn_weights.bin.gz");

// ── Quantized weight block (INT8 in memory) ─────────────────────────────

struct QWeight {
    data: Vec<i8>,     // INT8 weights, row-major
    scale: Vec<f32>,   // per-channel scale
    zp: Vec<f32>,      // per-channel zero_point (as f32 for SIMD)
    rows: usize,
    cols: usize,
}

impl QWeight {
    /// Get row `r` as an i8 slice.
    #[inline]
    fn row(&self, r: usize) -> &[i8] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
}

// ── Per-transformer-layer INT8 weight storage ───────────────────────────

struct LayerQ {
    aln_w: usize, aln_b: usize,   // offsets into f32_data
    q: QWeight, q_b: usize,
    k: QWeight,
    v: QWeight, v_b: usize,
    out: QWeight, out_b: usize,
    fln_w: usize, fln_b: usize,
    fc1: QWeight, fc1_b: usize,
    fc2: QWeight, fc2_b: usize,
}

// ── Scratch buffers ─────────────────────────────────────────────────────

struct Scratch {
    ln_buf:   Vec<f32>,  // SEQ * D
    q:        Vec<f32>,  // SEQ * D
    k:        Vec<f32>,  // SEQ * D
    v:        Vec<f32>,  // SEQ * D
    attn_out: Vec<f32>,  // SEQ * D
    scores:   Vec<f32>,  // SEQ * SEQ
    ln2:      Vec<f32>,  // SEQ * D
    ff:       Vec<f32>,  // SEQ * FF
    dq_row:   Vec<f32>,  // max(D, FF) — one dequantized row
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
            dq_row:   vec![0.0; FF],  // largest row width
        }
    }
}

// ── Engine ──────────────────────────────────────────────────────────────

pub struct SmartTurnEngine {
    f32_data: Vec<f32>,  // biases, layer norms, pos embeddings, dequanted conv/pool/cls
    layers: Vec<LayerQ>,
    // f32 offsets for dequanted weights
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
    pub fn new() -> Result<Self, String> {
        let mut decoder = flate2::read::GzDecoder::new(&WEIGHTS_GZ[..]);
        let mut raw = Vec::new();
        decoder.read_to_end(&mut raw)
            .map_err(|e| format!("Decompress failed: {}", e))?;

        let mut r = BinReader::new(&raw);
        let mut f32_data: Vec<f32> = Vec::new();

        // Helper: append f32 data and return starting offset
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
            "SmartTurnEngine: INT8 weights {:.1} MB, f32 data {:.1} MB, scratch {:.1} MB",
            i8_total as f64 / 1024.0 / 1024.0,
            f32_data.len() as f64 * 4.0 / 1024.0 / 1024.0,
            (SEQ * D * 5 + SEQ * SEQ + SEQ * FF + FF) as f64 * 4.0 / 1024.0 / 1024.0,
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

        // Conv1 + GELU
        let mut x = conv1d_k3(
            features, 80, 800,
            &self.f32_data[self.conv1_w..], &self.f32_data[self.conv1_b..],
            384, 1, 1,
        );
        gelu_inplace(&mut x);

        // Conv2 + GELU
        x = conv1d_k3(
            &x, 384, 800,
            &self.f32_data[self.conv2_w..], &self.f32_data[self.conv2_b..],
            384, 1, 2,
        );
        gelu_inplace(&mut x);

        // Transpose + pos embeddings
        let pos = &self.f32_data[self.pos_emb..self.pos_emb + SEQ * D];
        let mut seq_data = vec![0.0f32; SEQ * D];
        for s in 0..SEQ {
            for d in 0..D {
                seq_data[s * D + d] = x[d * SEQ + s] + pos[s * D + d];
            }
        }

        // Transformer layers
        for i in 0..N_LAYERS {
            self.transformer_layer(&mut seq_data, i);
        }

        // Final LayerNorm
        let fln_w = &self.f32_data[self.fln_w..self.fln_w + D];
        let fln_b = &self.f32_data[self.fln_b..self.fln_b + D];
        for s in 0..SEQ {
            layer_norm_inplace(&mut seq_data[s * D..(s + 1) * D], fln_w, fln_b);
        }

        // Attention pool (uses pre-dequanted f32 pool weights)
        let pool0_w = &self.f32_data[self.pool0_w..self.pool0_w + D * POOL_DIM];
        let pool0_b = &self.f32_data[self.pool0_b..self.pool0_b + POOL_DIM];
        let pool2_w = &self.f32_data[self.pool2_w..self.pool2_w + POOL_DIM];
        let pool2_b = self.f32_data[self.pool2_b];

        let mut energies = vec![0.0f32; SEQ];
        for s in 0..SEQ {
            let row = &seq_data[s * D..(s + 1) * D];
            let h = &mut self.scratch.dq_row[..POOL_DIM];
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

        // Classifier (pre-dequanted f32)
        let cls0_w = &self.f32_data[self.cls0_w..self.cls0_w + CLS_MID * D];
        let cls0_b = &self.f32_data[self.cls0_b..self.cls0_b + CLS_MID];
        let mut c = vec![0.0f32; CLS_MID];
        c.copy_from_slice(cls0_b);
        for n in 0..CLS_MID {
            c[n] += dot(&cls0_w[n * D..(n + 1) * D], &pooled);
        }
        layer_norm_inplace(
            &mut c,
            &self.f32_data[self.cls_ln_w..self.cls_ln_w + CLS_MID],
            &self.f32_data[self.cls_ln_b..self.cls_ln_b + CLS_MID],
        );
        gelu_inplace(&mut c);

        let cls4_w = &self.f32_data[self.cls4_w..self.cls4_w + CLS_SMALL * CLS_MID];
        let cls4_b = &self.f32_data[self.cls4_b..self.cls4_b + CLS_SMALL];
        let mut c2 = vec![0.0f32; CLS_SMALL];
        c2.copy_from_slice(cls4_b);
        for n in 0..CLS_SMALL {
            c2[n] += dot(&cls4_w[n * CLS_MID..(n + 1) * CLS_MID], &c);
        }
        gelu_inplace(&mut c2);

        let cls6_w = &self.f32_data[self.cls6_w..self.cls6_w + CLS_SMALL];
        let cls6_b = self.f32_data[self.cls6_b];
        sigmoid(cls6_b + dot(cls6_w, &c2))
    }

    fn transformer_layer(&mut self, x: &mut [f32], layer_idx: usize) {
        let l = &self.layers[layer_idx];
        let aln_w = &self.f32_data[l.aln_w..l.aln_w + D];
        let aln_b = &self.f32_data[l.aln_b..l.aln_b + D];

        // LayerNorm
        self.scratch.ln_buf.copy_from_slice(&x[..SEQ * D]);
        for s in 0..SEQ {
            layer_norm_inplace(&mut self.scratch.ln_buf[s * D..(s + 1) * D], aln_w, aln_b);
        }

        // ── Fused QKV with INT8 dequant ──
        let q_b = &self.f32_data[l.q_b..l.q_b + D];
        let v_b = &self.f32_data[l.v_b..l.v_b + D];
        for s in 0..SEQ {
            self.scratch.q[s * D..(s + 1) * D].copy_from_slice(q_b);
            self.scratch.v[s * D..(s + 1) * D].copy_from_slice(v_b);
        }
        self.scratch.k.iter_mut().for_each(|v| *v = 0.0);

        // Pointers to layer's quantized weights (avoid borrow conflict)
        let l = &self.layers[layer_idx];
        for s in 0..SEQ {
            let inp = &self.scratch.ln_buf[s * D..(s + 1) * D];
            let q_out = &mut self.scratch.q[s * D..(s + 1) * D];
            let k_out = &mut self.scratch.k[s * D..(s + 1) * D];
            let v_out = &mut self.scratch.v[s * D..(s + 1) * D];

            for d in 0..D {
                let val = inp[d];
                // Fused dequant + axpy: y += val * ((w_int8 - zp) * scale)
                dequant_axpy(val, l.q.row(d), &l.q.scale, &l.q.zp, q_out);
                dequant_axpy(val, l.k.row(d), &l.k.scale, &l.k.zp, k_out);
                dequant_axpy(val, l.v.row(d), &l.v.scale, &l.v.zp, v_out);
            }
        }

        // Scale Q, K
        for v in self.scratch.q.iter_mut() { *v *= ATTN_SCALE; }
        for v in self.scratch.k.iter_mut() { *v *= ATTN_SCALE; }

        // Multi-head attention
        self.scratch.attn_out.iter_mut().for_each(|v| *v = 0.0);
        for h in 0..HEADS {
            let ho = h * HD;
            for s1 in 0..SEQ {
                let q_slice = &self.scratch.q[s1 * D + ho..s1 * D + ho + HD];
                for s2 in 0..SEQ {
                    let k_slice = &self.scratch.k[s2 * D + ho..s2 * D + ho + HD];
                    self.scratch.scores[s1 * SEQ + s2] = dot(q_slice, k_slice);
                }
                softmax_inplace(&mut self.scratch.scores[s1 * SEQ..(s1 + 1) * SEQ]);
            }
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

        // Out projection (INT8 dequant) + residual
        let l = &self.layers[layer_idx];
        let out_b = &self.f32_data[l.out_b..l.out_b + D];
        for s in 0..SEQ {
            let x_row = &mut x[s * D..(s + 1) * D];
            // Add bias first (residual will be added on top)
            let mut proj = vec![0.0f32; D];
            proj.copy_from_slice(out_b);
            let inp = &self.scratch.attn_out[s * D..(s + 1) * D];
            for d in 0..D {
                dequant_axpy(inp[d], l.out.row(d), &l.out.scale, &l.out.zp, &mut proj);
            }
            for n in 0..D { x_row[n] += proj[n]; }
        }

        // ── FFN ──
        let l = &self.layers[layer_idx];
        let fln_w = &self.f32_data[l.fln_w..l.fln_w + D];
        let fln_b = &self.f32_data[l.fln_b..l.fln_b + D];
        let fc1_b = &self.f32_data[l.fc1_b..l.fc1_b + FF];
        let fc2_b = &self.f32_data[l.fc2_b..l.fc2_b + D];

        self.scratch.ln2.copy_from_slice(&x[..SEQ * D]);
        for s in 0..SEQ {
            layer_norm_inplace(&mut self.scratch.ln2[s * D..(s + 1) * D], fln_w, fln_b);
        }

        // FC1 (INT8 dequant)
        for s in 0..SEQ {
            let ff_row = &mut self.scratch.ff[s * FF..(s + 1) * FF];
            ff_row.copy_from_slice(fc1_b);
            let inp = &self.scratch.ln2[s * D..(s + 1) * D];
            for d in 0..D {
                dequant_axpy(inp[d], l.fc1.row(d), &l.fc1.scale, &l.fc1.zp, ff_row);
            }
        }
        gelu_inplace(&mut self.scratch.ff[..SEQ * FF]);

        // FC2 (INT8 dequant) + residual
        for s in 0..SEQ {
            let x_row = &mut x[s * D..(s + 1) * D];
            for n in 0..D { x_row[n] += fc2_b[n]; }
            let inp = &self.scratch.ff[s * FF..(s + 1) * FF];
            for d in 0..FF {
                dequant_axpy(inp[d], l.fc2.row(d), &l.fc2.scale, &l.fc2.zp, x_row);
            }
        }
    }
}

// ── Binary reader ───────────────────────────────────────────────────────

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

// ── Dequantize helpers (for init-time small weights) ────────────────────

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

// ── Fused dequant + axpy (the hot path) ─────────────────────────────────
// y[c] += a * ((w_i8[c] - zp[c]) * scale[c])   axis=1 layout

#[inline]
fn dequant_axpy(a: f32, w: &[i8], scale: &[f32], zp: &[f32], y: &mut [f32]) {
    let n = w.len();
    debug_assert_eq!(n, scale.len());
    debug_assert_eq!(n, zp.len());
    debug_assert_eq!(n, y.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { dequant_axpy_avx2(a, w, scale, zp, y); }
            return;
        }
    }
    for c in 0..n {
        y[c] += a * (w[c] as f32 - zp[c]) * scale[c];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_axpy_avx2(a: f32, w: &[i8], scale: &[f32], zp: &[f32], y: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = w.len();
    let wp = w.as_ptr();
    let sp = scale.as_ptr();
    let zp_ptr = zp.as_ptr();
    let yp = y.as_mut_ptr();
    let va = _mm256_set1_ps(a);

    let chunks8 = n / 8;
    for i in 0..chunks8 {
        let j = i * 8;
        // Load 8 INT8 → sign-extend to 8 INT32
        let w8 = _mm_loadl_epi64(wp.add(j) as *const __m128i);
        let w32 = _mm256_cvtepi8_epi32(w8);
        // Convert to f32
        let wf = _mm256_cvtepi32_ps(w32);
        // Subtract zero_point
        let zpf = _mm256_loadu_ps(zp_ptr.add(j));
        let dq = _mm256_sub_ps(wf, zpf);
        // Multiply by scale
        let sc = _mm256_loadu_ps(sp.add(j));
        let scaled = _mm256_mul_ps(dq, sc);
        // FMA: y += a * scaled
        let yi = _mm256_loadu_ps(yp.add(j));
        let result = _mm256_fmadd_ps(va, scaled, yi);
        _mm256_storeu_ps(yp.add(j), result);
    }

    // Tail
    let tail = chunks8 * 8;
    for c in tail..n {
        *yp.add(c) += a * (*wp.add(c) as f32 - *zp_ptr.add(c)) * *sp.add(c);
    }
}

// ── SIMD dot + axpy (for f32 weights) ───────────────────────────────────

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
        _mm256_storeu_ps(yp.add(j),      _mm256_fmadd_ps(va, _mm256_loadu_ps(xp.add(j)),      _mm256_loadu_ps(yp.add(j))));
        _mm256_storeu_ps(yp.add(j + 8),  _mm256_fmadd_ps(va, _mm256_loadu_ps(xp.add(j + 8)),  _mm256_loadu_ps(yp.add(j + 8))));
        _mm256_storeu_ps(yp.add(j + 16), _mm256_fmadd_ps(va, _mm256_loadu_ps(xp.add(j + 16)), _mm256_loadu_ps(yp.add(j + 16))));
        _mm256_storeu_ps(yp.add(j + 24), _mm256_fmadd_ps(va, _mm256_loadu_ps(xp.add(j + 24)), _mm256_loadu_ps(yp.add(j + 24))));
    }
    let done = chunks32 * 32;
    let chunks8 = (n - done) / 8;
    for i in 0..chunks8 {
        let j = done + i * 8;
        _mm256_storeu_ps(yp.add(j), _mm256_fmadd_ps(va, _mm256_loadu_ps(xp.add(j)), _mm256_loadu_ps(yp.add(j))));
    }
    let tail = done + chunks8 * 8;
    for i in tail..n { *yp.add(i) += a * *xp.add(i); }
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
