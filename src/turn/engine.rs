//! Pure Rust smart-turn-v3 inference engine — zero external dependencies.
//!
//! Weights are embedded at compile time (~31 MB added to binary).
//!
//! ```ignore
//! let engine = SmartTurnEngine::new()?;
//! let prob = engine.infer(&features_80x800);
//! ```

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
const ATTN_SCALE: f32 = 0.353_553_39; // 1/sqrt(8)

const WEIGHTS_BYTES: &[u8] = include_bytes!("smart_turn_weights.bin");

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
}

impl SmartTurnEngine {
    /// Create engine from embedded weights.
    pub fn new() -> Result<Self, String> {
        let w: Vec<f32> = WEIGHTS_BYTES
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        if w.len() < EXPECTED_FLOATS {
            return Err(format!(
                "Embedded weights too small: {} floats, expected {}",
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

        assert_eq!(o, EXPECTED_FLOATS, "Weight count mismatch");

        log::info!(
            "SmartTurnEngine: loaded {} floats ({:.1} MB) from embedded weights",
            w.len(), w.len() as f64 * 4.0 / 1024.0 / 1024.0
        );

        Ok(Self {
            w, conv1_w, conv1_b, conv2_w, conv2_b, pos_emb,
            layers, fln_w, fln_b,
            pool0_w, pool0_b, pool2_w, pool2_b,
            cls0_w, cls0_b, cls_ln_w, cls_ln_b,
            cls4_w, cls4_b, cls6_w, cls6_b,
        })
    }

    /// Run inference on mel features `[80, 800]` flat → probability `[0, 1]`.
    pub fn infer(&self, features: &[f32]) -> f32 {
        debug_assert_eq!(features.len(), 80 * 800);

        let mut x = conv1d(
            features, 80, 800,
            &self.w[self.conv1_w..], &self.w[self.conv1_b..],
            384, 3, 1, 1,
        );
        gelu_inplace(&mut x);

        x = conv1d(
            &x, 384, 800,
            &self.w[self.conv2_w..], &self.w[self.conv2_b..],
            384, 3, 1, 2,
        );
        gelu_inplace(&mut x);

        // Transpose [384, 400] → [400, 384] + positional embeddings
        let pos = &self.w[self.pos_emb..self.pos_emb + SEQ * D];
        let mut seq_data = vec![0.0f32; SEQ * D];
        for s in 0..SEQ {
            for d in 0..D {
                seq_data[s * D + d] = x[d * SEQ + s] + pos[s * D + d];
            }
        }

        for l in &self.layers {
            self.transformer_layer(&mut seq_data, l);
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
        for s in 0..SEQ {
            let row = &seq_data[s * D..(s + 1) * D];
            let mut energy = pool2_b;
            for n in 0..POOL_DIM {
                let mut val = pool0_b[n];
                for d in 0..D {
                    val += row[d] * pool0_w[d * POOL_DIM + n];
                }
                energy += val.tanh() * pool2_w[n];
            }
            energies[s] = energy;
        }
        softmax_inplace(&mut energies);

        let mut pooled = vec![0.0f32; D];
        for s in 0..SEQ {
            let w = energies[s];
            for d in 0..D {
                pooled[d] += seq_data[s * D + d] * w;
            }
        }

        // Classifier
        let cls0_w = &self.w[self.cls0_w..self.cls0_w + CLS_MID * D];
        let cls0_b = &self.w[self.cls0_b..self.cls0_b + CLS_MID];
        let mut c = vec![0.0f32; CLS_MID];
        for n in 0..CLS_MID {
            c[n] = cls0_b[n] + dot(&cls0_w[n * D..(n + 1) * D], &pooled);
        }
        layer_norm_inplace(
            &mut c,
            &self.w[self.cls_ln_w..self.cls_ln_w + CLS_MID],
            &self.w[self.cls_ln_b..self.cls_ln_b + CLS_MID],
        );
        gelu_inplace(&mut c);

        let cls4_w = &self.w[self.cls4_w..self.cls4_w + CLS_SMALL * CLS_MID];
        let cls4_b = &self.w[self.cls4_b..self.cls4_b + CLS_SMALL];
        let mut c2 = vec![0.0f32; CLS_SMALL];
        for n in 0..CLS_SMALL {
            c2[n] = cls4_b[n] + dot(&cls4_w[n * CLS_MID..(n + 1) * CLS_MID], &c);
        }
        gelu_inplace(&mut c2);

        let cls6_w = &self.w[self.cls6_w..self.cls6_w + CLS_SMALL];
        let cls6_b = self.w[self.cls6_b];
        sigmoid(cls6_b + dot(cls6_w, &c2))
    }

    fn transformer_layer(&self, x: &mut [f32], l: &LayerOffsets) {
        let aln_w = &self.w[l.aln_w..l.aln_w + D];
        let aln_b = &self.w[l.aln_b..l.aln_b + D];
        let q_w = &self.w[l.q_w..l.q_w + D * D];
        let q_b = &self.w[l.q_b..l.q_b + D];
        let k_w = &self.w[l.k_w..l.k_w + D * D];
        let v_w = &self.w[l.v_w..l.v_w + D * D];
        let v_b = &self.w[l.v_b..l.v_b + D];
        let out_w = &self.w[l.out_w..l.out_w + D * D];
        let out_b = &self.w[l.out_b..l.out_b + D];

        // LayerNorm
        let mut ln_buf = vec![0.0f32; SEQ * D];
        ln_buf.copy_from_slice(&x[..SEQ * D]);
        for s in 0..SEQ {
            layer_norm_inplace(&mut ln_buf[s * D..(s + 1) * D], aln_w, aln_b);
        }

        // Q, K, V projections + scale
        let mut q = vec![0.0f32; SEQ * D];
        let mut k = vec![0.0f32; SEQ * D];
        let mut v = vec![0.0f32; SEQ * D];

        for s in 0..SEQ {
            let inp = &ln_buf[s * D..(s + 1) * D];
            for n in 0..D {
                let mut sq = q_b[n];
                let mut sk = 0.0f32;
                let mut sv = v_b[n];
                for d in 0..D {
                    let val = inp[d];
                    sq += val * q_w[d * D + n];
                    sk += val * k_w[d * D + n];
                    sv += val * v_w[d * D + n];
                }
                q[s * D + n] = sq * ATTN_SCALE;
                k[s * D + n] = sk * ATTN_SCALE;
                v[s * D + n] = sv;
            }
        }

        // Multi-head attention
        let mut attn_out = vec![0.0f32; SEQ * D];
        let mut scores = vec![0.0f32; SEQ * SEQ];

        for h in 0..HEADS {
            let ho = h * HD;
            for s1 in 0..SEQ {
                for s2 in 0..SEQ {
                    let mut sum = 0.0f32;
                    for hd in 0..HD {
                        sum += q[s1 * D + ho + hd] * k[s2 * D + ho + hd];
                    }
                    scores[s1 * SEQ + s2] = sum;
                }
                softmax_inplace(&mut scores[s1 * SEQ..(s1 + 1) * SEQ]);
            }
            for s1 in 0..SEQ {
                for hd in 0..HD {
                    let mut sum = 0.0f32;
                    for s2 in 0..SEQ {
                        sum += scores[s1 * SEQ + s2] * v[s2 * D + ho + hd];
                    }
                    attn_out[s1 * D + ho + hd] = sum;
                }
            }
        }

        // Out projection + residual
        for s in 0..SEQ {
            let inp = &attn_out[s * D..(s + 1) * D];
            for n in 0..D {
                let mut sum = out_b[n];
                for d in 0..D {
                    sum += inp[d] * out_w[d * D + n];
                }
                x[s * D + n] += sum;
            }
        }

        // Feed-Forward
        let fln_w = &self.w[l.fln_w..l.fln_w + D];
        let fln_b = &self.w[l.fln_b..l.fln_b + D];
        let fc1_w = &self.w[l.fc1_w..l.fc1_w + D * FF];
        let fc1_b = &self.w[l.fc1_b..l.fc1_b + FF];
        let fc2_w = &self.w[l.fc2_w..l.fc2_w + FF * D];
        let fc2_b = &self.w[l.fc2_b..l.fc2_b + D];

        let mut ln2 = vec![0.0f32; SEQ * D];
        ln2.copy_from_slice(&x[..SEQ * D]);
        for s in 0..SEQ {
            layer_norm_inplace(&mut ln2[s * D..(s + 1) * D], fln_w, fln_b);
        }

        let mut ff = vec![0.0f32; SEQ * FF];
        for s in 0..SEQ {
            let inp = &ln2[s * D..(s + 1) * D];
            for n in 0..FF {
                let mut sum = fc1_b[n];
                for d in 0..D {
                    sum += inp[d] * fc1_w[d * FF + n];
                }
                ff[s * FF + n] = sum;
            }
        }
        gelu_inplace(&mut ff);

        for s in 0..SEQ {
            let inp = &ff[s * FF..(s + 1) * FF];
            for n in 0..D {
                let mut sum = fc2_b[n];
                for d in 0..FF {
                    sum += inp[d] * fc2_w[d * D + n];
                }
                x[s * D + n] += sum;
            }
        }
    }
}

// ── Ops ─────────────────────────────────────────────────────────────────

fn conv1d(
    x: &[f32], in_ch: usize, in_len: usize,
    weight: &[f32], bias: &[f32],
    out_ch: usize, k: usize, pad: usize, stride: usize,
) -> Vec<f32> {
    let padded_len = in_len + 2 * pad;
    let out_len = (padded_len - k) / stride + 1;
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
                let wb = (co * in_ch + ci) * k;
                let xb = ci * padded_len + ps;
                for ki in 0..k {
                    sum += weight[wb + ki] * padded[xb + ki];
                }
            }
            output[co * out_len + t] = sum;
        }
    }
    output
}

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

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() { sum += a[i] * b[i]; }
    sum
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
