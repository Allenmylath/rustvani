//! G.711 µ-law (PCMU) codec plus resample helpers for telephony serializers.
//!
//! Mirrors pipecat's `pipecat.audio.utils.pcm_to_ulaw` / `ulaw_to_pcm`, which
//! wrap `audioop.lin2ulaw` / `audioop.ulaw2lin` (sample width 2) around a
//! streaming resampler. As in pipecat, encode order is **resample → encode**
//! and decode order is **decode → resample**.
//!
//! Audio is 16-bit little-endian mono PCM throughout, matching
//! [`AudioRawData`](crate::frames::AudioRawData).

use crate::audio_process::resamplers::StreamResampler;

// ---------------------------------------------------------------------------
// G.711 µ-law sample codec (ITU-T)
// ---------------------------------------------------------------------------

/// Encode one 16-bit PCM sample to a µ-law byte.
///
/// ITU-T G.711 µ-law, matching Python `audioop.lin2ulaw(..., 2)` exactly —
/// the encoding Twilio and the PSTN expect. Note this replicates audioop's
/// 16-bit → 14-bit right shift (`>> 2`) and scaled bias (`BIAS >> 2`); a codec
/// that skips those produces a non-standard, DC-offset result (e.g. silence
/// encodes to `0xDB` instead of `0xFF`).
pub fn linear_to_ulaw(sample: i16) -> u8 {
    const BIAS: i32 = 0x84;
    const CLIP: i32 = 8159;
    // Upper bound of each µ-law segment, on the 14-bit + bias scale.
    const SEG_UEND: [i32; 8] = [0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF];

    let mut pcm = (sample as i32) >> 2; // 16-bit -> 14-bit, as audioop does
    let mask;

    if pcm < 0 {
        pcm = -pcm;
        mask = 0x7F;
    } else {
        mask = 0xFF;
    }

    if pcm > CLIP {
        pcm = CLIP;
    }
    pcm += BIAS >> 2;

    let mut seg = 8;
    for (i, &end) in SEG_UEND.iter().enumerate() {
        if pcm <= end {
            seg = i;
            break;
        }
    }

    if seg >= 8 {
        (0x7F ^ mask) as u8
    } else {
        let uval = ((seg as i32) << 4) | ((pcm >> (seg as i32 + 1)) & 0x0F);
        (uval ^ mask) as u8
    }
}

/// Decode one µ-law byte to a 16-bit PCM sample.
///
/// ITU-T G.711 µ-law expand (the classic Sun reference), matching Python
/// `audioop.ulaw2lin(..., 2)` and the exact inverse of [`linear_to_ulaw`].
pub fn ulaw_to_linear(ulaw: u8) -> i16 {
    const EXP_LUT: [i32; 8] = [0, 132, 396, 924, 1980, 4092, 8316, 16764];

    let u = !ulaw;
    let sign = u & 0x80;
    let exponent = ((u >> 4) & 0x07) as i32;
    let mantissa = (u & 0x0F) as i32;
    let sample = EXP_LUT[exponent as usize] + (mantissa << (exponent + 3));

    if sign != 0 { (-sample) as i16 } else { sample as i16 }
}

// ---------------------------------------------------------------------------
// PCM <-> byte helpers (16-bit little-endian)
// ---------------------------------------------------------------------------

fn bytes_to_i16(audio: &[u8]) -> Vec<i16> {
    audio.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
}

fn i16_to_bytes(samples: &[i16]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

fn i16_to_f32(samples: &[i16]) -> Vec<f32> {
    samples.iter().map(|&s| s as f32).collect()
}

fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&s| s.clamp(i16::MIN as f32, i16::MAX as f32) as i16)
        .collect()
}

// ---------------------------------------------------------------------------
// Resample + companding helpers
// ---------------------------------------------------------------------------

/// Convert 16-bit PCM to µ-law, resampling first when a resampler is given.
///
/// Pass `resampler = Some(..)` when the pipeline rate differs from the telephony
/// rate; pass `None` when they are equal (mirrors pipecat's SOXR `in == out`
/// no-op and skips the resampler's buffering latency). The resampler must be
/// configured for `pipeline_rate -> telephony_rate`.
pub fn pcm_to_ulaw(pcm: &[u8], resampler: Option<&mut StreamResampler>) -> Vec<u8> {
    let samples = bytes_to_i16(pcm);

    let resampled = match resampler {
        Some(r) => f32_to_i16(&r.process(&i16_to_f32(&samples))),
        None => samples,
    };

    resampled.iter().map(|&s| linear_to_ulaw(s)).collect()
}

/// Convert µ-law to 16-bit PCM, resampling after decoding when a resampler is
/// given.
///
/// Pass `resampler = Some(..)` when the telephony rate differs from the pipeline
/// rate; pass `None` when they are equal. The resampler must be configured for
/// `telephony_rate -> pipeline_rate`.
pub fn ulaw_to_pcm(ulaw: &[u8], resampler: Option<&mut StreamResampler>) -> Vec<u8> {
    let samples: Vec<i16> = ulaw.iter().map(|&b| ulaw_to_linear(b)).collect();

    match resampler {
        Some(r) => i16_to_bytes(&f32_to_i16(&r.process(&i16_to_f32(&samples)))),
        None => i16_to_bytes(&samples),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference vectors from Python `audioop.lin2ulaw(struct.pack('<h', x), 2)`.
    #[test]
    fn linear_to_ulaw_reference_vectors() {
        assert_eq!(linear_to_ulaw(0), 0xFF);
        assert_eq!(linear_to_ulaw(-1), 0x7E);
        assert_eq!(linear_to_ulaw(32767), 0x80);
        assert_eq!(linear_to_ulaw(-32768), 0x00);
        assert_eq!(linear_to_ulaw(1000), 0xCE);
        assert_eq!(linear_to_ulaw(-1000), 0x4E);
    }

    // Reference vectors from Python `audioop.ulaw2lin(bytes([x]), 2)`.
    #[test]
    fn ulaw_to_linear_reference_vectors() {
        assert_eq!(ulaw_to_linear(0xFF), 0);
        assert_eq!(ulaw_to_linear(0x80), 32124);
        assert_eq!(ulaw_to_linear(0x00), -32124);
    }

    // µ-law is lossy, but decode(encode(x)) must land in the same quantization
    // step and preserve sign — the standard G.711 round-trip guarantee.
    #[test]
    fn ulaw_round_trip_is_stable() {
        for x in (-32768..=32767).step_by(97) {
            let x = x as i16;
            let once = ulaw_to_linear(linear_to_ulaw(x));
            let twice = ulaw_to_linear(linear_to_ulaw(once));
            assert_eq!(once, twice, "re-encoding a decoded sample must be idempotent");
            if x != 0 {
                assert_eq!(once.signum(), x.signum(), "sign must be preserved for {x}");
            }
        }
    }

    #[test]
    fn pcm_to_ulaw_no_resampler_is_pure_encode() {
        let pcm = i16_to_bytes(&[0, -1, 1000, -1000]);
        let ulaw = pcm_to_ulaw(&pcm, None);
        assert_eq!(ulaw, vec![0xFF, 0x7E, 0xCE, 0x4E]);
    }

    #[test]
    fn ulaw_to_pcm_no_resampler_round_trips_bytes() {
        let ulaw = vec![0xFF, 0x7F, 0xDE, 0x5E];
        let pcm = ulaw_to_pcm(&ulaw, None);
        let samples = bytes_to_i16(&pcm);
        assert_eq!(samples.len(), 4);
        assert_eq!(samples[0], ulaw_to_linear(0xFF));
        assert_eq!(samples[3], ulaw_to_linear(0x5E));
    }
}
