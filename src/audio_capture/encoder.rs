use crate::error::{PipecatError, Result};

/// Encode raw PCM bytes to WAV format using the `hound` crate.
///
/// `pcm` must be 16-bit signed little-endian samples (the standard format
/// used throughout the pipeline via `AudioRawData`).
pub fn encode_pcm_to_wav(pcm: &[u8], sample_rate: u32, num_channels: u16) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels:        num_channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format:   hound::SampleFormat::Int,
    };

    let mut buf = std::io::Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut buf, spec)
            .map_err(|e| PipecatError::pipeline(format!("wav writer create: {e}")))?;

        // PCM is little-endian i16 — interpret 2 bytes at a time
        for chunk in pcm.chunks_exact(2) {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            writer.write_sample(sample)
                .map_err(|e| PipecatError::pipeline(format!("wav write sample: {e}")))?;
        }

        writer.finalize()
            .map_err(|e| PipecatError::pipeline(format!("wav finalize: {e}")))?;
    }

    Ok(buf.into_inner())
}

/// Compute audio duration in milliseconds from raw PCM bytes.
pub fn pcm_duration_ms(pcm_bytes: usize, sample_rate: u32, num_channels: u16) -> f64 {
    if sample_rate == 0 || num_channels == 0 {
        return 0.0;
    }
    let bytes_per_sample = 2u32; // 16-bit
    let total_samples = pcm_bytes as f64 / (bytes_per_sample as f64 * num_channels as f64);
    (total_samples / sample_rate as f64) * 1000.0
}

/// Resample mono 16-bit PCM from `from_rate` to `to_rate` using linear interpolation.
pub fn resample_pcm(pcm: &[u8], from_rate: u32, to_rate: u32) -> Vec<u8> {
    if from_rate == to_rate || pcm.is_empty() {
        return pcm.to_vec();
    }
    let samples_in: Vec<i16> = pcm
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    let n_in = samples_in.len();
    let n_out = ((n_in as f64) * (to_rate as f64) / (from_rate as f64)).ceil() as usize;
    let mut out = Vec::with_capacity(n_out * 2);
    for i in 0..n_out {
        let src = i as f64 * from_rate as f64 / to_rate as f64;
        let lo = src as usize;
        let hi = (lo + 1).min(n_in.saturating_sub(1));
        let frac = src - lo as f64;
        let s = (samples_in[lo] as f64 + frac * (samples_in[hi] as f64 - samples_in[lo] as f64))
            .round()
            .clamp(i16::MIN as f64, i16::MAX as f64) as i16;
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Downmix multi-channel 16-bit PCM to mono by averaging channels.
pub fn downmix_to_mono(pcm: &[u8], num_channels: u16) -> Vec<u8> {
    if num_channels <= 1 || pcm.is_empty() {
        return pcm.to_vec();
    }
    let ch = num_channels as usize;
    let samples: Vec<i16> = pcm.chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    let mut mono = Vec::with_capacity(samples.len() / ch * 2);
    for frame in samples.chunks_exact(ch) {
        let avg = (frame.iter().map(|&s| s as i32).sum::<i32>() / ch as i32)
            .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        mono.extend_from_slice(&avg.to_le_bytes());
    }
    mono
}
