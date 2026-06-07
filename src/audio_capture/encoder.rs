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
