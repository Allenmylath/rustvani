//! Opus codec + resampling glue for the `vaniwebrtc` transport.
//!
//! webrtc-rs carries RTP/SRTP but not the Opus codec itself, so we decode the
//! inbound Opus payloads → PCM and encode the outbound PCM → Opus here using
//! `audiopus` (libopus). Sample-rate conversion between WebRTC's fixed 48 kHz
//! and the pipeline rates reuses the existing
//! [`StreamResampler`](crate::audio_process::resamplers::StreamResampler).

use audiopus::{
    coder::{Decoder, Encoder},
    Application, Channels, SampleRate,
};

use crate::audio_process::resamplers::{ResamplerQuality, StreamResampler};

/// WebRTC always negotiates Opus at 48 kHz.
const OPUS_RATE: u32 = 48_000;
/// 20 ms Opus frame at 48 kHz, mono.
const OPUS_FRAME_SAMPLES: usize = 960;

// ---------------------------------------------------------------------------
// Denoiser48k — the DeepFilterNet insertion point
// ---------------------------------------------------------------------------

/// Optional full-band (48 kHz) denoiser applied to decoded inbound audio
/// *before* it is downsampled for the pipeline.
///
/// This is the DeepFilterNet insertion point: a future DFN backend implements
/// this trait and is plugged in per-connection via
/// [`VaniWebRTCParams::denoiser_factory`](super::params::VaniWebRTCParams).
/// v1 ships no implementation, so the inbound path is a transparent
/// pass-through.
pub trait Denoiser48k: Send {
    /// Denoise a chunk of 48 kHz mono i16 PCM. May buffer internally and return
    /// fewer samples than given.
    fn process(&mut self, pcm_48k: &[i16]) -> Vec<i16>;
    /// Flush any buffered tail at end of utterance.
    fn flush(&mut self) -> Vec<i16>;
    /// Reset internal state (e.g. on interruption / new utterance).
    fn reset(&mut self);
}

// ---------------------------------------------------------------------------
// Inbound: remote Opus RTP → PCM for the pipeline
// ---------------------------------------------------------------------------

/// Decodes remote Opus RTP payloads to i16 PCM at the pipeline input rate.
pub struct OpusInbound {
    decoder:   Decoder,
    denoiser:  Option<Box<dyn Denoiser48k>>,
    resampler: Option<StreamResampler>, // 48 kHz → out_rate (None when equal)
    out_rate:  u32,
    pcm_48k:   Vec<i16>, // scratch decode buffer (sized for ≤120 ms packets)
}

impl OpusInbound {
    pub fn new(out_rate: u32, denoiser: Option<Box<dyn Denoiser48k>>) -> Self {
        let decoder = Decoder::new(SampleRate::Hz48000, Channels::Mono)
            .expect("failed to create Opus decoder");
        let resampler = (out_rate != OPUS_RATE)
            .then(|| StreamResampler::new(OPUS_RATE, out_rate, ResamplerQuality::Quick));
        Self {
            decoder,
            denoiser,
            resampler,
            out_rate,
            pcm_48k: vec![0i16; OPUS_FRAME_SAMPLES * 6],
        }
    }

    /// Decode one Opus RTP payload → i16 LE PCM bytes at the pipeline rate.
    ///
    /// Returns empty bytes if the resampler is still buffering or on decode
    /// error (logged).
    pub fn push_rtp(&mut self, opus_payload: &[u8]) -> bytes::Bytes {
        if opus_payload.is_empty() {
            return bytes::Bytes::new();
        }

        // 1. Opus decode → 48 kHz mono i16 (disjoint borrows of two fields).
        let n = match self
            .decoder
            .decode(Some(opus_payload), &mut self.pcm_48k[..], false)
        {
            Ok(n) => n,
            Err(e) => {
                log::warn!("OpusInbound: decode error: {}", e);
                return bytes::Bytes::new();
            }
        };
        // Copy out of the scratch buffer to release the borrow on `self`.
        let mut pcm: Vec<i16> = self.pcm_48k[..n].to_vec();

        // 2. Optional full-band denoise (DeepFilterNet) at native 48 kHz.
        if let Some(d) = self.denoiser.as_mut() {
            pcm = d.process(&pcm);
            if pcm.is_empty() {
                return bytes::Bytes::new();
            }
        }

        // 3. Resample 48 kHz → pipeline rate (i16 → f32 → i16).
        let out_samples: Vec<i16> = match self.resampler.as_mut() {
            Some(r) => {
                let f32_in: Vec<f32> = pcm.iter().map(|&s| s as f32 / 32_768.0).collect();
                f32_to_i16(&r.process(&f32_in))
            }
            None => pcm,
        };
        if out_samples.is_empty() {
            return bytes::Bytes::new();
        }

        let mut out = Vec::with_capacity(out_samples.len() * 2);
        for s in out_samples {
            out.extend_from_slice(&s.to_le_bytes());
        }
        bytes::Bytes::from(out)
    }

    /// Pipeline input sample rate this decoder emits at.
    pub fn out_rate(&self) -> u32 {
        self.out_rate
    }
}

// ---------------------------------------------------------------------------
// Outbound: pipeline PCM → Opus packets for the local track
// ---------------------------------------------------------------------------

/// Encodes outbound pipeline PCM to 20 ms Opus packets at 48 kHz.
pub struct OpusOutbound {
    encoder:   Encoder,
    resampler: Option<StreamResampler>, // src_rate → 48 kHz (None when equal)
    src_rate:  u32,
    buf_48k:   Vec<i16>, // accumulates until a whole 20 ms frame is ready
    enc_out:   Vec<u8>,  // scratch encode buffer
}

impl OpusOutbound {
    pub fn new() -> Self {
        let encoder = Encoder::new(SampleRate::Hz48000, Channels::Mono, Application::Voip)
            .expect("failed to create Opus encoder");
        Self {
            encoder,
            resampler: None,
            src_rate: 0,
            buf_48k: Vec::with_capacity(OPUS_FRAME_SAMPLES * 4),
            enc_out: vec![0u8; 4_000],
        }
    }

    /// Push PCM (i16 LE bytes at `src_rate`) and return any complete 20 ms Opus
    /// packets ready to hand to `TrackLocalStaticSample::write_sample`.
    pub fn push_pcm(&mut self, pcm_bytes: &[u8], src_rate: u32) -> Vec<Vec<u8>> {
        // (Re)build the resampler if the source rate changed — the TTS output
        // rate is not fixed (e.g. 24 kHz), so don't hard-code it.
        if self.src_rate != src_rate {
            self.src_rate = src_rate;
            self.resampler = (src_rate != OPUS_RATE)
                .then(|| StreamResampler::new(src_rate, OPUS_RATE, ResamplerQuality::Quick));
            self.buf_48k.clear();
        }

        // i16 LE bytes → f32 → resample to 48 kHz → i16 into buf_48k.
        let f32_in: Vec<f32> = pcm_bytes
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32_768.0)
            .collect();
        let f32_48k = match self.resampler.as_mut() {
            Some(r) => r.process(&f32_in),
            None => f32_in,
        };
        self.buf_48k.extend(f32_48k.iter().map(|&s| {
            (s * 32_768.0).clamp(-32_768.0, 32_767.0) as i16
        }));

        // Emit whole 20 ms frames.
        let mut packets = Vec::new();
        while self.buf_48k.len() >= OPUS_FRAME_SAMPLES {
            let frame: Vec<i16> = self.buf_48k.drain(..OPUS_FRAME_SAMPLES).collect();
            match self.encoder.encode(&frame, &mut self.enc_out[..]) {
                Ok(len) => packets.push(self.enc_out[..len].to_vec()),
                Err(e) => log::warn!("OpusOutbound: encode error: {}", e),
            }
        }
        packets
    }

    /// Drop all buffered audio (called on interruption / barge-in).
    pub fn reset(&mut self) {
        self.buf_48k.clear();
        if let Some(r) = self.resampler.as_mut() {
            r.reset();
        }
    }
}

impl Default for OpusOutbound {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert normalised f32 samples (−1.0 … 1.0) to i16 PCM.
fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&s| (s * 32_768.0).clamp(-32_768.0, 32_767.0) as i16)
        .collect()
}
