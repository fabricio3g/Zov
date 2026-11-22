//! 16 kHz mono 16-bit PCM WAV encoding, resampling, and Whisper encoder window sizing.
//!
//! Ported from `apps/desktop/renderer/src/lib/features/chat/composer/wav-encode.ts`
//! and `apps/desktop/main/voice/whisper-window.ts`.

use anyhow::{bail, Context, Result};
use std::path::Path;

pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Encode mono float samples in `[-1.0, 1.0]` as 16-bit PCM WAV bytes.
pub fn encode_wav_pcm16(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_bytes = (samples.len() * 2) as u32;
    let total_len = 44 + data_bytes as usize;
    let mut bytes = Vec::with_capacity(total_len);

    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16_u32.to_le_bytes()); // PCM chunk size
    bytes.extend_from_slice(&1_u16.to_le_bytes()); // PCM format = 1
    bytes.extend_from_slice(&1_u16.to_le_bytes()); // Mono = 1 channel
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    bytes.extend_from_slice(&2_u16.to_le_bytes()); // block align
    bytes.extend_from_slice(&16_u16.to_le_bytes()); // bits per sample
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_bytes.to_le_bytes());

    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let pcm = if clamped < 0.0 {
            (clamped * 32768.0) as i16
        } else {
            (clamped * 32767.0) as i16
        };
        bytes.extend_from_slice(&pcm.to_le_bytes());
    }

    bytes
}

/// Downmix interleaved multi-channel audio to mono and resample to 16 kHz.
pub fn resample_to_16k_mono(interleaved: &[f32], src_rate: u32, channels: usize) -> Vec<f32> {
    if interleaved.is_empty() || channels == 0 || src_rate == 0 {
        return Vec::new();
    }
    let frames = interleaved.len() / channels;
    if frames == 0 {
        return Vec::new();
    }

    let mut mono = Vec::with_capacity(frames);
    for frame in interleaved.chunks_exact(channels) {
        let sum: f32 = frame.iter().copied().sum();
        mono.push(sum / channels as f32);
    }

    if src_rate == WHISPER_SAMPLE_RATE {
        return mono;
    }

    let ratio = f64::from(src_rate) / f64::from(WHISPER_SAMPLE_RATE);
    let out_frames = ((frames as f64) / ratio).ceil() as usize;
    let mut resampled = Vec::with_capacity(out_frames);

    for i in 0..out_frames {
        let src_pos = i as f64 * ratio;
        let idx = src_pos.floor() as usize;
        let frac = (src_pos - idx as f64) as f32;
        let s0 = mono.get(idx).copied().unwrap_or(0.0);
        let s1 = mono.get((idx + 1).min(frames - 1)).copied().unwrap_or(s0);
        resampled.push(s0 + (s1 - s0) * frac);
    }

    resampled
}

/// Whisper encoder positions span 20 ms each. Short 16 kHz mono WAV clips can
/// avoid encoding a full 30-second window (1500 positions), cutting latency by 3–5x.
/// Returns `0` for unknown formats or clips >= ~29s so the engine uses its default window.
pub fn live_whisper_context(wav: &[u8]) -> usize {
    if wav.len() < 44 || &wav[0..4] != b"RIFF" || &wav[8..12] != b"WAVE" {
        return 0;
    }
    let mut offset = 12usize;
    let mut rate = 0u32;
    let mut align = 0u16;
    let mut pcm = false;
    let mut data_bytes = 0usize;

    while offset + 8 <= wav.len() {
        let tag = &wav[offset..offset + 4];
        let size = u32::from_le_bytes([
            wav[offset + 4],
            wav[offset + 5],
            wav[offset + 6],
            wav[offset + 7],
        ]) as usize;
        let body = offset + 8;
        if body + size > wav.len() {
            return 0;
        }
        if tag == b"fmt " && size >= 16 {
            let format_tag = u16::from_le_bytes([wav[body], wav[body + 1]]);
            let channels = u16::from_le_bytes([wav[body + 2], wav[body + 3]]);
            let bits = u16::from_le_bytes([wav[body + 14], wav[body + 15]]);
            pcm = format_tag == 1 && channels == 1 && bits == 16;
            rate = u32::from_le_bytes([
                wav[body + 4],
                wav[body + 5],
                wav[body + 6],
                wav[body + 7],
            ]);
            align = u16::from_le_bytes([wav[body + 12], wav[body + 13]]);
        } else if tag == b"data" {
            data_bytes += size;
        }
        offset = body + size + (size % 2);
    }

    if !pcm || rate != 16_000 || align != 2 || data_bytes == 0 || data_bytes % (align as usize) != 0
    {
        return 0;
    }

    let seconds = data_bytes as f64 / (f64::from(rate) * f64::from(align));
    let blocks = ((seconds + 0.8) * 50.0 / 128.0).ceil() as usize;
    let positions = (blocks * 128).max(256);
    if positions < 1500 {
        positions
    } else {
        0
    }
}

/// Load any WAV file from disk and normalize it to 16 kHz mono 16-bit PCM WAV bytes.
pub fn load_wav_normalized(path: &Path) -> Result<Vec<u8>> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let spec = reader.spec();
    if spec.channels == 0 || spec.sample_rate == 0 {
        bail!("Invalid WAV header in {}", path.display());
    }
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 2_f32.powi(i32::from(spec.bits_per_sample) - 1);
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
    };
    let mono_16k = resample_to_16k_mono(&samples, spec.sample_rate, usize::from(spec.channels));
    Ok(encode_wav_pcm16(&mono_16k, WHISPER_SAMPLE_RATE))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_valid_16k_mono_wav_and_computes_live_context() {
        // 2 seconds of 16 kHz mono audio = 32,000 samples
        let samples = vec![0.1_f32; 32_000];
        let wav = encode_wav_pcm16(&samples, WHISPER_SAMPLE_RATE);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(wav.len(), 44 + 64_000);

        // 2.0s + 0.8s = 2.8s * 50 = 140 positions -> rounded to 256
        let ctx = live_whisper_context(&wav);
        assert_eq!(ctx, 256);
    }

    #[test]
    fn long_wav_returns_zero_context_for_full_window() {
        // 31 seconds of 16 kHz mono audio
        let samples = vec![0.0_f32; 16_000 * 31];
        let wav = encode_wav_pcm16(&samples, WHISPER_SAMPLE_RATE);
        assert_eq!(live_whisper_context(&wav), 0);
    }

    #[test]
    fn resamples_48k_stereo_to_16k_mono() {
        // 4800 stereo frames at 48 kHz = 0.1s -> 1600 mono frames at 16 kHz
        let stereo = vec![0.5_f32; 4800 * 2];
        let mono = resample_to_16k_mono(&stereo, 48_000, 2);
        assert_eq!(mono.len(), 1600);
        assert!((mono[0] - 0.5).abs() < 1e-4);
    }
}
