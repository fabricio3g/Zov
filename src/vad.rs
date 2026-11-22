//! Local voice activity detection (VAD), spectral speech gating, transcript cleaning,
//! and real-time waveform bar calculation.
//!
//! Ported from:
//! - `apps/desktop/renderer/src/lib/features/chat/composer/voice-activity.ts`
//! - `apps/desktop/renderer/src/lib/features/chat/composer/use-dictation.svelte.ts`

use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::{Arc, OnceLock};

/// Relative bar heights for the 15-bar live equalizer, normalized to the tallest (58).
pub const WAVE_WEIGHTS: [f32; 15] = [
    28.0 / 58.0,
    36.0 / 58.0,
    46.0 / 58.0,
    34.0 / 58.0,
    26.0 / 58.0,
    38.0 / 58.0,
    52.0 / 58.0,
    58.0 / 58.0,
    44.0 / 58.0,
    32.0 / 58.0,
    26.0 / 58.0,
    37.0 / 58.0,
    49.0 / 58.0,
    35.0 / 58.0,
    28.0 / 58.0,
];

/// Number of bars in the scrolling loudness waveform history.
pub const HISTORY_BARS: usize = 64;
/// Milliseconds between scrolling history bars.
pub const HISTORY_STEP_MS: u64 = 55;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VadEvent {
    Start,
    End,
}

/// Endpointing state machine: tracks background noise floor and triggers `Start`
/// after 120 ms of voice-like energy above threshold, and `End` after 600 ms of silence
/// (or 20 s maximum segment length).
#[derive(Clone, Debug)]
pub struct VoiceActivity {
    noise: f32,
    started_at_ms: u64,
    candidate_at_ms: Option<u64>,
    last_speech_at_ms: u64,
    pub speaking: bool,
}

impl Default for VoiceActivity {
    fn default() -> Self {
        Self {
            noise: 0.004,
            started_at_ms: 0,
            candidate_at_ms: None,
            last_speech_at_ms: 0,
            speaking: false,
        }
    }
}

impl VoiceActivity {
    pub fn reset(&mut self, now_ms: u64) {
        self.started_at_ms = now_ms;
        self.candidate_at_ms = None;
        self.last_speech_at_ms = 0;
        self.speaking = false;
    }

    pub fn sample(&mut self, rms: f32, now_ms: u64, voice_like: bool) -> Option<VadEvent> {
        if !rms.is_finite() {
            return None;
        }
        let threshold = (self.noise * 3.0).max(0.012);
        if voice_like && rms > threshold {
            let candidate = *self.candidate_at_ms.get_or_insert(now_ms);
            self.last_speech_at_ms = now_ms;
            if !self.speaking && now_ms.saturating_sub(candidate) >= 120 {
                self.speaking = true;
                self.started_at_ms = now_ms;
                return Some(VadEvent::Start);
            }
        } else {
            if !self.speaking {
                self.candidate_at_ms = None;
                self.noise = self.noise * 0.96 + rms.min(0.006) * 0.04;
            }
            if self.speaking && now_ms.saturating_sub(self.last_speech_at_ms) >= 600 {
                self.reset(now_ms);
                return Some(VadEvent::End);
            }
        }
        if self.speaking && now_ms.saturating_sub(self.started_at_ms) >= 20_000 {
            self.reset(now_ms);
            return Some(VadEvent::End);
        }
        None
    }
}

const SPECTRUM_SIZE: usize = 256;

struct SpectrumPlan {
    fft: Arc<dyn Fft<f32>>,
    hann: [f32; SPECTRUM_SIZE],
}

/// Local spectral gate. The plan and window are shared, while each recording
/// keeps its own working buffers.
pub struct VoiceSpectrum {
    plan: &'static SpectrumPlan,
    buffer: [Complex<f32>; SPECTRUM_SIZE],
    scratch: Vec<Complex<f32>>,
}

impl Default for VoiceSpectrum {
    fn default() -> Self {
        static PLAN: OnceLock<SpectrumPlan> = OnceLock::new();
        let plan = PLAN.get_or_init(|| SpectrumPlan {
            fft: FftPlanner::<f32>::new().plan_fft_forward(SPECTRUM_SIZE),
            hann: std::array::from_fn(|i| {
                0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / SPECTRUM_SIZE as f32).cos()
            }),
        });
        Self {
            plan,
            buffer: [Complex::default(); SPECTRUM_SIZE],
            scratch: vec![Complex::default(); plan.fft.get_inplace_scratch_len()],
        }
    }
}

impl VoiceSpectrum {
    /// Reject broad hiss, low hum, and tones before amplitude endpointing.
    pub fn is_voice_like(&mut self, samples: &[f32], sample_rate: u32) -> bool {
        let n = SPECTRUM_SIZE;
        if samples.len() < n || sample_rate == 0 {
            return true;
        }
        let slice = &samples[samples.len() - n..];
        for ((bin, &sample), &window) in self.buffer.iter_mut().zip(slice).zip(&self.plan.hann) {
            *bin = Complex::new(sample * window, 0.0);
        }
        self.plan
            .fft
            .process_with_scratch(&mut self.buffer, &mut self.scratch);

        let half = n / 2;
        let bin_hz = sample_rate as f32 / n as f32;
        let mut total = 0.0_f64;
        let mut band = 0.0_f64;
        let mut log_band = 0.0_f64;
        let mut bins = 0usize;
        let mut peak = 0.0_f64;
        let mut peak_index = 0usize;

        for index in 1..half {
            let hz = index as f32 * bin_hz;
            let power = f64::from(self.buffer[index].norm_sqr());
            total += power;
            if !(85.0..=6000.0).contains(&hz) {
                continue;
            }
            band += power;
            log_band += power.max(1e-12).ln();
            bins += 1;
            if power > peak {
                peak = power;
                peak_index = index;
            }
        }

        if band <= 0.0 || bins == 0 || total <= 0.0 || (band / total) < 0.6 {
            return false;
        }

        let flatness = (log_band / bins as f64).exp() / (band / bins as f64);
        let lo = peak_index.saturating_sub(1).max(1);
        let hi = (peak_index + 1).min(half - 1);
        let tone: f64 = self.buffer[lo..=hi]
            .iter()
            .map(|bin| f64::from(bin.norm_sqr()))
            .sum();

        flatness < 0.55 && (tone / band) < 0.8
    }
}

/// Discard explicit non-speech captions (`[blank_audio]`, `[silence]`, `(noise)`, `<|...|>`),
/// retaining normal short replies in any language.
pub fn spoken_transcript(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if !trimmed.chars().any(|c| c.is_alphanumeric()) {
        return None;
    }
    let lower = trimmed
        .trim_end_matches(|c: char| c == '.' || c == '!' || c.is_whitespace())
        .to_ascii_lowercase();

    const NON_SPEECH: &[&str] = &[
        "[blank_audio]",
        "[silence]",
        "[noise]",
        "[music]",
        "[inaudible]",
        "[applause]",
        "(silence)",
        "(noise)",
        "(music)",
        "(inaudible)",
        "(applause)",
    ];
    if NON_SPEECH.contains(&lower.as_str()) {
        return None;
    }
    if lower.starts_with("<|") && lower.ends_with("|>") && !lower[2..lower.len() - 2].contains('>')
    {
        return None;
    }
    Some(trimmed.to_string())
}

/// Live waveform state matching `use-dictation.svelte.ts`.
#[derive(Clone, Debug)]
pub struct WaveformState {
    pub smoothed_level: f32,
    pub audio_levels: [f32; 15],
    pub history: Vec<f32>,
    last_history_ms: u64,
    last_rms_ms: u64,
}

impl Default for WaveformState {
    fn default() -> Self {
        Self {
            smoothed_level: 0.0,
            audio_levels: [0.15; 15],
            history: vec![0.0; HISTORY_BARS],
            last_history_ms: 0,
            last_rms_ms: 0,
        }
    }
}

impl WaveformState {
    /// Feed real microphone RMS from an audio capture chunk.
    pub fn push_rms(&mut self, rms: f32, now_ms: u64) {
        self.last_rms_ms = now_ms;
        // Perceptual sqrt curve with gentle noise floor subtraction so both quiet webcam mics
        // (0.01..0.06 RMS) and loud studio mics animate cleanly across 0.0..1.0.
        let normalized = ((rms - 0.002).max(0.0) * 16.0).sqrt().clamp(0.0, 1.0);
        if normalized > self.smoothed_level {
            // Fast attack so speech syllables pop immediately
            self.smoothed_level = self.smoothed_level * 0.40 + normalized * 0.60;
        } else {
            // Smooth natural release
            self.smoothed_level = self.smoothed_level * 0.78 + normalized * 0.22;
        }
        self.tick(now_ms);
    }

    /// Advance 60 fps waveform & equalizer animation without injecting fake RMS.
    pub fn tick(&mut self, now_ms: u64) {
        // Gently decay if no fresh audio chunk arrived in the last 60 ms.
        if now_ms.saturating_sub(self.last_rms_ms) > 60 {
            self.smoothed_level *= 0.88;
            if self.smoothed_level < 0.005 {
                self.smoothed_level = 0.0;
            }
        }

        if now_ms.saturating_sub(self.last_history_ms) >= HISTORY_STEP_MS {
            self.last_history_ms = now_ms;
            if !self.history.is_empty() {
                self.history.remove(0);
            }
            self.history.push(self.smoothed_level.clamp(0.0, 1.0));
        }

        let now_sec = now_ms as f32 / 1000.0;
        for (i, weight) in WAVE_WEIGHTS.iter().enumerate() {
            // Subtle breathing idle wave so the overlay always looks alive while listening
            let idle = 0.14 + (now_sec * 3.4 + i as f32 * 0.55).sin() * 0.08;
            // Dynamic speech-reactive modulation across bars
            let shimmer = 0.70 + 0.30 * (now_sec * 11.5 + i as f32 * 0.9).sin();
            let speech = (self.smoothed_level * *weight * shimmer * 1.35).clamp(0.0, 1.0);
            self.audio_levels[i] = idle.max(speech).clamp(0.06, 1.0);
        }
    }
}

pub fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vad_starts_after_120ms_and_ends_after_600ms_silence() {
        let mut vad = VoiceActivity::default();
        assert_eq!(vad.sample(0.05, 0, true), None);
        assert_eq!(vad.sample(0.05, 80, true), None);
        assert_eq!(vad.sample(0.05, 130, true), Some(VadEvent::Start));
        assert!(vad.speaking);

        assert_eq!(vad.sample(0.001, 400, false), None);
        assert_eq!(vad.sample(0.001, 740, false), Some(VadEvent::End));
        assert!(!vad.speaking);
    }

    #[test]
    fn filters_whisper_non_speech_tokens() {
        assert_eq!(spoken_transcript("[BLANK_AUDIO]"), None);
        assert_eq!(spoken_transcript(" (silence). "), None);
        assert_eq!(spoken_transcript("<|nospeech|>"), None);
        assert_eq!(
            spoken_transcript("  Hola mundo "),
            Some("Hola mundo".to_string())
        );
    }

    #[test]
    fn rejects_single_pure_tone_in_spectrum() {
        // Pure 440 Hz sine wave at 16 kHz -> concentrated in 1 bin -> tone/band >= 0.8 -> false
        let samples: Vec<f32> = (0..256)
            .map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / 16_000.0).sin())
            .collect();
        assert!(!VoiceSpectrum::default().is_voice_like(&samples, 16_000));
    }
}
