//! Native microphone capture, real-time waveform analysis, local VoiceActivity endpointing,
//! and live real-time phrase slicing.
//!
//! - On Windows: uses native `winmm` (`waveIn*`) at 16 kHz mono 16-bit PCM with rotating
//!   buffers (`WHDR_DONE` polling) — zero COM/WASAPI conflicts, zero crashes when 0 mics
//!   are connected, and full support for USB/Bluetooth/virtual/Realtek microphones.
//! - On Linux/macOS: uses `cpal`.

use crate::vad::{compute_rms, voice_like_spectrum, VadEvent, VoiceActivity, WaveformState};
use crate::wav::{encode_wav_pcm16, WHISPER_SAMPLE_RATE};
use anyhow::{anyhow, Result};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const MAX_RECORDING_MS: u64 = 60_000;
const MIN_RECORDING_MS: u64 = 250;
/// Minimum samples in a real-time streamed clause (~0.45s at 16 kHz).
const MIN_CLAUSE_SAMPLES: usize = 7_200;

pub struct SharedAudioCapture {
    pub samples_16k: Vec<f32>,
    pub clause_start_idx: usize,
    pub waveform: WaveformState,
    pub vad: VoiceActivity,
    pub started_at: Instant,
    pub vad_auto_stop: bool,
    pub realtime_stream: bool,
    pub speech_detected: bool,
    pub clause_has_speech: bool,
    pub chunks_received: u64,
    pub peak_rms: f32,
}

pub struct ActiveRecording {
    stop_signal: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    pub shared: Arc<Mutex<SharedAudioCapture>>,
    pub should_stop: Arc<AtomicBool>,
    pub clause_ready: Arc<AtomicBool>,
    pub device_name: String,
}

impl Drop for ActiveRecording {
    fn drop(&mut self) {
        self.stop_signal.store(true, Ordering::SeqCst);
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }
}

pub struct CompletedRecording {
    pub wav_bytes: Vec<u8>,
    pub duration_secs: f32,
}

fn feed_16k_mono_chunk(
    mono_16k: &[f32],
    shared: &Arc<Mutex<SharedAudioCapture>>,
    should_stop: &Arc<AtomicBool>,
    clause_ready: &Arc<AtomicBool>,
) {
    if mono_16k.is_empty() || should_stop.load(Ordering::Relaxed) {
        return;
    }
    let rms = compute_rms(mono_16k);
    let mut guard = shared.lock();
    let elapsed_ms = guard.started_at.elapsed().as_millis() as u64;
    guard.chunks_received += 1;
    guard.peak_rms = guard.peak_rms.max(rms);
    guard.samples_16k.extend_from_slice(mono_16k);
    guard.waveform.push_rms(rms, elapsed_ms);

    let voice_like = voice_like_spectrum(&guard.samples_16k, WHISPER_SAMPLE_RATE);
    match guard.vad.sample(rms, elapsed_ms, voice_like) {
        Some(VadEvent::Start) => {
            guard.speech_detected = true;
            guard.clause_has_speech = true;
        }
        Some(VadEvent::End) => {
            if guard.realtime_stream && guard.clause_has_speech {
                let pending = guard
                    .samples_16k
                    .len()
                    .saturating_sub(guard.clause_start_idx);
                if pending >= MIN_CLAUSE_SAMPLES {
                    clause_ready.store(true, Ordering::Relaxed);
                }
            } else if guard.vad_auto_stop && guard.speech_detected {
                should_stop.store(true, Ordering::Relaxed);
            }
        }
        None => {}
    }

    if elapsed_ms >= MAX_RECORDING_MS {
        should_stop.store(true, Ordering::Relaxed);
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// WINDOWS NATIVE CAPTURE (winmm waveIn* — 16 kHz mono 16-bit PCM)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(windows)]
pub fn list_input_devices() -> Vec<String> {
    use windows_sys::Win32::Media::Audio::{
        waveInGetDevCapsW, waveInGetNumDevs, WAVEINCAPSW,
    };
    let count = unsafe { waveInGetNumDevs() };
    let mut names = Vec::with_capacity(count as usize);
    for id in 0..count {
        let mut caps: WAVEINCAPSW = unsafe { std::mem::zeroed() };
        let rc = unsafe {
            waveInGetDevCapsW(
                id as usize,
                &mut caps,
                std::mem::size_of::<WAVEINCAPSW>() as u32,
            )
        };
        if rc == 0 {
            let pname = caps.szPname;
            let end = pname.iter().position(|&c| c == 0).unwrap_or(pname.len());
            let name = String::from_utf16_lossy(&pname[..end]).trim().to_string();
            if !name.is_empty() {
                names.push(name);
            }
        }
    }
    names
}

#[cfg(windows)]
pub fn start_recording(
    preferred_device: Option<&str>,
    vad_auto_stop: bool,
    realtime_stream: bool,
) -> Result<ActiveRecording> {
    use windows_sys::Win32::Media::Audio::{
        waveInAddBuffer, waveInClose, waveInGetNumDevs, waveInOpen, waveInPrepareHeader,
        waveInReset, waveInStart, waveInUnprepareHeader, CALLBACK_NULL, HWAVEIN, WAVEFORMATEX,
        WAVEHDR, WAVE_FORMAT_PCM, WAVE_MAPPER, WHDR_DONE,
    };

    let devices = list_input_devices();
    let num_devs = unsafe { waveInGetNumDevs() };
    if num_devs == 0 {
        return Err(anyhow!(
            "No microphone detected. Plug in a microphone or enable it in Windows Sound Settings."
        ));
    }

    let (device_id, device_name) = if let Some(pref) =
        preferred_device.filter(|s| !s.trim().is_empty())
    {
        if let Some((idx, name)) = devices.iter().enumerate().find(|(_, n)| n.as_str() == pref) {
            (idx as u32, name.clone())
        } else {
            (
                WAVE_MAPPER,
                devices
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "Default Microphone".to_string()),
            )
        }
    } else {
        (
            WAVE_MAPPER,
            devices
                .first()
                .cloned()
                .unwrap_or_else(|| "Default Microphone".to_string()),
        )
    };

    let shared = Arc::new(Mutex::new(SharedAudioCapture {
        samples_16k: Vec::with_capacity(WHISPER_SAMPLE_RATE as usize * 15),
        clause_start_idx: 0,
        waveform: WaveformState::default(),
        vad: VoiceActivity::default(),
        started_at: Instant::now(),
        vad_auto_stop,
        realtime_stream,
        speech_detected: false,
        clause_has_speech: false,
        chunks_received: 0,
        peak_rms: 0.0,
    }));
    let should_stop = Arc::new(AtomicBool::new(false));
    let clause_ready = Arc::new(AtomicBool::new(false));
    let stop_signal = Arc::new(AtomicBool::new(false));

    let (init_tx, init_rx) = mpsc::channel::<Result<(), String>>();
    let shared_w = Arc::clone(&shared);
    let should_stop_w = Arc::clone(&should_stop);
    let clause_ready_w = Arc::clone(&clause_ready);
    let stop_signal_w = Arc::clone(&stop_signal);

    let worker = std::thread::spawn(move || {
        // 16 kHz, mono, 16-bit PCM
        let wfx = WAVEFORMATEX {
            wFormatTag: WAVE_FORMAT_PCM as u16,
            nChannels: 1,
            nSamplesPerSec: WHISPER_SAMPLE_RATE,
            nAvgBytesPerSec: WHISPER_SAMPLE_RATE * 2,
            nBlockAlign: 2,
            wBitsPerSample: 16,
            cbSize: 0,
        };

        let mut hwi: HWAVEIN = std::ptr::null_mut();
        let rc = unsafe {
            waveInOpen(
                &mut hwi,
                device_id,
                &wfx,
                0,
                0,
                CALLBACK_NULL,
            )
        };
        if rc != 0 || hwi.is_null() {
            let _ = init_tx.send(Err(format!(
                "Could not open microphone (winmm error {rc}). Check Windows Privacy → Microphone permissions."
            )));
            return;
        }

        // 4 rotating 40 ms buffers (640 samples * 2 bytes = 1280 bytes each)
        const NUM_BUFS: usize = 4;
        const CHUNK_SAMPLES: usize = 640;
        const CHUNK_BYTES: usize = CHUNK_SAMPLES * 2;

        let mut backing: Vec<Box<[u8; CHUNK_BYTES]>> = (0..NUM_BUFS)
            .map(|_| Box::new([0u8; CHUNK_BYTES]))
            .collect();
        let mut headers: Vec<Box<WAVEHDR>> = backing
            .iter_mut()
            .map(|buf| {
                let mut hdr: WAVEHDR = unsafe { std::mem::zeroed() };
                hdr.lpData = buf.as_mut_ptr();
                hdr.dwBufferLength = CHUNK_BYTES as u32;
                Box::new(hdr)
            })
            .collect();

        let hdr_size = std::mem::size_of::<WAVEHDR>() as u32;
        for hdr in &mut headers {
            unsafe {
                waveInPrepareHeader(hwi, &mut **hdr, hdr_size);
                waveInAddBuffer(hwi, &mut **hdr, hdr_size);
            }
        }

        let start_rc = unsafe { waveInStart(hwi) };
        if start_rc != 0 {
            unsafe {
                waveInReset(hwi);
                for hdr in &mut headers {
                    waveInUnprepareHeader(hwi, &mut **hdr, hdr_size);
                }
                waveInClose(hwi);
            }
            let _ = init_tx.send(Err(format!("Failed to start microphone capture (winmm {start_rc})")));
            return;
        }

        let _ = init_tx.send(Ok(()));

        let mut float_scratch = Vec::with_capacity(CHUNK_SAMPLES);
        while !stop_signal_w.load(Ordering::Relaxed) {
            let mut any_done = false;
            std::sync::atomic::fence(Ordering::Acquire);
            for (i, hdr) in headers.iter_mut().enumerate() {
                // WAVEHDR is #[repr(C, packed(1))] and mutated by WinMM's audio thread;
                // fence + unaligned raw pointer read guarantees fresh visibility without UB.
                let hdr_ptr: *mut WAVEHDR = &mut **hdr;
                let flags = unsafe { std::ptr::addr_of!((*hdr_ptr).dwFlags).read_unaligned() };
                if (flags & WHDR_DONE) != 0 {
                    std::sync::atomic::fence(Ordering::Acquire);
                    any_done = true;
                    let recorded_bytes = (unsafe {
                        std::ptr::addr_of!((*hdr_ptr).dwBytesRecorded).read_unaligned()
                    } as usize)
                        .min(CHUNK_BYTES);
                    let raw = &backing[i][..recorded_bytes];
                    float_scratch.clear();
                    for pair in raw.chunks_exact(2) {
                        let sample = i16::from_le_bytes([pair[0], pair[1]]);
                        float_scratch.push(f32::from(sample) / 32768.0);
                    }
                    feed_16k_mono_chunk(
                        &float_scratch,
                        &shared_w,
                        &should_stop_w,
                        &clause_ready_w,
                    );
                    if !stop_signal_w.load(Ordering::Relaxed) {
                        unsafe {
                            std::ptr::addr_of_mut!((*hdr_ptr).dwFlags)
                                .write_unaligned(flags & !WHDR_DONE);
                            std::ptr::addr_of_mut!((*hdr_ptr).dwBytesRecorded).write_unaligned(0);
                            std::sync::atomic::fence(Ordering::Release);
                            waveInAddBuffer(hwi, hdr_ptr, hdr_size);
                        }
                    }
                }
            }
            if !any_done {
                std::thread::sleep(Duration::from_millis(8));
            }
        }

        unsafe {
            waveInReset(hwi);
            for hdr in &mut headers {
                waveInUnprepareHeader(hwi, &mut **hdr, hdr_size);
            }
            waveInClose(hwi);
        }
    });

    match init_rx.recv() {
        Ok(Ok(())) => Ok(ActiveRecording {
            stop_signal,
            worker: Some(worker),
            shared,
            should_stop,
            clause_ready,
            device_name,
        }),
        Ok(Err(err)) => {
            let _ = worker.join();
            Err(anyhow!(err))
        }
        Err(_) => {
            let _ = worker.join();
            Err(anyhow!("Audio capture thread exited unexpectedly"))
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// NON-WINDOWS CAPTURE (cpal)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(not(windows))]
pub fn list_input_devices() -> Vec<String> {
    use cpal::traits::{DeviceTrait, HostTrait};
    let host = cpal::default_host();
    let mut names = Vec::new();
    if let Ok(devices) = host.input_devices() {
        for dev in devices {
            if let Ok(name) = dev.name() {
                names.push(name);
            }
        }
    }
    names
}

#[cfg(not(windows))]
pub fn start_recording(
    preferred_device: Option<&str>,
    vad_auto_stop: bool,
    realtime_stream: bool,
) -> Result<ActiveRecording> {
    use crate::wav::resample_to_16k_mono;
    use anyhow::Context;
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    let pref_owned = preferred_device
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.to_string());

    let shared = Arc::new(Mutex::new(SharedAudioCapture {
        samples_16k: Vec::with_capacity(WHISPER_SAMPLE_RATE as usize * 15),
        clause_start_idx: 0,
        waveform: WaveformState::default(),
        vad: VoiceActivity::default(),
        started_at: Instant::now(),
        vad_auto_stop,
        realtime_stream,
        speech_detected: false,
        clause_has_speech: false,
        chunks_received: 0,
        peak_rms: 0.0,
    }));
    let should_stop = Arc::new(AtomicBool::new(false));
    let clause_ready = Arc::new(AtomicBool::new(false));
    let stop_signal = Arc::new(AtomicBool::new(false));
    let (init_tx, init_rx) = mpsc::channel::<Result<String, String>>();

    let shared_w = Arc::clone(&shared);
    let should_stop_w = Arc::clone(&should_stop);
    let clause_ready_w = Arc::clone(&clause_ready);
    let stop_signal_w = Arc::clone(&stop_signal);

    let worker = std::thread::spawn(move || {
        let setup = (|| -> Result<(cpal::Stream, String)> {
            let host = cpal::default_host();
            let device = if let Some(ref p) = pref_owned {
                host.input_devices()
                    .ok()
                    .and_then(|mut devs| devs.find(|d| d.name().ok().as_deref() == Some(p.as_str())))
                    .or_else(|| host.default_input_device())
            } else {
                host.default_input_device()
            }
            .context("No microphone detected")?;
            let name = device.name().unwrap_or_else(|_| "Default Microphone".to_string());
            let supported = device.default_input_config()?;
            let fmt = supported.sample_format();
            let cfg: cpal::StreamConfig = supported.into();
            let rate = cfg.sample_rate.0;
            let ch = usize::from(cfg.channels).max(1);
            let stream = match fmt {
                cpal::SampleFormat::F32 => device.build_input_stream(
                    &cfg,
                    move |data: &[f32], _| {
                        let mono = resample_to_16k_mono(data, rate, ch);
                        feed_16k_mono_chunk(&mono, &shared_w, &should_stop_w, &clause_ready_w);
                    },
                    |_| {},
                    None,
                )?,
                _ => anyhow::bail!("Unsupported format"),
            };
            stream.play()?;
            Ok((stream, name))
        })();
        match setup {
            Ok((stream, name)) => {
                let _ = init_tx.send(Ok(name));
                while !stop_signal_w.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(15));
                }
                let _ = stream.pause();
                drop(stream);
            }
            Err(e) => {
                let _ = init_tx.send(Err(e.to_string()));
            }
        }
    });

    match init_rx.recv() {
        Ok(Ok(device_name)) => Ok(ActiveRecording {
            stop_signal,
            worker: Some(worker),
            shared,
            should_stop,
            clause_ready,
            device_name,
        }),
        Ok(Err(e)) => {
            let _ = worker.join();
            Err(anyhow!(e))
        }
        Err(_) => {
            let _ = worker.join();
            Err(anyhow!("Audio capture thread exited unexpectedly"))
        }
    }
}

impl ActiveRecording {
    pub fn elapsed_secs(&self) -> u64 {
        self.shared.lock().started_at.elapsed().as_secs()
    }

    pub fn diag_stats(&self) -> (u64, usize, f32) {
        let guard = self.shared.lock();
        (guard.chunks_received, guard.samples_16k.len(), guard.peak_rms)
    }

    pub fn snapshot_waveform(&self) -> ([f32; 15], Vec<f32>, bool) {
        let mut guard = self.shared.lock();
        let now_ms = guard.started_at.elapsed().as_millis() as u64;
        guard.waveform.tick(now_ms);
        (
            guard.waveform.audio_levels,
            guard.waveform.history.clone(),
            guard.vad.speaking,
        )
    }

    pub fn take_completed_clause(&self) -> Option<CompletedRecording> {
        if !self.clause_ready.swap(false, Ordering::Relaxed) {
            return None;
        }
        let mut guard = self.shared.lock();
        let start = guard.clause_start_idx;
        let end = guard.samples_16k.len();
        if end.saturating_sub(start) < MIN_CLAUSE_SAMPLES {
            return None;
        }
        let slice = &guard.samples_16k[start..end];
        let duration_secs = slice.len() as f32 / WHISPER_SAMPLE_RATE as f32;
        let wav_bytes = encode_wav_pcm16(slice, WHISPER_SAMPLE_RATE);
        guard.clause_start_idx = end;
        guard.clause_has_speech = false;
        Some(CompletedRecording {
            wav_bytes,
            duration_secs,
        })
    }

    pub fn finish(mut self) -> Option<CompletedRecording> {
        self.stop_signal.store(true, Ordering::SeqCst);
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
        let guard = self.shared.lock();
        let elapsed_ms = guard.started_at.elapsed().as_millis() as u64;
        let start = if guard.realtime_stream {
            guard.clause_start_idx
        } else {
            0
        };
        let remaining = &guard.samples_16k[start.min(guard.samples_16k.len())..];
        if elapsed_ms < MIN_RECORDING_MS || remaining.len() < 2000 {
            return None;
        }
        let duration_secs = remaining.len() as f32 / WHISPER_SAMPLE_RATE as f32;
        let wav_bytes = encode_wav_pcm16(remaining, WHISPER_SAMPLE_RATE);
        Some(CompletedRecording {
            wav_bytes,
            duration_secs,
        })
    }
}
