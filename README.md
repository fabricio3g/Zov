# Enio

A fast, native GPU-accelerated (`wgpu` + `winit` + `cosmic-text`) desktop dictation and speech-to-text application written in Rust.

## Features

- **Native `wgpu` Desktop Interface**:
  - 100% GPU-accelerated instanced SDF renderer & HiDPI `cosmic-text` glyph/SVG atlas (`resvg`).
  - Minimal graphite dark/light theme (`#1c1c1c` canvas, `#232323` panels, `#e8793a` accent) with custom frameless border resize, smooth hover/press transitions, interactive search & scrollbars, and close-to-tray background operation.
  - Real-time 15-bar live speech equalizer and 64-bar scrolling loudness waveform.
- **System-Wide Dictation (`Ctrl+Shift+Space`)**:
  - Non-activating, always-on-top rounded floating speech overlay pill over any active application (Word, browsers, editors, chat apps).
  - Supports **Smart Hold-to-Talk / Tap**, **Push-to-Talk (Release)**, and **Real-Time Live Streaming**.
  - **Auto-Paste (`Ctrl+V`) + Clipboard**, **Auto-Type (Win32 `SendInput` Unicode) + Clipboard**, or **Clipboard Only** delivery modes.
- **Zero-FFmpeg 16 kHz Audio Pipeline & Local VAD**:
  - Native Windows `winmm` (`waveIn*`) capture at 16 kHz mono 16-bit PCM (`cpal` on Linux/macOS).
  - Adaptive 20 ms Whisper encoder context window sizing (`live_whisper_context`) and local FFT spectral voice-activity gating (`voice_like_spectrum`).
- **On-Device & Cloud Speech Engines**:
  - **On-Device Non-Whisper GGUF (`audio.cpp`)**: Qwen3-ASR (0.6B / 1.7B), NVIDIA Parakeet-TDT 0.6B v3, Canary 1B v2, SenseVoiceSmall, Moonshine Base/Tiny, IBM Granite Speech 3.3 2B, Mistral Voxtral Mini 3B.
  - **On-Device Whisper GGML (`whisper.cpp`)**: Warm `whisper-server` + `whisper-cli` fallback (`tiny_q5` → `large_v3_turbo_q8`, `distil_small_en`, `distil_large_v3`, `kotoba_v2_q5`).
  - **Hugging Face Inference Router** (`HF_TOKEN`) + **15 Commercial Cloud STT Providers** (OpenAI, Groq, Deepgram, ElevenLabs Scribe, Mistral Voxtral, Cartesia Ink, Soniox, Google Gemini Audio, AssemblyAI, Gladia, Fireworks, Together, Speechmatics, Rev AI, Custom/Self-Hosted).

## Building & Running

```bash
cargo run --release
```

CLI subcommands:

```bash
enio status       # Inspect local runtimes, GPU, microphones, and provider status
enio models       # List all on-device, Hugging Face, and commercial STT/TTS models
enio setup        # One-step download of whisper.cpp + small_q5 model
enio transcribe <file.wav>
enio diag-mic     # Test microphone capture & RMS levels
```
