<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/icons/app-icon-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/icons/app-icon-light.svg">
  <img src="assets/icons/app-icon-dark.svg" width="88" height="88" alt="Zov icon">
</picture>

# Zov

**Native, GPU-accelerated desktop dictation — speak anywhere, land in any app.**

*Built in Rust with `wgpu`, `winit`, and `cosmic-text`. Runs locally on your machine or through 15+ cloud and self-hosted speech engines.*

</div>

---

## Why Zov

Most dictation tools feel bolted on: slow browser wrappers, heavyweight runtimes, or overlays that steal focus right out of the window you are typing in.

**Zov** stays out of your way until you need it. Press **`Ctrl+Shift+Space`** from any editor, browser, document, or chat box and a compact, non-activating speech pill floats over your cursor while your keyboard focus stays untouched. Speak naturally, release, and your words drop straight into place—transcribed on-device or by your preferred cloud provider.

Close the window and Zov tucks cleanly into the System Tray so global dictation stays ready in the background.

## Highlights

- **System-wide dictation (`Ctrl+Shift+Space`)** — Choose between **Smart Hold-to-Talk / Tap**, strict **Push-to-Talk**, or **Real-Time Live Streaming** that slices clauses automatically on natural pauses.
- **Flexible text delivery** — Insert via **Auto-Paste (`Ctrl+V`)**, **Auto-Type (Win32 Unicode `SendInput`)**, or **Clipboard Only**, with full transcript history and one-click copy, re-type, or delete.
- **Native GPU interface** — A custom instanced SDF `wgpu` renderer with HiDPI `cosmic-text` typography, animated SVG iconography, adaptive dark & light monochrome themes, smooth hover/press transitions, interactive search and scrollbars, and a live speech-reactive waveform.
- **Zero-FFmpeg 16 kHz audio pipeline** — Direct Windows `winmm` (`waveIn*`) capture (`cpal` on Linux/macOS), local FFT spectral voice-activity detection (`85–6000 Hz` speech band + flatness gate), and adaptive 20 ms Whisper encoder context sizing so short clips transcribe fast.
- **Resumable on-device model management** — One-click setup for local runtimes and weights with pause, resume (HTTP byte-range), retry, and cancel controls built right into the UI.

## Speech Engines & Models

Switch between local models, Hugging Face, commercial APIs, and self-hosted servers from the composer dropdown or the **Models** view.

| Category | Engine / Providers | Highlights |
|---|---|---|
| **Local Non-Whisper (GGUF)** | `audio.cpp` | Qwen3-ASR (`0.6B` / `1.7B`), NVIDIA Parakeet-TDT `0.6B` v3, Canary `1B` v2, SenseVoiceSmall, Moonshine Base/Tiny, IBM Granite Speech `3.3 2B`, Mistral Voxtral Mini `3B` |
| **Local Whisper (GGML)** | `whisper.cpp` (warm `whisper-server` + `whisper-cli` fallback) | `tiny_q5` through `large_v3_turbo_q8`, `distil_small_en`, `distil_large_v3`, `kotoba_v2_q5` |
| **Hugging Face Hub** | Serverless Inference Router (`HF_TOKEN`) | Open-ASR leaderboard models and gated downloads |
| **Commercial Cloud (15+)** | OpenAI, Groq, Deepgram, ElevenLabs Scribe, Mistral Voxtral, Cartesia Ink, Soniox, Google Gemini Audio, AssemblyAI, Gladia, Fireworks, Together, Speechmatics, Rev AI | Per-provider API key & custom Base URL overrides in Settings |
| **Self-Hosted** | Any OpenAI-compatible `/v1/audio/transcriptions` server | vLLM, Speaches, Sherpa-ONNX, faster-whisper — configurable endpoint URL, model ID, and optional token |

## Getting Started

Requires a recent Rust toolchain (`rustup`).

```bash
git clone https://github.com/fabricio3g/Enio.git
cd Enio
cargo run --release
```

### CLI Commands

Zov also ships a lightweight CLI alongside the desktop app:

```bash
cargo run --release -- status                   # Inspect runtimes, GPU, microphones, and provider readiness
cargo run --release -- models                   # List all on-device, Hugging Face, and commercial models
cargo run --release -- setup small_q5           # Download whisper.cpp runtime + a local model in one step
cargo run --release -- transcribe clip.wav      # Transcribe an audio file directly from the terminal
cargo run --release -- diag-mic                 # Verify microphone detection and live RMS capture levels
```

## License

MIT
