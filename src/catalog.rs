//! Complete catalog of On-Device Local Speech Models (Non-Whisper + Whisper) and
//! Commercial / Hugging Face Cloud STT & TTS engines.
//!
//! Includes:
//! - **9 Local Non-Whisper Models** (`audio.cpp` / GGUF):
//!   Alibaba `Qwen3-ASR-0.6B` & `Qwen3-ASR-1.7B`, NVIDIA `Parakeet-TDT-0.6B-v3` & `Canary-1B-v2`,
//!   FunAudioLLM `SenseVoiceSmall`, Useful Sensors `Moonshine-Base` & `Moonshine-Tiny`,
//!   IBM `Granite-Speech-3.3-2B`, and Mistral `Voxtral-Mini-3B`.
//! - **14 Local Whisper Models** (`whisper.cpp` GGML):
//!   Tiny → Large v3 Turbo Q8 + Distil-Whisper + Kotoba-Whisper v2.
//! - **16 Cloud / Commercial STT Providers (48+ Models)** including **Hugging Face Inference Router (`HF_TOKEN`)**
//!   for non-Whisper models (`Qwen/Qwen3-ASR-1.7B`, `nvidia/parakeet-tdt-0.6b-v3`, `nvidia/canary-1b-v2`,
//!   `FunAudioLLM/SenseVoiceSmall`, `UsefulSensors/moonshine-base`, `ibm-granite/granite-speech-3.3-2b`,
//!   `microsoft/Phi-4-multimodal-instruct`, `facebook/seamless-m4t-v2-large`, `facebook/mms-1b-all`),
//!   plus OpenAI, Groq, Deepgram, ElevenLabs Scribe v3, Mistral Voxtral 3, Cartesia Ink, Soniox v6,
//!   Google Gemini 2.5, AssemblyAI Universal-3.6 Pro, Gladia Solaria-3, Fireworks, Together,
//!   Speechmatics, Rev AI, and Self-Hosted servers.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalEngineKind {
    /// Runs via `whisper.cpp` (`whisper-server` / `whisper-cli`).
    WhisperCpp,
    /// Non-Whisper architecture running via `audio.cpp` (`audiocpp_server` / `audiocpp_cli`).
    AudioCpp,
}

#[derive(Clone, Debug, Serialize)]
pub struct LocalWhisperSpec {
    pub id: &'static str,
    pub engine: LocalEngineKind,
    pub architecture: &'static str,
    pub tier: &'static str,
    pub label: &'static str,
    pub repo: &'static str,
    pub file: &'static str,
    pub url: &'static str,
    pub size_mb: u32,
    pub multilingual: bool,
    pub recommended: bool,
    /// True if downloading from Hugging Face may benefit from or require `hf_token`.
    pub hf_gated_or_hub: bool,
}

pub const LOCAL_WHISPER_CATALOG: &[LocalWhisperSpec] = &[
    // ── Local Non-Whisper Architectures (audio.cpp GGUF) ─────────────────────
    LocalWhisperSpec {
        id: "qwen3_asr_0_6b_q8",
        engine: LocalEngineKind::AudioCpp,
        architecture: "Qwen3-ASR (Non-Whisper)",
        tier: "non-whisper-fast",
        label: "[Non-Whisper] Alibaba Qwen3-ASR 0.6B (Q8_0) — 30 Languages & 22 Dialects",
        repo: "audio-cpp/audio.cpp-gguf",
        file: "qwen3-asr-0.6b-q8_0.gguf",
        url: "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-q8_0.gguf",
        size_mb: 640,
        multilingual: true,
        recommended: true,
        hf_gated_or_hub: true,
    },
    LocalWhisperSpec {
        id: "qwen3_asr_1_7b_q5",
        engine: LocalEngineKind::AudioCpp,
        architecture: "Qwen3-ASR (Non-Whisper)",
        tier: "non-whisper-best",
        label: "[Non-Whisper] Alibaba Qwen3-ASR 1.7B (Q5_K_M) — #1 Open-ASR Leaderboard",
        repo: "audio-cpp/audio.cpp-gguf",
        file: "qwen3-asr-1.7b-q5_k_m.gguf",
        url: "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Qwen3-ASR-1.7B-GGUF/qwen3-asr-1.7b-q5_k_m.gguf",
        size_mb: 1250,
        multilingual: true,
        recommended: true,
        hf_gated_or_hub: true,
    },
    LocalWhisperSpec {
        id: "parakeet_tdt_0_6b_v3",
        engine: LocalEngineKind::AudioCpp,
        architecture: "NVIDIA Parakeet-TDT (Non-Whisper)",
        tier: "non-whisper-turbo",
        label: "[Non-Whisper] NVIDIA Parakeet-TDT 0.6B v3 (Q8_0) — Ultra-Fast TDT Transducer",
        repo: "audio-cpp/audio.cpp-gguf",
        file: "parakeet-tdt-0.6b-v3-q8_0.gguf",
        url: "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Parakeet-TDT-0.6B-v3-GGUF/parakeet-tdt-0.6b-v3-q8_0.gguf",
        size_mb: 620,
        multilingual: true,
        recommended: true,
        hf_gated_or_hub: true,
    },
    LocalWhisperSpec {
        id: "sensevoice_small_q8",
        engine: LocalEngineKind::AudioCpp,
        architecture: "FunASR SenseVoice (Non-Whisper)",
        tier: "non-whisper-lowlatency",
        label: "[Non-Whisper] Alibaba SenseVoiceSmall (Q8_0) — Non-Autoregressive + Emotion Tags",
        repo: "audio-cpp/audio.cpp-gguf",
        file: "sensevoice-small-q8_0.gguf",
        url: "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/SenseVoiceSmall-GGUF/sensevoice-small-q8_0.gguf",
        size_mb: 245,
        multilingual: true,
        recommended: true,
        hf_gated_or_hub: true,
    },
    LocalWhisperSpec {
        id: "moonshine_base_q8",
        engine: LocalEngineKind::AudioCpp,
        architecture: "Moonshine (Non-Whisper)",
        tier: "non-whisper-edge",
        label: "[Non-Whisper] Useful Sensors Moonshine Base (Q8_0) — Fast Edge Dictation",
        repo: "audio-cpp/audio.cpp-gguf",
        file: "moonshine-base-q8_0.gguf",
        url: "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Moonshine-Base-GGUF/moonshine-base-q8_0.gguf",
        size_mb: 78,
        multilingual: false,
        recommended: false,
        hf_gated_or_hub: true,
    },
    LocalWhisperSpec {
        id: "moonshine_tiny_q8",
        engine: LocalEngineKind::AudioCpp,
        architecture: "Moonshine (Non-Whisper)",
        tier: "non-whisper-edge",
        label: "[Non-Whisper] Useful Sensors Moonshine Tiny (Q8_0) — 34 MB Ultra-Light Edge ASR",
        repo: "audio-cpp/audio.cpp-gguf",
        file: "moonshine-tiny-q8_0.gguf",
        url: "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Moonshine-Tiny-GGUF/moonshine-tiny-q8_0.gguf",
        size_mb: 34,
        multilingual: false,
        recommended: false,
        hf_gated_or_hub: true,
    },
    LocalWhisperSpec {
        id: "canary_1b_v2",
        engine: LocalEngineKind::AudioCpp,
        architecture: "NVIDIA Canary (Non-Whisper)",
        tier: "non-whisper-multilingual",
        label: "[Non-Whisper] NVIDIA Canary 1B v2 (Q5_K_M) — Multilingual ASR & Speech Translation",
        repo: "audio-cpp/audio.cpp-gguf",
        file: "canary-1b-v2-q5_k_m.gguf",
        url: "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Canary-1B-v2-GGUF/canary-1b-v2-q5_k_m.gguf",
        size_mb: 740,
        multilingual: true,
        recommended: false,
        hf_gated_or_hub: true,
    },
    LocalWhisperSpec {
        id: "granite_speech_2b_q5",
        engine: LocalEngineKind::AudioCpp,
        architecture: "IBM Granite CTC (Non-Whisper)",
        tier: "non-whisper-enterprise",
        label: "[Non-Whisper] IBM Granite Speech 3.3 2B (Q5_K_M) — Enterprise TurboCTC",
        repo: "audio-cpp/audio.cpp-gguf",
        file: "granite-speech-3.3-2b-q5_k_m.gguf",
        url: "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Granite-Speech-3.3-2B-GGUF/granite-speech-3.3-2b-q5_k_m.gguf",
        size_mb: 1420,
        multilingual: true,
        recommended: false,
        hf_gated_or_hub: true,
    },
    LocalWhisperSpec {
        id: "voxtral_mini_3b_q4",
        engine: LocalEngineKind::AudioCpp,
        architecture: "Mistral Voxtral (Non-Whisper)",
        tier: "non-whisper-voxtral",
        label: "[Non-Whisper] Mistral Voxtral Mini 3B (Q4_K_M) — Open-Weights Speech LLM",
        repo: "audio-cpp/audio.cpp-gguf",
        file: "voxtral-mini-3b-2507-q4_k_m.gguf",
        url: "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Voxtral-Mini-3B-GGUF/voxtral-mini-3b-2507-q4_k_m.gguf",
        size_mb: 1950,
        multilingual: true,
        recommended: false,
        hf_gated_or_hub: true,
    },

    // ── Local Whisper Architectures (whisper.cpp GGML) ───────────────────────
    LocalWhisperSpec {
        id: "tiny_q5",
        engine: LocalEngineKind::WhisperCpp,
        architecture: "Whisper GGML",
        tier: "ultra-fast",
        label: "Whisper Tiny (Q5_1) — Ultra-Fast Multilingual",
        repo: "ggerganov/whisper.cpp",
        file: "ggml-tiny-q5_1.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny-q5_1.bin",
        size_mb: 32,
        multilingual: true,
        recommended: false,
        hf_gated_or_hub: false,
    },
    LocalWhisperSpec {
        id: "tiny_en_q5",
        engine: LocalEngineKind::WhisperCpp,
        architecture: "Whisper GGML",
        tier: "ultra-fast",
        label: "Whisper Tiny.en (Q5_1) — Ultra-Fast English",
        repo: "ggerganov/whisper.cpp",
        file: "ggml-tiny.en-q5_1.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q5_1.bin",
        size_mb: 32,
        multilingual: false,
        recommended: false,
        hf_gated_or_hub: false,
    },
    LocalWhisperSpec {
        id: "base_q5",
        engine: LocalEngineKind::WhisperCpp,
        architecture: "Whisper GGML",
        tier: "fast",
        label: "Whisper Base (Q5_1) — Fast Multilingual",
        repo: "ggerganov/whisper.cpp",
        file: "ggml-base-q5_1.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q5_1.bin",
        size_mb: 60,
        multilingual: true,
        recommended: false,
        hf_gated_or_hub: false,
    },
    LocalWhisperSpec {
        id: "base_en_q5",
        engine: LocalEngineKind::WhisperCpp,
        architecture: "Whisper GGML",
        tier: "fast",
        label: "Whisper Base.en (Q5_1) — Fast English",
        repo: "ggerganov/whisper.cpp",
        file: "ggml-base.en-q5_1.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en-q5_1.bin",
        size_mb: 60,
        multilingual: false,
        recommended: false,
        hf_gated_or_hub: false,
    },
    LocalWhisperSpec {
        id: "small_q5",
        engine: LocalEngineKind::WhisperCpp,
        architecture: "Whisper GGML",
        tier: "balanced",
        label: "Whisper Small (Q5_1) — Balanced Multilingual",
        repo: "ggerganov/whisper.cpp",
        file: "ggml-small-q5_1.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin",
        size_mb: 190,
        multilingual: true,
        recommended: true,
        hf_gated_or_hub: false,
    },
    LocalWhisperSpec {
        id: "small_en_q5",
        engine: LocalEngineKind::WhisperCpp,
        architecture: "Whisper GGML",
        tier: "balanced",
        label: "Whisper Small.en (Q5_1) — Balanced English",
        repo: "ggerganov/whisper.cpp",
        file: "ggml-small.en-q5_1.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en-q5_1.bin",
        size_mb: 190,
        multilingual: false,
        recommended: false,
        hf_gated_or_hub: false,
    },
    LocalWhisperSpec {
        id: "distil_small_en",
        engine: LocalEngineKind::WhisperCpp,
        architecture: "Distil-Whisper GGML",
        tier: "fast-en",
        label: "Distil-Whisper Small.en — High-Speed English",
        repo: "distil-whisper/distil-small.en",
        file: "ggml-distil-small.en.bin",
        url: "https://huggingface.co/distil-whisper/distil-small.en/resolve/main/ggml-model-bin",
        size_mb: 336,
        multilingual: false,
        recommended: false,
        hf_gated_or_hub: false,
    },
    LocalWhisperSpec {
        id: "medium_q5",
        engine: LocalEngineKind::WhisperCpp,
        architecture: "Whisper GGML",
        tier: "high",
        label: "Whisper Medium (Q5_0) — High-Accuracy Multilingual",
        repo: "ggerganov/whisper.cpp",
        file: "ggml-medium-q5_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin",
        size_mb: 539,
        multilingual: true,
        recommended: false,
        hf_gated_or_hub: false,
    },
    LocalWhisperSpec {
        id: "medium_en_q5",
        engine: LocalEngineKind::WhisperCpp,
        architecture: "Whisper GGML",
        tier: "high",
        label: "Whisper Medium.en (Q5_0) — High-Accuracy English",
        repo: "ggerganov/whisper.cpp",
        file: "ggml-medium.en-q5_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en-q5_0.bin",
        size_mb: 539,
        multilingual: false,
        recommended: false,
        hf_gated_or_hub: false,
    },
    LocalWhisperSpec {
        id: "large_v3_turbo_q5",
        engine: LocalEngineKind::WhisperCpp,
        architecture: "Whisper GGML",
        tier: "best",
        label: "Whisper Large v3 Turbo (Q5_0) — Best Speed & Accuracy",
        repo: "ggerganov/whisper.cpp",
        file: "ggml-large-v3-turbo-q5_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
        size_mb: 574,
        multilingual: true,
        recommended: true,
        hf_gated_or_hub: false,
    },
    LocalWhisperSpec {
        id: "large_v3_turbo_q8",
        engine: LocalEngineKind::WhisperCpp,
        architecture: "Whisper GGML",
        tier: "best-q8",
        label: "Whisper Large v3 Turbo (Q8_0) — Near-Lossless Turbo",
        repo: "ggerganov/whisper.cpp",
        file: "ggml-large-v3-turbo-q8_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q8_0.bin",
        size_mb: 874,
        multilingual: true,
        recommended: false,
        hf_gated_or_hub: false,
    },
    LocalWhisperSpec {
        id: "distil_large_v3",
        engine: LocalEngineKind::WhisperCpp,
        architecture: "Distil-Whisper GGML",
        tier: "best-en",
        label: "Distil-Whisper Large v3 — Fast Large English",
        repo: "distil-whisper/distil-large-v3-ggml",
        file: "ggml-distil-large-v3.bin",
        url: "https://huggingface.co/distil-whisper/distil-large-v3-ggml/resolve/main/ggml-distil-large-v3.bin",
        size_mb: 756,
        multilingual: false,
        recommended: false,
        hf_gated_or_hub: false,
    },
    LocalWhisperSpec {
        id: "large_v3_q5",
        engine: LocalEngineKind::WhisperCpp,
        architecture: "Whisper GGML",
        tier: "max",
        label: "Whisper Large v3 (Q5_0) — Maximum Multilingual Accuracy",
        repo: "ggerganov/whisper.cpp",
        file: "ggml-large-v3-q5_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-q5_0.bin",
        size_mb: 1080,
        multilingual: true,
        recommended: false,
        hf_gated_or_hub: false,
    },
    LocalWhisperSpec {
        id: "kotoba_v2_q5",
        engine: LocalEngineKind::WhisperCpp,
        architecture: "Kotoba-Whisper GGML",
        tier: "ja",
        label: "Kotoba-Whisper v2.0 (Q5_0) — Japanese Specialist",
        repo: "kotoba-tech/kotoba-whisper-v2.0-ggml",
        file: "ggml-kotoba-whisper-v2.0-q5_0.bin",
        url: "https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-ggml/resolve/main/ggml-kotoba-whisper-v2.0-q5_0.bin",
        size_mb: 538,
        multilingual: false,
        recommended: false,
        hf_gated_or_hub: true,
    },
];

pub fn find_local_whisper_spec(id: &str) -> Option<&'static LocalWhisperSpec> {
    LOCAL_WHISPER_CATALOG.iter().find(|m| m.id == id)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// COMMERCIAL & HUGGING FACE CLOUD STT PROVIDERS & MODELS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommercialSttProtocol {
    /// Hugging Face Serverless Inference Router (`POST https://router.huggingface.co/hf-inference/models/{model}`)
    /// with `Authorization: Bearer <HF_TOKEN>` and raw `audio/wav` bytes.
    HuggingFaceInference,
    /// OpenAI `/v1/audio/transcriptions` multipart form (`file`, `model`, `language`).
    /// Used by OpenAI, Groq, Mistral Voxtral, Fireworks, Together, and Custom/Self-hosted.
    OpenAiMultipart,
    /// Deepgram `/v1/listen?model=...&smart_format=true` raw audio POST with `Authorization: Token <key>`.
    Deepgram,
    /// ElevenLabs `/v1/speech-to-text` multipart form (`file`, `model_id`, `language_code`) with `xi-api-key`.
    ElevenLabsScribe,
    /// Cartesia Ink `POST https://api.cartesia.ai/stt` multipart form (`file`, `model`, `language`) with `X-API-Key` & `Cartesia-Version`.
    CartesiaInk,
    /// Soniox STT v6 `POST /v1/files` + `POST /v1/transcriptions` (`stt-async-v6`).
    Soniox,
    /// Google Gemini `generateContent` with `inlineData` base64 WAV + transcription prompt.
    GeminiAudio,
    /// AssemblyAI `/v2/upload` + `/v2/transcript` (polls until completed).
    AssemblyAi,
    /// Gladia `/v2/upload` + `/v2/pre-recorded`.
    Gladia,
    /// Speechmatics `/v2/jobs` batch transcription.
    Speechmatics,
    /// Rev AI `/speechtotext/v1/jobs` transcription.
    RevAi,
}

#[derive(Clone, Debug, Serialize)]
pub struct CommercialModelSpec {
    pub id: &'static str,
    pub label: &'static str,
    pub note: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct CommercialProviderSpec {
    pub id: &'static str,
    pub name: &'static str,
    pub protocol: CommercialSttProtocol,
    pub default_base_url: &'static str,
    pub default_model: &'static str,
    pub env_vars: &'static [&'static str],
    pub models: &'static [CommercialModelSpec],
}

pub const COMMERCIAL_STT_PROVIDERS: &[CommercialProviderSpec] = &[
    CommercialProviderSpec {
        id: "huggingface",
        name: "Hugging Face Hub (Non-Whisper & Open ASR)",
        protocol: CommercialSttProtocol::HuggingFaceInference,
        default_base_url: "https://router.huggingface.co/hf-inference/models",
        default_model: "Qwen/Qwen3-ASR-1.7B",
        env_vars: &["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"],
        models: &[
            CommercialModelSpec {
                id: "Qwen/Qwen3-ASR-1.7B",
                label: "[Non-Whisper] Alibaba Qwen3-ASR 1.7B",
                note: "#1 Hugging Face Open-ASR Leaderboard, 30 languages & 22 dialects",
            },
            CommercialModelSpec {
                id: "Qwen/Qwen3-ASR-0.6B",
                label: "[Non-Whisper] Alibaba Qwen3-ASR 0.6B",
                note: "Fast non-Whisper Apache-2.0 multilingual ASR",
            },
            CommercialModelSpec {
                id: "nvidia/parakeet-tdt-0.6b-v3",
                label: "[Non-Whisper] NVIDIA Parakeet-TDT 0.6B v3",
                note: "Blazing-fast Token-and-Duration Transducer (NeMo TDT)",
            },
            CommercialModelSpec {
                id: "nvidia/canary-1b-v2",
                label: "[Non-Whisper] NVIDIA Canary 1B v2",
                note: "Multilingual transcription & speech translation powerhouse",
            },
            CommercialModelSpec {
                id: "FunAudioLLM/SenseVoiceSmall",
                label: "[Non-Whisper] Alibaba SenseVoiceSmall",
                note: "Non-autoregressive ultra-low-latency ASR + emotion & event tagging",
            },
            CommercialModelSpec {
                id: "UsefulSensors/moonshine-base",
                label: "[Non-Whisper] Useful Sensors Moonshine Base",
                note: "Rotary-transformer non-Whisper ASR optimized for short utterances",
            },
            CommercialModelSpec {
                id: "ibm-granite/granite-speech-3.3-2b",
                label: "[Non-Whisper] IBM Granite Speech 3.3 2B",
                note: "Enterprise CTC speech recognition model",
            },
            CommercialModelSpec {
                id: "microsoft/Phi-4-multimodal-instruct",
                label: "[Non-Whisper] Microsoft Phi-4 Multimodal Speech",
                note: "Conformer speech encoder + Phi-4-mini Mixture-of-LoRAs",
            },
            CommercialModelSpec {
                id: "facebook/seamless-m4t-v2-large",
                label: "[Non-Whisper] Meta SeamlessM4T v2 Large",
                note: "Meta 100-language unified speech-to-text & translation",
            },
            CommercialModelSpec {
                id: "facebook/mms-1b-all",
                label: "[Non-Whisper] Meta MMS-1B (1,100+ Languages)",
                note: "Massively Multilingual Speech Wav2Vec2 architecture",
            },
        ],
    },
    CommercialProviderSpec {
        id: "openai",
        name: "OpenAI Audio",
        protocol: CommercialSttProtocol::OpenAiMultipart,
        default_base_url: "https://api.openai.com/v1",
        default_model: "gpt-4o-transcribe",
        env_vars: &["OPENAI_API_KEY"],
        models: &[
            CommercialModelSpec {
                id: "gpt-4o-transcribe",
                label: "GPT-4o Transcribe",
                note: "Flagship commercial speech-to-text, lowest word error rate",
            },
            CommercialModelSpec {
                id: "gpt-4o-mini-transcribe",
                label: "GPT-4o Mini Transcribe",
                note: "Fast, low-cost GPT-4o speech recognition",
            },
            CommercialModelSpec {
                id: "whisper-1",
                label: "OpenAI Whisper-1",
                note: "Classic OpenAI Whisper v2/v3 cloud endpoint",
            },
        ],
    },
    CommercialProviderSpec {
        id: "groq",
        name: "Groq Cloud LPU",
        protocol: CommercialSttProtocol::OpenAiMultipart,
        default_base_url: "https://api.groq.com/openai/v1",
        default_model: "whisper-large-v3-turbo",
        env_vars: &["GROQ_API_KEY"],
        models: &[
            CommercialModelSpec {
                id: "whisper-large-v3-turbo",
                label: "Whisper Large v3 Turbo (Groq LPU)",
                note: "216x real-time speed, multilingual, sub-300ms latency",
            },
            CommercialModelSpec {
                id: "whisper-large-v3",
                label: "Whisper Large v3 (Groq LPU)",
                note: "Maximum multilingual accuracy on Groq LPU",
            },
            CommercialModelSpec {
                id: "distil-whisper-large-v3-en",
                label: "Distil-Whisper Large v3 EN (Groq LPU)",
                note: "250x real-time speed, ultra-low cost English",
            },
        ],
    },
    CommercialProviderSpec {
        id: "deepgram",
        name: "Deepgram (Non-Whisper Nova-3 & Flux)",
        protocol: CommercialSttProtocol::Deepgram,
        default_base_url: "https://api.deepgram.com/v1",
        default_model: "nova-3",
        env_vars: &["DEEPGRAM_API_KEY"],
        models: &[
            CommercialModelSpec {
                id: "nova-3",
                label: "Deepgram Nova-3",
                note: "Flagship commercial non-Whisper ASR with smart formatting",
            },
            CommercialModelSpec {
                id: "flux",
                label: "Deepgram Flux",
                note: "Next-gen multilingual conversational turn-detection ASR",
            },
            CommercialModelSpec {
                id: "nova-3-medical",
                label: "Deepgram Nova-3 Medical",
                note: "Clinical and medical terminology dictation",
            },
            CommercialModelSpec {
                id: "nova-2",
                label: "Deepgram Nova-2 General",
                note: "Fast 36+ language commercial transcription",
            },
            CommercialModelSpec {
                id: "nova-2-meeting",
                label: "Deepgram Nova-2 Meeting",
                note: "Optimized for multi-speaker rooms and far-field mics",
            },
            CommercialModelSpec {
                id: "nova-2-finance",
                label: "Deepgram Nova-2 Finance",
                note: "Earnings, financial terms, and tickers",
            },
        ],
    },
    CommercialProviderSpec {
        id: "elevenlabs",
        name: "ElevenLabs Scribe (Non-Whisper)",
        protocol: CommercialSttProtocol::ElevenLabsScribe,
        default_base_url: "https://api.elevenlabs.io/v1",
        default_model: "scribe_v3",
        env_vars: &["ELEVENLABS_API_KEY", "XI_API_KEY"],
        models: &[
            CommercialModelSpec {
                id: "scribe_v3",
                label: "ElevenLabs Scribe v3",
                note: "Latest flagship (102+ languages, 27% lower WER than Scribe v2)",
            },
            CommercialModelSpec {
                id: "scribe_v2",
                label: "ElevenLabs Scribe v2",
                note: "90+ language commercial ASR with keyterm & entity detection",
            },
            CommercialModelSpec {
                id: "scribe_v1",
                label: "ElevenLabs Scribe v1",
                note: "Original 99-language Scribe model",
            },
        ],
    },
    CommercialProviderSpec {
        id: "mistral",
        name: "Mistral AI (Voxtral Non-Whisper)",
        protocol: CommercialSttProtocol::OpenAiMultipart,
        default_base_url: "https://api.mistral.ai/v1",
        default_model: "voxtral-mini-transcribe-3",
        env_vars: &["MISTRAL_API_KEY"],
        models: &[
            CommercialModelSpec {
                id: "voxtral-mini-transcribe-3",
                label: "Voxtral Mini Transcribe 3 (Voxtral 3)",
                note: "Mistral's latest Voxtral 3 high-speed multilingual transcription",
            },
            CommercialModelSpec {
                id: "voxtral-mini-latest",
                label: "Voxtral Mini Latest (voxtral-mini-2507)",
                note: "Mistral's fast multilingual speech transcription model",
            },
            CommercialModelSpec {
                id: "voxtral-small-latest",
                label: "Voxtral Small 24B",
                note: "Mistral's high-accuracy commercial audio model",
            },
        ],
    },
    CommercialProviderSpec {
        id: "cartesia",
        name: "Cartesia Ink",
        protocol: CommercialSttProtocol::CartesiaInk,
        default_base_url: "https://api.cartesia.ai",
        default_model: "ink-whisper",
        env_vars: &["CARTESIA_API_KEY"],
        models: &[
            CommercialModelSpec {
                id: "ink-whisper",
                label: "Cartesia Ink-Whisper",
                note: "High-accuracy structured dictation (emails, UUIDs, numbers, code)",
            },
            CommercialModelSpec {
                id: "ink-2",
                label: "Cartesia Ink-2",
                note: "Low-latency conversational & voice-agent speech recognition",
            },
        ],
    },
    CommercialProviderSpec {
        id: "soniox",
        name: "Soniox STT v6 (Non-Whisper)",
        protocol: CommercialSttProtocol::Soniox,
        default_base_url: "https://api.soniox.com/v1",
        default_model: "stt-async-v6",
        env_vars: &["SONIOX_API_KEY"],
        models: &[
            CommercialModelSpec {
                id: "stt-async-v6",
                label: "Soniox STT v6 Async",
                note: "60+ languages with native code-switching & currency/number formatting",
            },
            CommercialModelSpec {
                id: "stt-rt-v6",
                label: "Soniox STT v6 Realtime",
                note: "Ultra-low-latency voice typing & conversational turn model",
            },
        ],
    },
    CommercialProviderSpec {
        id: "gemini",
        name: "Google Gemini Audio (Non-Whisper)",
        protocol: CommercialSttProtocol::GeminiAudio,
        default_base_url: "https://generativelanguage.googleapis.com/v1beta",
        default_model: "gemini-2.5-flash",
        env_vars: &["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        models: &[
            CommercialModelSpec {
                id: "gemini-2.5-flash",
                label: "Gemini 2.5 Flash Audio",
                note: "Fast native multimodal speech transcription & punctuation",
            },
            CommercialModelSpec {
                id: "gemini-2.5-pro",
                label: "Gemini 2.5 Pro Audio",
                note: "Highest accuracy for technical jargon & code dictation",
            },
            CommercialModelSpec {
                id: "gemini-2.0-flash",
                label: "Gemini 2.0 Flash Audio",
                note: "Low-latency multimodal transcription",
            },
        ],
    },
    CommercialProviderSpec {
        id: "assemblyai",
        name: "AssemblyAI (Non-Whisper Universal-3.6 & Slam-1)",
        protocol: CommercialSttProtocol::AssemblyAi,
        default_base_url: "https://api.assemblyai.com/v2",
        default_model: "universal-3.6-pro",
        env_vars: &["ASSEMBLYAI_API_KEY"],
        models: &[
            CommercialModelSpec {
                id: "universal-3.6-pro",
                label: "AssemblyAI Universal-3.6 Pro",
                note: "Top-ranked lowest Word Error Rate commercial speech model",
            },
            CommercialModelSpec {
                id: "universal",
                label: "AssemblyAI Universal-2",
                note: "Broad 99-language commercial speech recognition",
            },
            CommercialModelSpec {
                id: "slam-1",
                label: "AssemblyAI Slam-1",
                note: "Speech LLM Audio Model for high contextual accuracy",
            },
            CommercialModelSpec {
                id: "nano",
                label: "AssemblyAI Nano",
                note: "Ultra-low-cost commercial transcription tier",
            },
        ],
    },
    CommercialProviderSpec {
        id: "gladia",
        name: "Gladia (Solaria-3)",
        protocol: CommercialSttProtocol::Gladia,
        default_base_url: "https://api.gladia.io/v2",
        default_model: "solaria-3",
        env_vars: &["GLADIA_API_KEY"],
        models: &[
            CommercialModelSpec {
                id: "solaria-3",
                label: "Gladia Solaria-3",
                note: "Latest flagship 100+ language ASR with instant code-switching",
            },
            CommercialModelSpec {
                id: "solaria-1",
                label: "Gladia Solaria-1",
                note: "Enterprise multilingual ASR",
            },
            CommercialModelSpec {
                id: "whisper-zero",
                label: "Gladia Whisper-Zero",
                note: "Zero-hallucination tuned commercial Whisper",
            },
        ],
    },
    CommercialProviderSpec {
        id: "fireworks",
        name: "Fireworks AI Audio",
        protocol: CommercialSttProtocol::OpenAiMultipart,
        default_base_url: "https://audio-turbo.us-virginia-1.direct.fireworks.ai/v1",
        default_model: "whisper-v3-turbo",
        env_vars: &["FIREWORKS_API_KEY"],
        models: &[
            CommercialModelSpec {
                id: "whisper-v3-turbo",
                label: "Fireworks Whisper v3 Turbo",
                note: "Ultra-fast serverless Whisper v3 Turbo",
            },
            CommercialModelSpec {
                id: "whisper-v3",
                label: "Fireworks Whisper v3",
                note: "Full Whisper Large v3 on Fireworks",
            },
        ],
    },
    CommercialProviderSpec {
        id: "together",
        name: "Together AI Audio",
        protocol: CommercialSttProtocol::OpenAiMultipart,
        default_base_url: "https://api.together.xyz/v1",
        default_model: "openai/whisper-large-v3",
        env_vars: &["TOGETHER_API_KEY"],
        models: &[CommercialModelSpec {
            id: "openai/whisper-large-v3",
            label: "Together Whisper Large v3",
            note: "OpenAI Whisper Large v3 hosted on Together AI",
        }],
    },
    CommercialProviderSpec {
        id: "speechmatics",
        name: "Speechmatics (Non-Whisper Ursa-2)",
        protocol: CommercialSttProtocol::Speechmatics,
        default_base_url: "https://asr.api.speechmatics.com/v2",
        default_model: "enhanced",
        env_vars: &["SPEECHMATICS_API_KEY"],
        models: &[
            CommercialModelSpec {
                id: "enhanced",
                label: "Speechmatics Enhanced (Ursa-2)",
                note: "Enterprise noise resilience & 56-language accent accuracy",
            },
            CommercialModelSpec {
                id: "standard",
                label: "Speechmatics Standard",
                note: "Fast commercial transcription",
            },
        ],
    },
    CommercialProviderSpec {
        id: "revai",
        name: "Rev AI (Non-Whisper Reverb)",
        protocol: CommercialSttProtocol::RevAi,
        default_base_url: "https://api.rev.ai/speechtotext/v1",
        default_model: "reverb",
        env_vars: &["REVAI_API_KEY"],
        models: &[
            CommercialModelSpec {
                id: "reverb",
                label: "Rev AI Reverb",
                note: "Specialized high-accuracy English & media ASR",
            },
            CommercialModelSpec {
                id: "machine_v2",
                label: "Rev AI Reverie v2",
                note: "Commercial verbatim & punctuated ASR",
            },
            CommercialModelSpec {
                id: "fusion",
                label: "Rev AI Fusion",
                note: "Multi-engine ensemble ASR",
            },
        ],
    },
    CommercialProviderSpec {
        id: "custom",
        name: "Self-Hosted / Local Server (vLLM / Speaches / Sherpa-ONNX)",
        protocol: CommercialSttProtocol::OpenAiMultipart,
        default_base_url: "http://127.0.0.1:8000/v1",
        default_model: "Qwen/Qwen3-ASR-1.7B",
        env_vars: &["CUSTOM_STT_API_KEY", "AZURE_OPENAI_API_KEY"],
        models: &[
            CommercialModelSpec {
                id: "Qwen/Qwen3-ASR-1.7B",
                label: "Alibaba Qwen3-ASR 1.7B (via vLLM / OpenAI Server)",
                note: "Top non-Whisper multilingual ASR served locally",
            },
            CommercialModelSpec {
                id: "nvidia/parakeet-tdt-0.6b-v3",
                label: "NVIDIA Parakeet-TDT 0.6B v3 (via Speaches / ONNX Server)",
                note: "Blazing-fast Token-and-Duration Transducer local server model",
            },
            CommercialModelSpec {
                id: "nvidia/canary-1b-v2",
                label: "NVIDIA Canary 1B v2 (via Speaches / NeMo Server)",
                note: "Top open multilingual transcription & translation model",
            },
            CommercialModelSpec {
                id: "FunAudioLLM/SenseVoiceSmall",
                label: "Alibaba SenseVoiceSmall (via FunASR / Speaches Server)",
                note: "Non-autoregressive ASR with emotion & event detection",
            },
            CommercialModelSpec {
                id: "usefulsensors/moonshine-base",
                label: "Moonshine Base (via Speaches / Edge Server)",
                note: "Ultra-lightweight edge dictation model by Useful Sensors",
            },
            CommercialModelSpec {
                id: "Qwen/Qwen2-Audio-7B-Instruct",
                label: "Qwen2-Audio 7B (via vLLM / OpenAI Server)",
                note: "Alibaba end-to-end multimodal audio-language model",
            },
        ],
    },
];

pub fn find_commercial_provider(id: &str) -> Option<&'static CommercialProviderSpec> {
    COMMERCIAL_STT_PROVIDERS.iter().find(|p| p.id == id)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// COMMERCIAL TTS (TEXT-TO-SPEECH / READ-ALOUD) CATALOG
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Clone, Debug, Serialize)]
pub struct CommercialTtsSpec {
    pub provider_id: &'static str,
    pub provider_name: &'static str,
    pub model_id: &'static str,
    pub label: &'static str,
    pub default_voice: &'static str,
    pub base_url: &'static str,
    pub env_vars: &'static [&'static str],
}

pub const COMMERCIAL_TTS_CATALOG: &[CommercialTtsSpec] = &[
    CommercialTtsSpec {
        provider_id: "openai",
        provider_name: "OpenAI TTS",
        model_id: "gpt-4o-mini-tts",
        label: "OpenAI GPT-4o Mini TTS (Steerable)",
        default_voice: "coral",
        base_url: "https://api.openai.com/v1",
        env_vars: &["OPENAI_API_KEY"],
    },
    CommercialTtsSpec {
        provider_id: "openai",
        provider_name: "OpenAI TTS HD",
        model_id: "tts-1-hd",
        label: "OpenAI TTS-1 HD",
        default_voice: "nova",
        base_url: "https://api.openai.com/v1",
        env_vars: &["OPENAI_API_KEY"],
    },
    CommercialTtsSpec {
        provider_id: "elevenlabs",
        provider_name: "ElevenLabs",
        model_id: "eleven_multilingual_v2",
        label: "ElevenLabs Multilingual v2",
        default_voice: "JBFqnCBsd6RMkjVDRZzb",
        base_url: "https://api.elevenlabs.io/v1",
        env_vars: &["ELEVENLABS_API_KEY", "XI_API_KEY"],
    },
    CommercialTtsSpec {
        provider_id: "elevenlabs",
        provider_name: "ElevenLabs Flash",
        model_id: "eleven_flash_v2_5",
        label: "ElevenLabs Flash v2.5 (75ms Ultra-Low Latency)",
        default_voice: "JBFqnCBsd6RMkjVDRZzb",
        base_url: "https://api.elevenlabs.io/v1",
        env_vars: &["ELEVENLABS_API_KEY", "XI_API_KEY"],
    },
    CommercialTtsSpec {
        provider_id: "cartesia",
        provider_name: "Cartesia Sonic-3.6",
        model_id: "sonic-3.6",
        label: "Cartesia Sonic-3.6 (Sub-90ms 44-Language Streaming TTS)",
        default_voice: "a0e99841-438c-4a64-b679-ae501e7d6091",
        base_url: "https://api.cartesia.ai",
        env_vars: &["CARTESIA_API_KEY"],
    },
    CommercialTtsSpec {
        provider_id: "mistral",
        provider_name: "Mistral Voxtral TTS 3",
        model_id: "voxtral-mini-tts-2603",
        label: "Mistral Voxtral TTS 3 (Multilingual + Voice Cloning)",
        default_voice: "default",
        base_url: "https://api.mistral.ai/v1",
        env_vars: &["MISTRAL_API_KEY"],
    },
    CommercialTtsSpec {
        provider_id: "deepgram",
        provider_name: "Deepgram Aura-2",
        model_id: "aura-2-thalia-en",
        label: "Deepgram Aura-2 Thalia (Enterprise Voice)",
        default_voice: "aura-2-thalia-en",
        base_url: "https://api.deepgram.com/v1",
        env_vars: &["DEEPGRAM_API_KEY"],
    },
    CommercialTtsSpec {
        provider_id: "groq",
        provider_name: "Groq PlayAI TTS",
        model_id: "playai-tts",
        label: "Groq PlayAI Dialog TTS (LPU Accelerated)",
        default_voice: "Arista-PlayAI",
        base_url: "https://api.groq.com/openai/v1",
        env_vars: &["GROQ_API_KEY"],
    },
];

/// Flat unified model selector entry so the UI can browse both Local (Non-Whisper + Whisper)
/// and Commercial / Hugging Face STT models in one picker.
#[derive(Clone, Debug)]
pub struct UnifiedSttChoice {
    pub backend: String,
    pub provider_label: String,
    pub model_id: String,
    pub label: String,
    pub detail: String,
    pub is_local: bool,
}

pub fn all_stt_choices() -> Vec<UnifiedSttChoice> {
    let mut list = Vec::new();
    for spec in LOCAL_WHISPER_CATALOG {
        let engine_tag = match spec.engine {
            LocalEngineKind::AudioCpp => "Local audio.cpp · Non-Whisper",
            LocalEngineKind::WhisperCpp => "Local whisper.cpp",
        };
        list.push(UnifiedSttChoice {
            backend: "local".to_string(),
            provider_label: engine_tag.to_string(),
            model_id: spec.id.to_string(),
            label: spec.label.to_string(),
            detail: format!(
                "{} · {} MB · {}",
                spec.architecture,
                spec.size_mb,
                if spec.multilingual {
                    "multilingual"
                } else {
                    "english"
                }
            ),
            is_local: true,
        });
    }
    for prov in COMMERCIAL_STT_PROVIDERS {
        for model in prov.models {
            list.push(UnifiedSttChoice {
                backend: prov.id.to_string(),
                provider_label: prov.name.to_string(),
                model_id: model.id.to_string(),
                label: format!("{} · {}", prov.name, model.label),
                detail: model.note.to_string(),
                is_local: false,
            });
        }
    }
    list
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_contains_non_whisper_local_and_huggingface_models() {
        assert!(LOCAL_WHISPER_CATALOG.len() >= 23);
        assert!(COMMERCIAL_STT_PROVIDERS.len() >= 16);
        let choices = all_stt_choices();
        // Non-Whisper local models
        assert!(choices
            .iter()
            .any(|c| c.is_local && c.model_id == "qwen3_asr_0_6b_q8"));
        assert!(choices
            .iter()
            .any(|c| c.is_local && c.model_id == "parakeet_tdt_0_6b_v3"));
        assert!(choices
            .iter()
            .any(|c| c.is_local && c.model_id == "sensevoice_small_q8"));
        assert!(choices
            .iter()
            .any(|c| c.is_local && c.model_id == "moonshine_base_q8"));
        // Hugging Face non-Whisper cloud models
        assert!(choices
            .iter()
            .any(|c| c.backend == "huggingface" && c.model_id == "Qwen/Qwen3-ASR-1.7B"));
        assert!(choices
            .iter()
            .any(|c| c.backend == "huggingface" && c.model_id == "nvidia/parakeet-tdt-0.6b-v3"));
        assert!(choices
            .iter()
            .any(|c| c.backend == "huggingface" && c.model_id == "FunAudioLLM/SenseVoiceSmall"));
    }
}
