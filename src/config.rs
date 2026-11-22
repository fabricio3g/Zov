//! Configuration, paths (standalone + automatic Agente Desktop discovery),
//! commercial provider credentials, and dictation history persistence.

use crate::catalog::find_commercial_provider;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct AppPaths {
    pub root: PathBuf,
    pub config_path: PathBuf,
    pub history_path: PathBuf,
    pub whisper_bin_root: PathBuf,
    pub audio_bin_root: PathBuf,
    pub whisper_models_root: PathBuf,
    pub cache_root: PathBuf,
    /// Shared Agente Desktop paths so already-downloaded Whisper/audio.cpp engines, models & GPU plugins work immediately.
    pub shared_whisper_bin_roots: Vec<PathBuf>,
    pub shared_audio_bin_roots: Vec<PathBuf>,
    pub shared_whisper_model_roots: Vec<PathBuf>,
    pub shared_llama_bin_roots: Vec<PathBuf>,
}

impl AppPaths {
    pub fn resolve() -> Self {
        let base_data = dirs::data_dir()
            .or_else(dirs::config_dir)
            .unwrap_or_else(|| PathBuf::from("."));

        let root = base_data.join("Enio");
        let whisper_bin_root = root.join("engines").join("whisper.cpp");
        let audio_bin_root = root.join("engines").join("audio.cpp");
        let whisper_models_root = root.join("models").join("whisper");
        let cache_root = root.join("cache");

        // Also discover existing installs in AgenteDictate / Agente so previously downloaded
        // models, engines, and GPU plugins work immediately with zero re-downloads.
        let legacy_dictate = base_data.join("AgenteDictate");
        let agente_root = base_data.join("Agente");
        let shared_whisper_bin_roots = vec![
            whisper_bin_root.clone(),
            legacy_dictate.join("engines").join("whisper.cpp"),
            agente_root.join("engines").join("whisper.cpp"),
            agente_root.join("engines").join("whisper.cpp").join("default"),
        ];
        let shared_audio_bin_roots = vec![
            audio_bin_root.clone(),
            legacy_dictate.join("engines").join("audio.cpp"),
            agente_root.join("engines").join("audio.cpp"),
        ];
        let shared_whisper_model_roots = vec![
            whisper_models_root.clone(),
            legacy_dictate.join("models").join("whisper"),
            agente_root.join("models").join("speech").join("whisper"),
            agente_root.join("models").join("speech"),
            agente_root.join("models").join("whisper"),
        ];
        let shared_llama_bin_roots = vec![
            root.join("engines").join("llama.cpp"),
            legacy_dictate.join("engines").join("llama.cpp"),
            agente_root.join("engines").join("llama.cpp"),
            agente_root.join("engines").join("llama.cpp").join("default"),
        ];

        Self {
            config_path: root.join("config.json"),
            history_path: root.join("history.json"),
            root,
            whisper_bin_root,
            audio_bin_root,
            whisper_models_root,
            cache_root,
            shared_whisper_bin_roots,
            shared_audio_bin_roots,
            shared_whisper_model_roots,
            shared_llama_bin_roots,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryMode {
    /// Copies transcript to clipboard and types Unicode characters into the active window.
    TypeAndClipboard,
    /// Copies transcript to clipboard and sends Ctrl+V into the active window.
    PasteAndClipboard,
    /// Only copies transcript to clipboard.
    ClipboardOnly,
}

impl DeliveryMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::TypeAndClipboard => "Auto-Type + Clipboard",
            Self::PasteAndClipboard => "Auto-Paste (Ctrl+V) + Clipboard",
            Self::ClipboardOnly => "Clipboard Only",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::TypeAndClipboard => Self::PasteAndClipboard,
            Self::PasteAndClipboard => Self::ClipboardOnly,
            Self::ClipboardOnly => Self::TypeAndClipboard,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HotkeyMode {
    /// Hold Ctrl+Shift+Space to speak (shows floating overlay); release keys to transcribe & insert.
    /// Quick-tapping (< 350ms) latches open until pressed again.
    SmartHoldOrTap,
    /// Strictly Push-to-Talk: recording runs only while Ctrl+Shift+Space is held down.
    HoldToTalk,
    /// Real-time live streaming: transcribes and types each spoken phrase continuously as you pause.
    RealTimeStreaming,
}

impl Default for HotkeyMode {
    fn default() -> Self {
        Self::SmartHoldOrTap
    }
}

impl HotkeyMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::SmartHoldOrTap => "Hotkey: Hold-to-Talk / Tap",
            Self::HoldToTalk => "Hotkey: Push-to-Talk (Release)",
            Self::RealTimeStreaming => "Hotkey: Real-Time Live Stream",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::SmartHoldOrTap => Self::HoldToTalk,
            Self::HoldToTalk => Self::RealTimeStreaming,
            Self::RealTimeStreaming => Self::SmartHoldOrTap,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProviderCredentials {
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub base_url: String,
    #[serde(default)]
    pub model: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DictateConfig {
    /// `"local"` for on-device engines (`whisper.cpp` / `audio.cpp`), or commercial/HF provider ID (`"huggingface"`, `"openai"`, `"groq"`, `"deepgram"`, etc.).
    pub backend: String,
    /// Selected local model ID from `LOCAL_WHISPER_CATALOG` (e.g. `"small_q5"`, `"qwen3_asr_0_6b_q8"`, `"parakeet_tdt_0_6b_v3"`).
    pub local_model: String,
    /// Selected commercial/cloud model ID when `backend != "local"`.
    pub commercial_model: String,
    /// Spoken language code (`"auto"`, `"en"`, `"es"`, `"pt"`, `"fr"`, `"de"`, `"it"`, `"ja"`, `"zh"`).
    pub language: String,
    /// How transcribed text is delivered to the system.
    pub delivery_mode: DeliveryMode,
    /// How the global hotkey (`Ctrl+Shift+Space`) behaves (Hold-to-Talk, Smart Hold/Tap, Real-Time Live Stream).
    #[serde(default)]
    pub hotkey_mode: HotkeyMode,
    /// Hands-free auto-stop on silence via local VoiceActivity detector.
    pub vad_auto_stop: bool,
    /// Optional preferred microphone device name.
    pub audio_device: Option<String>,
    /// Explicit path to `whisper-cli` / `whisper-server` binary.
    pub runtime_path: Option<String>,
    /// Hugging Face API Token (`hf_...`) for gated Hugging Face model downloads and Hugging Face Inference Router.
    #[serde(default)]
    pub hf_token: String,
    /// Per-provider API keys and base URL overrides.
    #[serde(default)]
    pub providers: HashMap<String, ProviderCredentials>,
    /// UI theme (`"dark"` or `"light"`).
    #[serde(default = "default_theme")]
    pub theme: String,
}

fn default_theme() -> String {
    "dark".to_string()
}

impl Default for DictateConfig {
    fn default() -> Self {
        Self {
            backend: "local".to_string(),
            local_model: "small_q5".to_string(),
            commercial_model: "gpt-4o-transcribe".to_string(),
            language: "auto".to_string(),
            delivery_mode: DeliveryMode::PasteAndClipboard,
            hotkey_mode: HotkeyMode::SmartHoldOrTap,
            vad_auto_stop: false,
            audio_device: None,
            runtime_path: None,
            hf_token: String::new(),
            providers: HashMap::new(),
            theme: default_theme(),
        }
    }
}

pub const LANGUAGES: &[&str] = &["auto", "en", "es", "pt", "fr", "de", "it", "ja", "ko", "zh"];

impl DictateConfig {
    pub fn load(paths: &AppPaths) -> Self {
        if let Ok(raw) = fs::read_to_string(&paths.config_path) {
            if let Ok(cfg) = serde_json::from_str::<Self>(&raw) {
                return cfg;
            }
        }
        Self::default()
    }

    pub fn save(&self, paths: &AppPaths) {
        let _ = fs::create_dir_all(&paths.root);
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = fs::write(&paths.config_path, json);
        }
    }

    pub fn cycle_language(&mut self) {
        let idx = LANGUAGES
            .iter()
            .position(|&l| l == self.language)
            .unwrap_or(0);
        self.language = LANGUAGES[(idx + 1) % LANGUAGES.len()].to_string();
    }

    /// Resolve Hugging Face token from config (`hf_token` or `providers["huggingface"].api_key`) or env (`HF_TOKEN`).
    pub fn resolve_hf_token(&self) -> String {
        if !self.hf_token.trim().is_empty() {
            return self.hf_token.trim().to_string();
        }
        if let Some(saved) = self.providers.get("huggingface") {
            if !saved.api_key.trim().is_empty() {
                return saved.api_key.trim().to_string();
            }
        }
        for env_name in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
            if let Ok(val) = std::env::var(env_name) {
                if !val.trim().is_empty() {
                    return val.trim().to_string();
                }
            }
        }
        String::new()
    }

    /// Resolve API key, base URL, and model for a commercial or Hugging Face STT provider.
    pub fn resolve_commercial_auth(
        &self,
        provider_id: &str,
    ) -> Option<(String, String, String)> {
        let spec = find_commercial_provider(provider_id)?;
        let saved = self.providers.get(provider_id);

        let mut api_key = if provider_id == "huggingface" {
            self.resolve_hf_token()
        } else {
            saved
                .map(|c| c.api_key.trim().to_string())
                .unwrap_or_default()
        };

        if api_key.is_empty() {
            for &env_name in spec.env_vars {
                if let Ok(val) = std::env::var(env_name) {
                    if !val.trim().is_empty() {
                        api_key = val.trim().to_string();
                        break;
                    }
                }
            }
        }

        let base_url = saved
            .map(|c| c.base_url.trim())
            .filter(|u| !u.is_empty())
            .unwrap_or(spec.default_base_url)
            .trim_end_matches('/')
            .to_string();

        let model = if self.backend == provider_id && !self.commercial_model.trim().is_empty() {
            self.commercial_model.clone()
        } else {
            saved
                .map(|c| c.model.trim())
                .filter(|m| !m.is_empty())
                .unwrap_or(spec.default_model)
                .to_string()
        };

        Some((api_key, base_url, model))
    }

    pub fn has_api_key(&self, provider_id: &str) -> bool {
        if let Some((key, base_url, _)) = self.resolve_commercial_auth(provider_id) {
            if !key.is_empty() {
                return true;
            }
            // Local self-hosted endpoints (localhost / 127.0.0.1) don't require an API key
            return base_url.contains("127.0.0.1") || base_url.contains("localhost");
        }
        false
    }

    pub fn set_provider_config(
        &mut self,
        provider_id: &str,
        api_key: String,
        base_url: String,
        model: String,
    ) {
        let key = api_key.trim().to_string();
        let url = base_url.trim().trim_end_matches('/').to_string();
        let mdl = model.trim().to_string();

        if provider_id == "huggingface" {
            self.hf_token = key.clone();
        }
        let entry = self
            .providers
            .entry(provider_id.to_string())
            .or_default();
        entry.api_key = key;
        entry.base_url = url;
        if !mdl.is_empty() {
            entry.model = mdl.clone();
            if self.backend == provider_id {
                self.commercial_model = mdl;
            }
        }
    }

    pub fn active_summary(&self) -> String {
        if self.backend == "local" {
            format!("Local · {}", self.local_model)
        } else if let Some(prov) = find_commercial_provider(&self.backend) {
            format!("{} · {}", prov.name, self.commercial_model)
        } else {
            format!("{} · {}", self.backend, self.commercial_model)
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub id: String,
    pub text: String,
    pub created_at: DateTime<Utc>,
    pub duration_secs: f32,
    pub backend: String,
    pub model: String,
    pub elapsed_ms: u64,
}

pub fn load_history(paths: &AppPaths) -> Vec<HistoryEntry> {
    if let Ok(raw) = fs::read_to_string(&paths.history_path) {
        if let Ok(entries) = serde_json::from_str::<Vec<HistoryEntry>>(&raw) {
            return entries;
        }
    }
    Vec::new()
}

pub fn save_history(paths: &AppPaths, entries: &[HistoryEntry]) {
    let _ = fs::create_dir_all(&paths.root);
    if let Ok(json) = serde_json::to_string_pretty(entries) {
        let _ = fs::write(&paths.history_path, json);
    }
}
