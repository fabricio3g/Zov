//! Unified Speech-to-Text Engine:
//! - On-device `whisper.cpp` (warm `whisper-server` with borrowed GPU plugins + `whisper-cli` fallback)
//! - One-step automatic `whisper.cpp` runtime & Hugging Face model downloader
//! - All 13 Commercial Cloud STT Providers (OpenAI, Groq, Deepgram, ElevenLabs Scribe,
//!   Mistral Voxtral, Google Gemini Audio, AssemblyAI, Gladia, Fireworks, Together,
//!   Speechmatics, Rev AI, Custom/Self-hosted).

use crate::catalog::{
    find_commercial_provider, find_local_whisper_spec, CommercialSttProtocol, LocalEngineKind,
    LOCAL_WHISPER_CATALOG,
};
use crate::config::{AppPaths, DictateConfig};
use crate::vad::spoken_transcript;
use crate::wav::live_whisper_context;
use anyhow::{bail, Context, Result};
use base64::Engine as _;
use futures_util::StreamExt;
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::fs;
use std::io::Write;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

const WHISPER_CLI_NAMES: &[&str] = if cfg!(windows) {
    &["whisper-cli.exe", "whisper.exe", "main.exe"]
} else {
    &["whisper-cli", "whisper", "main"]
};

const WHISPER_SERVER_NAME: &str = if cfg!(windows) {
    "whisper-server.exe"
} else {
    "whisper-server"
};

const AUDIOCPP_SERVER_NAME: &str = if cfg!(windows) {
    "audiocpp_server.exe"
} else {
    "audiocpp_server"
};

const GPU_PLUGINS: &[&str] = if cfg!(windows) {
    &["ggml-cuda.dll", "ggml-vulkan.dll", "ggml-hip.dll"]
} else if cfg!(target_os = "macos") {
    &[]
} else {
    &["libggml-cuda.so", "libggml-vulkan.so", "libggml-hip.so"]
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MULTIPART FORM BUILDER (zero extra crate dependencies)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub fn build_multipart_form(
    fields: &[(&str, &str)],
    file_field: &str,
    filename: &str,
    mime_type: &str,
    file_bytes: &[u8],
) -> (String, Vec<u8>) {
    let boundary = format!("----AgenteDictateBoundary{}", uuid::Uuid::new_v4().simple());
    let mut body = Vec::with_capacity(file_bytes.len() + 1024);

    for &(name, value) in fields {
        let _ = write!(
            body,
            "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
        );
    }

    let _ = write!(
        body,
        "--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; filename=\"{filename}\"\r\nContent-Type: {mime_type}\r\n\r\n"
    );
    body.extend_from_slice(file_bytes);
    let _ = write!(body, "\r\n--{boundary}--\r\n");

    let content_type = format!("multipart/form-data; boundary={boundary}");
    (content_type, body)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// LOCAL WHISPER DISCOVERY (Standalone + Shared Agente Desktop folders)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn find_binary_in_dir(dir: &Path, names: &[&str]) -> Option<PathBuf> {
    if !dir.exists() {
        return None;
    }
    for &bin in names {
        let top = dir.join(bin);
        if top.is_file() {
            return Some(top);
        }
    }
    let mut found = Vec::new();
    fn walk(dir: &Path, depth: usize, names: &[&str], out: &mut Vec<PathBuf>) {
        if depth > 3 {
            return;
        }
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let full = entry.path();
            if full.is_dir() {
                walk(&full, depth + 1, names, out);
            } else if let Some(name) = full.file_name().and_then(|s| s.to_str()) {
                if names.contains(&name) {
                    out.push(full);
                }
            }
        }
    }
    walk(dir, 0, names, &mut found);
    found.sort_by_key(|p| {
        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
        names.iter().position(|&n| n == name).unwrap_or(usize::MAX)
    });
    found.into_iter().next()
}

pub fn resolve_whisper_binary(paths: &AppPaths, cfg: &DictateConfig) -> Option<PathBuf> {
    if let Some(explicit) = cfg.runtime_path.as_deref().filter(|s| !s.trim().is_empty()) {
        let p = PathBuf::from(explicit);
        if p.is_file() {
            return Some(p);
        }
    }
    for root in &paths.shared_whisper_bin_roots {
        if let Some(bin) = find_binary_in_dir(root, WHISPER_CLI_NAMES) {
            return Some(bin);
        }
    }
    None
}

pub fn find_whisper_server(cli_path: &Path) -> Option<PathBuf> {
    let dir = cli_path.parent()?;
    let candidate = dir.join(WHISPER_SERVER_NAME);
    candidate.is_file().then_some(candidate)
}

pub fn resolve_audiocpp_binary(paths: &AppPaths) -> Option<PathBuf> {
    for root in &paths.shared_audio_bin_roots {
        for sub in ["vulkan", "cuda", "cpu", ""] {
            let dir = if sub.is_empty() {
                root.clone()
            } else {
                root.join(sub)
            };
            let candidate = dir.join(AUDIOCPP_SERVER_NAME);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
        if let Some(bin) = find_binary_in_dir(root, &[AUDIOCPP_SERVER_NAME]) {
            return Some(bin);
        }
    }
    None
}

pub fn is_local_engine_ready(paths: &AppPaths, cfg: &DictateConfig) -> bool {
    let Some(spec) = find_local_whisper_spec(&cfg.local_model) else {
        return false;
    };
    let model_ok = check_local_whisper_model(paths, spec.id).is_some();
    if !model_ok {
        return false;
    }
    resolve_speech_engine(paths, cfg, spec.engine).is_some()
}

fn resolve_speech_engine(
    paths: &AppPaths,
    cfg: &DictateConfig,
    engine: LocalEngineKind,
) -> Option<PathBuf> {
    match engine {
        LocalEngineKind::WhisperCpp => resolve_whisper_binary(paths, cfg),
        LocalEngineKind::AudioCpp => resolve_audiocpp_binary(paths),
    }
}

pub fn check_local_whisper_model(paths: &AppPaths, model_id: &str) -> Option<PathBuf> {
    let spec = find_local_whisper_spec(model_id)?;
    for root in &paths.shared_whisper_model_roots {
        let direct = root.join(spec.file);
        if direct.is_file() {
            return Some(direct);
        }
        let nested = root.join("whisper-ggml").join(spec.file);
        if nested.is_file() {
            return Some(nested);
        }
    }
    None
}

pub fn installed_local_models(paths: &AppPaths) -> Vec<String> {
    LOCAL_WHISPER_CATALOG
        .iter()
        .filter(|spec| check_local_whisper_model(paths, spec.id).is_some())
        .map(|spec| spec.id.to_string())
        .collect()
}

#[derive(Default, PartialEq, Eq)]
pub struct LocalInventory {
    pub installed: Vec<String>,
    whisper: bool,
    audio: bool,
}

impl LocalInventory {
    pub fn scan(paths: &AppPaths, config: &DictateConfig) -> Self {
        Self {
            installed: installed_local_models(paths),
            whisper: resolve_whisper_binary(paths, config).is_some(),
            audio: resolve_audiocpp_binary(paths).is_some(),
        }
    }

    pub fn contains(&self, model_id: &str) -> bool {
        self.installed.iter().any(|id| id == model_id)
    }

    pub fn is_ready(&self, model_id: &str) -> bool {
        self.contains(model_id)
            && find_local_whisper_spec(model_id).is_some_and(|spec| self.has_engine(spec.engine))
    }

    pub fn has_engine(&self, engine: LocalEngineKind) -> bool {
        match engine {
            LocalEngineKind::WhisperCpp => self.whisper,
            LocalEngineKind::AudioCpp => self.audio,
        }
    }
}

fn has_own_gpu_backend(server_dir: &Path) -> bool {
    GPU_PLUGINS.iter().any(|name| server_dir.join(name).is_file())
}

fn borrowed_gpu_plugin(paths: &AppPaths, server_dir: &Path, gpu_disabled: bool) -> Option<PathBuf> {
    if gpu_disabled || has_own_gpu_backend(server_dir) {
        return None;
    }
    for root in &paths.shared_llama_bin_roots {
        if !root.exists() {
            continue;
        }
        for &plugin in GPU_PLUGINS {
            let direct = root.join(plugin);
            if direct.is_file() {
                return Some(direct);
            }
            if let Ok(entries) = fs::read_dir(root) {
                for entry in entries.flatten() {
                    let sub = entry.path().join(plugin);
                    if sub.is_file() {
                        return Some(sub);
                    }
                }
            }
        }
    }
    None
}

#[cfg(windows)]
fn hide_console_window(cmd: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    cmd.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn hide_console_window(_cmd: &mut Command) {}

pub fn has_nvidia_gpu() -> bool {
    let mut cmd = Command::new("nvidia-smi");
    cmd.args(["--query-gpu=name", "--format=csv,noheader"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    hide_console_window(&mut cmd);
    cmd.output()
        .ok()
        .is_some_and(|o| o.status.success() && !String::from_utf8_lossy(&o.stdout).trim().is_empty())
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// WARM WHISPER-SERVER MANAGER
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct RunningWhisperServer {
    child: Child,
    port: u16,
    key: String,
    last_used: Instant,
}

struct ServerState {
    active: Option<RunningWhisperServer>,
    gpu_disabled: bool,
}

static WHISPER_SERVER: OnceLock<Mutex<ServerState>> = OnceLock::new();

fn server_state() -> &'static Mutex<ServerState> {
    WHISPER_SERVER.get_or_init(|| {
        Mutex::new(ServerState {
            active: None,
            gpu_disabled: false,
        })
    })
}

pub fn stop_whisper_server() {
    let mut guard = server_state().lock();
    if let Some(mut srv) = guard.active.take() {
        let _ = srv.child.kill();
        let _ = srv.child.wait();
    }
}

fn free_port() -> Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?.port())
}

fn spawn_whisper_server(
    server_path: &Path,
    model_path: &Path,
    gpu_plugin: Option<&Path>,
) -> Result<(Child, u16)> {
    let port = free_port()?;
    let threads = std::thread::available_parallelism()
        .map(|n| n.get().min(4))
        .unwrap_or(4);
    let dir = server_path.parent().unwrap_or_else(|| Path::new("."));

    let mut cmd = Command::new(server_path);
    cmd.current_dir(dir)
        .args([
            "-m",
            &model_path.to_string_lossy(),
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "-t",
            &threads.to_string(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    hide_console_window(&mut cmd);

    if let Some(plugin) = gpu_plugin {
        cmd.env("GGML_BACKEND_PATH", plugin);
    } else {
        cmd.env_remove("GGML_BACKEND_PATH");
    }

    let child = cmd.spawn().context("Failed to spawn whisper-server")?;
    Ok((child, port))
}

async fn ensure_whisper_server(
    paths: &AppPaths,
    server_path: &Path,
    model_path: &Path,
) -> Result<u16> {
    let server_dir = server_path.parent().unwrap_or_else(|| Path::new("."));
    let gpu_disabled = server_state().lock().gpu_disabled;
    let gpu_plugin = borrowed_gpu_plugin(paths, server_dir, gpu_disabled);
    let key = format!(
        "{}|{}|{}",
        server_path.display(),
        model_path.display(),
        gpu_plugin.as_ref().map(|p| p.display().to_string()).unwrap_or_default()
    );

    {
        let mut guard = server_state().lock();
        if let Some(srv) = guard.active.as_mut() {
            if srv.key == key && srv.child.try_wait().ok().flatten().is_none() {
                srv.last_used = Instant::now();
                return Ok(srv.port);
            }
        }
        if let Some(mut old) = guard.active.take() {
            let _ = old.child.kill();
            let _ = old.child.wait();
        }
    }

    let (mut child, port) =
        spawn_whisper_server(server_path, model_path, gpu_plugin.as_deref())?;

    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(1000))
        .build()?;
    let url = format!("http://127.0.0.1:{port}/");
    let deadline = Instant::now() + Duration::from_secs(45);
    let mut ready = false;

    while Instant::now() < deadline {
        if let Some(status) = child.try_wait()? {
            bail!("whisper-server exited early with status {status}");
        }
        if let Ok(resp) = client.get(&url).send().await {
            if resp.status().is_success() || resp.status().as_u16() == 404 {
                ready = true;
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    if !ready {
        let _ = child.kill();
        bail!("whisper-server did not become ready in time");
    }

    let mut guard = server_state().lock();
    guard.active = Some(RunningWhisperServer {
        child,
        port,
        key,
        last_used: Instant::now(),
    });
    Ok(port)
}

async fn transcribe_with_whisper_server(
    paths: &AppPaths,
    server_path: &Path,
    model_path: &Path,
    wav_bytes: &[u8],
    language: &str,
    audio_ctx: usize,
) -> Result<String> {
    let port = match ensure_whisper_server(paths, server_path, model_path).await {
        Ok(p) => p,
        Err(_) => {
            server_state().lock().gpu_disabled = true;
            ensure_whisper_server(paths, server_path, model_path).await?
        }
    };

    let ctx_str = audio_ctx.to_string();
    let mut fields = vec![
        ("response_format", "json"),
        ("temperature", "0"),
        ("translate", "false"),
        ("language", if language.is_empty() { "auto" } else { language }),
    ];
    if audio_ctx > 0 {
        fields.push(("audio_ctx", &ctx_str));
    }

    let (content_type, body) =
        build_multipart_form(&fields, "file", "audio.wav", "audio/wav", wav_bytes);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()?;
    let resp = client
        .post(format!("http://127.0.0.1:{port}/inference"))
        .header("Content-Type", content_type)
        .body(body)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        bail!("whisper-server HTTP {status}: {text}");
    }

    let data: Value = resp.json().await?;
    if let Some(err) = data.get("error").and_then(Value::as_str) {
        bail!("whisper-server error: {err}");
    }
    let raw = data.get("text").and_then(Value::as_str).unwrap_or("");
    Ok(raw.split_whitespace().collect::<Vec<_>>().join(" "))
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// WHISPER-CLI ONE-SHOT FALLBACK
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub fn parse_whisper_output(output: &str) -> String {
    let mut parts = Vec::new();
    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Strip [00:00.000 --> 00:03.500] timestamp prefix if present
        if trimmed.starts_with('[') {
            if let Some(end_bracket) = trimmed.find(']') {
                let after = trimmed[end_bracket + 1..].trim();
                if !after.is_empty() {
                    parts.push(after.to_string());
                }
                continue;
            }
        }
        let lower = trimmed.to_ascii_lowercase();
        if lower.starts_with("whisper_model_load")
            || lower.starts_with("system_info")
            || lower.starts_with("ggml_")
            || lower.starts_with("main:")
            || lower.starts_with("loading")
            || lower.starts_with("print_")
            || lower.starts_with("init ")
            || lower.starts_with("compute ")
            || lower.starts_with("ram:")
            || lower.starts_with("processing")
        {
            continue;
        }
        if trimmed.len() > 1 {
            parts.push(trimmed.to_string());
        }
    }
    parts.join(" ").split_whitespace().collect::<Vec<_>>().join(" ")
}

fn run_whisper_cli(
    binary_path: &Path,
    model_path: &Path,
    wav_bytes: &[u8],
    language: &str,
    audio_ctx: usize,
) -> Result<String> {
    let tmp_dir = std::env::temp_dir().join("agente-dictate-stt");
    fs::create_dir_all(&tmp_dir)?;
    let wav_path = tmp_dir.join(format!("clip-{}.wav", uuid::Uuid::new_v4().simple()));
    fs::write(&wav_path, wav_bytes)?;

    let threads = std::thread::available_parallelism()
        .map(|n| n.get().min(4))
        .unwrap_or(4);

    let mut args = vec![
        "-m".to_string(),
        model_path.to_string_lossy().to_string(),
        "-f".to_string(),
        wav_path.to_string_lossy().to_string(),
        "-nt".to_string(),
        "-l".to_string(),
        if language.is_empty() { "auto" } else { language }.to_string(),
        "-t".to_string(),
        threads.to_string(),
        "-np".to_string(),
    ];
    if audio_ctx > 0 {
        args.push("-ac".to_string());
        args.push(audio_ctx.to_string());
    }

    let mut cmd = Command::new(binary_path);
    cmd.args(&args).stdin(Stdio::null());
    hide_console_window(&mut cmd);
    let output = cmd.output();

    let _ = fs::remove_file(&wav_path);
    let output = output.context("Failed to execute whisper-cli")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("whisper-cli exited with {}: {}", output.status, stderr.trim());
    }
    Ok(parse_whisper_output(&String::from_utf8_lossy(&output.stdout)))
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// COMMERCIAL CLOUD STT ENGINE (All 13 Providers)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async fn transcribe_commercial(
    cfg: &DictateConfig,
    provider_id: &str,
    wav_bytes: &[u8],
) -> Result<String> {
    let spec = find_commercial_provider(provider_id)
        .with_context(|| format!("Unknown commercial STT provider: {provider_id}"))?;
    let (api_key, base_url, model) = cfg
        .resolve_commercial_auth(provider_id)
        .context("Could not resolve provider settings")?;

    let is_localhost = base_url.contains("127.0.0.1") || base_url.contains("localhost");
    if api_key.is_empty() && !is_localhost {
        bail!(
            "No API key configured for {}. Press [k] to enter your API key or set ${}.",
            spec.name,
            spec.env_vars.first().copied().unwrap_or("API_KEY")
        );
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(90))
        .build()?;
    let lang = cfg.language.trim();

    match spec.protocol {
        CommercialSttProtocol::HuggingFaceInference => {
            let url = format!("{base_url}/{model}");
            for attempt in 0..4 {
                let resp = client
                    .post(&url)
                    .header("Authorization", format!("Bearer {api_key}"))
                    .header("Content-Type", "audio/wav")
                    .header("x-wait-for-model", "true")
                    .body(wav_bytes.to_vec())
                    .send()
                    .await?;
                if resp.status().as_u16() == 503 && attempt < 3 {
                    tokio::time::sleep(Duration::from_secs(3)).await;
                    continue;
                }
                if !resp.status().is_success() {
                    let status = resp.status();
                    let err_body = resp.text().await.unwrap_or_default();
                    bail!("Hugging Face Inference returned HTTP {status}: {err_body}");
                }
                let data: Value = resp.json().await?;
                let text = data
                    .get("text")
                    .and_then(Value::as_str)
                    .or_else(|| {
                        data.as_array()
                            .and_then(|arr| arr.first())
                            .and_then(|v| v.get("text"))
                            .and_then(Value::as_str)
                    })
                    .context("Hugging Face Inference response missing 'text'")?;
                return Ok(text.trim().to_string());
            }
            bail!("Hugging Face model is still warming up — try again in a few seconds")
        }

        CommercialSttProtocol::OpenAiMultipart => {
            let mut fields = vec![("model", model.as_str())];
            if !lang.is_empty() && lang != "auto" {
                fields.push(("language", lang));
            }
            let (content_type, body) =
                build_multipart_form(&fields, "file", "recording.wav", "audio/wav", wav_bytes);
            let url = format!("{base_url}/audio/transcriptions");
            let mut req = client.post(&url).header("Content-Type", content_type).body(body);
            if !api_key.is_empty() {
                req = req.header("Authorization", format!("Bearer {api_key}"));
            }
            let resp = req.send().await?;
            if !resp.status().is_success() {
                let status = resp.status();
                let err_body = resp.text().await.unwrap_or_default();
                bail!("{} returned HTTP {status}: {}", spec.name, err_body);
            }
            let data: Value = resp.json().await?;
            let text = data
                .get("text")
                .and_then(Value::as_str)
                .context("Response missing 'text' field")?;
            Ok(text.trim().to_string())
        }

        CommercialSttProtocol::Deepgram => {
            let lang_query = if !lang.is_empty() && lang != "auto" {
                format!("language={lang}")
            } else {
                "detect_language=true".to_string()
            };
            let url = format!(
                "{base_url}/listen?model={model}&smart_format=true&punctuate=true&{lang_query}"
            );
            let resp = client
                .post(&url)
                .header("Authorization", format!("Token {api_key}"))
                .header("Content-Type", "audio/wav")
                .body(wav_bytes.to_vec())
                .send()
                .await?;
            if !resp.status().is_success() {
                let status = resp.status();
                let err_body = resp.text().await.unwrap_or_default();
                bail!("Deepgram returned HTTP {status}: {err_body}");
            }
            let data: Value = resp.json().await?;
            let text = data
                .pointer("/results/channels/0/alternatives/0/transcript")
                .and_then(Value::as_str)
                .context("Deepgram response missing transcript")?;
            Ok(text.trim().to_string())
        }

        CommercialSttProtocol::ElevenLabsScribe => {
            let mut fields = vec![("model_id", model.as_str())];
            if !lang.is_empty() && lang != "auto" {
                fields.push(("language_code", lang));
            }
            let (content_type, body) =
                build_multipart_form(&fields, "file", "recording.wav", "audio/wav", wav_bytes);
            let url = format!("{base_url}/speech-to-text");
            let resp = client
                .post(&url)
                .header("xi-api-key", &api_key)
                .header("Content-Type", content_type)
                .body(body)
                .send()
                .await?;
            if !resp.status().is_success() {
                let status = resp.status();
                let err_body = resp.text().await.unwrap_or_default();
                bail!("ElevenLabs Scribe returned HTTP {status}: {err_body}");
            }
            let data: Value = resp.json().await?;
            let text = data
                .get("text")
                .and_then(Value::as_str)
                .context("ElevenLabs response missing 'text'")?;
            Ok(text.trim().to_string())
        }

        CommercialSttProtocol::CartesiaInk => {
            let mut fields = vec![("model", model.as_str())];
            if !lang.is_empty() && lang != "auto" {
                fields.push(("language", lang));
            }
            let (content_type, body) =
                build_multipart_form(&fields, "file", "recording.wav", "audio/wav", wav_bytes);
            let url = format!("{base_url}/stt");
            let resp = client
                .post(&url)
                .header("X-API-Key", &api_key)
                .header("Cartesia-Version", "2026-08-14")
                .header("Content-Type", content_type)
                .body(body)
                .send()
                .await?;
            if !resp.status().is_success() {
                let status = resp.status();
                let err_body = resp.text().await.unwrap_or_default();
                bail!("Cartesia Ink returned HTTP {status}: {err_body}");
            }
            let data: Value = resp.json().await?;
            let text = data
                .get("text")
                .and_then(Value::as_str)
                .context("Cartesia response missing 'text'")?;
            Ok(text.trim().to_string())
        }

        CommercialSttProtocol::Soniox => {
            // Step 1: Upload audio file via POST /files
            let (content_type, body) =
                build_multipart_form(&[], "file", "recording.wav", "audio/wav", wav_bytes);
            let up_resp = client
                .post(format!("{base_url}/files"))
                .header("Authorization", format!("Bearer {api_key}"))
                .header("Content-Type", content_type)
                .body(body)
                .send()
                .await?;
            if !up_resp.status().is_success() {
                let status = up_resp.status();
                let err_body = up_resp.text().await.unwrap_or_default();
                bail!("Soniox file upload returned HTTP {status}: {err_body}");
            }
            let up_json: Value = up_resp.json().await?;
            let file_id = up_json
                .get("id")
                .and_then(Value::as_str)
                .context("Soniox upload response missing 'id'")?;

            // Step 2: Create transcription job with stt-async-v6
            let tx_resp = client
                .post(format!("{base_url}/transcriptions"))
                .header("Authorization", format!("Bearer {api_key}"))
                .json(&json!({
                    "model": "stt-async-v6",
                    "file_id": file_id,
                }))
                .send()
                .await?;
            if !tx_resp.status().is_success() {
                let status = tx_resp.status();
                let err_body = tx_resp.text().await.unwrap_or_default();
                bail!("Soniox transcription start returned HTTP {status}: {err_body}");
            }
            let tx_json: Value = tx_resp.json().await?;
            let tx_id = tx_json
                .get("id")
                .and_then(Value::as_str)
                .context("Soniox missing transcription 'id'")?;

            // Step 3: Poll completion and fetch transcript
            let status_url = format!("{base_url}/transcriptions/{tx_id}");
            for _ in 0..120 {
                tokio::time::sleep(Duration::from_millis(400)).await;
                let poll = client
                    .get(&status_url)
                    .header("Authorization", format!("Bearer {api_key}"))
                    .send()
                    .await?;
                let poll_json: Value = poll.json().await?;
                match poll_json.get("status").and_then(Value::as_str) {
                    Some("completed") => {
                        let tr = client
                            .get(format!("{base_url}/transcriptions/{tx_id}/transcript"))
                            .header("Authorization", format!("Bearer {api_key}"))
                            .send()
                            .await?;
                        let tr_json: Value = tr.json().await?;
                        let text = tr_json
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .trim()
                            .to_string();
                        return Ok(text);
                    }
                    Some("error") | Some("failed") => {
                        bail!("Soniox transcription job failed");
                    }
                    _ => {}
                }
            }
            bail!("Soniox transcription timed out")
        }

        CommercialSttProtocol::GeminiAudio => {
            let b64 = base64::engine::general_purpose::STANDARD.encode(wav_bytes);
            let prompt = if !lang.is_empty() && lang != "auto" {
                format!("Transcribe the spoken audio accurately in language '{lang}'. Return ONLY the verbatim transcript text with natural punctuation, no commentary.")
            } else {
                "Transcribe the spoken audio accurately. Return ONLY the verbatim transcript text with natural punctuation, no commentary.".to_string()
            };
            let payload = json!({
                "contents": [{
                    "parts": [
                        { "text": prompt },
                        {
                            "inlineData": {
                                "mimeType": "audio/wav",
                                "data": b64
                            }
                        }
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.0
                }
            });
            let url = format!("{base_url}/models/{model}:generateContent?key={api_key}");
            let resp = client.post(&url).json(&payload).send().await?;
            if !resp.status().is_success() {
                let status = resp.status();
                let err_body = resp.text().await.unwrap_or_default();
                bail!("Gemini Audio returned HTTP {status}: {err_body}");
            }
            let data: Value = resp.json().await?;
            let text = data
                .pointer("/candidates/0/content/parts/0/text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();
            Ok(text)
        }

        CommercialSttProtocol::AssemblyAi => {
            // Step 1: Upload raw WAV bytes
            let up_resp = client
                .post(format!("{base_url}/upload"))
                .header("Authorization", &api_key)
                .header("Content-Type", "application/octet-stream")
                .body(wav_bytes.to_vec())
                .send()
                .await?;
            if !up_resp.status().is_success() {
                bail!("AssemblyAI upload failed: HTTP {}", up_resp.status());
            }
            let up_json: Value = up_resp.json().await?;
            let upload_url = up_json
                .get("upload_url")
                .and_then(Value::as_str)
                .context("AssemblyAI missing upload_url")?;

            // Step 2: Create transcript job
            let mut body = json!({
                "audio_url": upload_url,
                "speech_model": model,
            });
            if !lang.is_empty() && lang != "auto" {
                body["language_code"] = json!(lang);
            } else {
                body["language_detection"] = json!(true);
            }
            let tx_resp = client
                .post(format!("{base_url}/transcript"))
                .header("Authorization", &api_key)
                .json(&body)
                .send()
                .await?;
            if !tx_resp.status().is_success() {
                bail!("AssemblyAI transcript request failed: HTTP {}", tx_resp.status());
            }
            let tx_json: Value = tx_resp.json().await?;
            let id = tx_json
                .get("id")
                .and_then(Value::as_str)
                .context("AssemblyAI missing job id")?;

            // Step 3: Poll completion
            let poll_url = format!("{base_url}/transcript/{id}");
            for _ in 0..120 {
                tokio::time::sleep(Duration::from_millis(450)).await;
                let poll = client
                    .get(&poll_url)
                    .header("Authorization", &api_key)
                    .send()
                    .await?;
                let status_json: Value = poll.json().await?;
                match status_json.get("status").and_then(Value::as_str) {
                    Some("completed") => {
                        return Ok(status_json
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .trim()
                            .to_string());
                    }
                    Some("error") => {
                        let err = status_json
                            .get("error")
                            .and_then(Value::as_str)
                            .unwrap_or("AssemblyAI transcription failed");
                        bail!("{err}");
                    }
                    _ => {}
                }
            }
            bail!("AssemblyAI transcription timed out")
        }

        CommercialSttProtocol::Gladia => {
            let (content_type, body) =
                build_multipart_form(&[], "audio", "recording.wav", "audio/wav", wav_bytes);
            let up_resp = client
                .post(format!("{base_url}/upload"))
                .header("x-gladia-key", &api_key)
                .header("Content-Type", content_type)
                .body(body)
                .send()
                .await?;
            if !up_resp.status().is_success() {
                bail!("Gladia upload failed: HTTP {}", up_resp.status());
            }
            let up_json: Value = up_resp.json().await?;
            let audio_url = up_json
                .get("audio_url")
                .and_then(Value::as_str)
                .context("Gladia missing audio_url")?;

            let tx_resp = client
                .post(format!("{base_url}/pre-recorded"))
                .header("x-gladia-key", &api_key)
                .json(&json!({ "audio_url": audio_url }))
                .send()
                .await?;
            if !tx_resp.status().is_success() {
                bail!("Gladia transcription start failed: HTTP {}", tx_resp.status());
            }
            let tx_json: Value = tx_resp.json().await?;
            let result_url = tx_json
                .get("result_url")
                .and_then(Value::as_str)
                .context("Gladia missing result_url")?;

            for _ in 0..120 {
                tokio::time::sleep(Duration::from_millis(450)).await;
                let poll = client
                    .get(result_url)
                    .header("x-gladia-key", &api_key)
                    .send()
                    .await?;
                let data: Value = poll.json().await?;
                match data.get("status").and_then(Value::as_str) {
                    Some("done") => {
                        let text = data
                            .pointer("/result/transcription/full_transcript")
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        return Ok(text.trim().to_string());
                    }
                    Some("error") => bail!("Gladia transcription error"),
                    _ => {}
                }
            }
            bail!("Gladia transcription timed out")
        }

        CommercialSttProtocol::Speechmatics | CommercialSttProtocol::RevAi => {
            // Both Speechmatics and Rev AI also accept multipart jobs; provide OpenAI-compatible
            // fallback or job submission.
            let config_json = json!({
                "type": "transcription",
                "transcription_config": {
                    "operating_point": model,
                    "language": if lang.is_empty() || lang == "auto" { "en" } else { lang }
                }
            })
            .to_string();
            let fields = [("config", config_json.as_str())];
            let (content_type, body) =
                build_multipart_form(&fields, "data_file", "recording.wav", "audio/wav", wav_bytes);
            let resp = client
                .post(format!("{base_url}/jobs"))
                .header("Authorization", format!("Bearer {api_key}"))
                .header("Content-Type", content_type)
                .body(body)
                .send()
                .await?;
            if !resp.status().is_success() {
                let status = resp.status();
                let err_body = resp.text().await.unwrap_or_default();
                bail!("{} returned HTTP {status}: {err_body}", spec.name);
            }
            let job: Value = resp.json().await?;
            let id = job.get("id").and_then(Value::as_str).context("Missing job id")?;
            for _ in 0..120 {
                tokio::time::sleep(Duration::from_millis(500)).await;
                let tr = client
                    .get(format!("{base_url}/jobs/{id}/transcript?format=txt"))
                    .header("Authorization", format!("Bearer {api_key}"))
                    .send()
                    .await?;
                if tr.status().is_success() {
                    return Ok(tr.text().await?.trim().to_string());
                }
            }
            bail!("{} transcription timed out", spec.name)
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// UNIFIED TRANSCRIPTION ENTRYPOINT
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub struct TranscriptionOutput {
    pub text: String,
    pub backend: String,
    pub model: String,
    pub elapsed_ms: u64,
}

pub async fn transcribe_wav(
    paths: &AppPaths,
    cfg: &DictateConfig,
    wav_bytes: &[u8],
) -> Result<TranscriptionOutput> {
    let started = Instant::now();

    if cfg.backend != "local" {
        let raw = transcribe_commercial(cfg, &cfg.backend, wav_bytes).await?;
        let cleaned = spoken_transcript(&raw).unwrap_or_default();
        return Ok(TranscriptionOutput {
            text: cleaned,
            backend: cfg.backend.clone(),
            model: cfg.commercial_model.clone(),
            elapsed_ms: started.elapsed().as_millis() as u64,
        });
    }

    let model_id = &cfg.local_model;
    let spec = find_local_whisper_spec(model_id)
        .with_context(|| format!("Unknown local model '{model_id}'"))?;
    let model_path = check_local_whisper_model(paths, model_id).with_context(|| {
        format!(
            "Local model '{model_id}' is not installed yet. Click 'Install Local Engine' or press [i] to download it."
        )
    })?;

    if spec.engine == LocalEngineKind::AudioCpp {
        let audio_bin = resolve_audiocpp_binary(paths).context(
            "audio.cpp runtime (for non-Whisper GGUF models) is not installed yet. Click 'Install Local Engine' or press [i].",
        )?;
        let raw = transcribe_with_audiocpp_server(&audio_bin, &model_path, wav_bytes, &cfg.language)
            .await?;
        let cleaned = spoken_transcript(&raw).unwrap_or_default();
        return Ok(TranscriptionOutput {
            text: cleaned,
            backend: "local".to_string(),
            model: model_id.clone(),
            elapsed_ms: started.elapsed().as_millis() as u64,
        });
    }

    let binary_path = resolve_whisper_binary(paths, cfg).context(
        "whisper.cpp runtime is not installed yet. Press [i] or [m] to install it automatically.",
    )?;

    let audio_ctx = live_whisper_context(wav_bytes);
    let raw = if let Some(server_path) = find_whisper_server(&binary_path) {
        match transcribe_with_whisper_server(
            paths,
            &server_path,
            &model_path,
            wav_bytes,
            &cfg.language,
            audio_ctx,
        )
        .await
        {
            Ok(text) => text,
            Err(_) => {
                run_whisper_cli(&binary_path, &model_path, wav_bytes, &cfg.language, audio_ctx)?
            }
        }
    } else {
        run_whisper_cli(&binary_path, &model_path, wav_bytes, &cfg.language, audio_ctx)?
    };

    let cleaned = spoken_transcript(&raw).unwrap_or_default();
    Ok(TranscriptionOutput {
        text: cleaned,
        backend: "local".to_string(),
        model: model_id.clone(),
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ONE-STEP SETUP: DOWNLOAD WHISPER.CPP RUNTIME + HUGGING FACE MODEL
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

const RELEASES_URL: &str = "https://api.github.com/repos/ggml-org/whisper.cpp/releases?per_page=10";
const AUDIOCPP_RELEASES_URL: &str =
    "https://api.github.com/repos/0xShug0/audio.cpp/releases?per_page=10";

async fn transcribe_with_audiocpp_server(
    server_path: &Path,
    model_path: &Path,
    wav_bytes: &[u8],
    language: &str,
) -> Result<String> {
    let port = free_port()?;
    let threads = std::thread::available_parallelism()
        .map(|n| n.get().min(4))
        .unwrap_or(4);
    let dir = server_path.parent().unwrap_or_else(|| Path::new("."));

    let mut cmd = Command::new(server_path);
    cmd.current_dir(dir)
        .args([
            "--model",
            &model_path.to_string_lossy(),
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--threads",
            &threads.to_string(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    hide_console_window(&mut cmd);
    let mut child = cmd
        .spawn()
        .context("Failed to start audiocpp_server")?;

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()?;
    let deadline = Instant::now() + Duration::from_secs(45);
    let mut ready = false;
    while Instant::now() < deadline {
        if let Some(st) = child.try_wait()? {
            bail!("audiocpp_server exited early with status {st}");
        }
        if let Ok(r) = client.get(format!("http://127.0.0.1:{port}/health")).send().await {
            if r.status().is_success() || r.status().as_u16() == 404 {
                ready = true;
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    if !ready {
        let _ = child.kill();
        bail!("audiocpp_server did not start in time");
    }

    let mut fields = vec![("model", "local")];
    if !language.is_empty() && language != "auto" {
        fields.push(("language", language));
    }
    let (content_type, body) =
        build_multipart_form(&fields, "file", "recording.wav", "audio/wav", wav_bytes);
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/audio/transcriptions"))
        .header("Content-Type", content_type)
        .body(body)
        .send()
        .await;

    let _ = child.kill();
    let _ = child.wait();

    let resp = resp?;
    if !resp.status().is_success() {
        let status = resp.status();
        let err_text = resp.text().await.unwrap_or_default();
        bail!("audiocpp_server returned HTTP {status}: {err_text}");
    }
    let data: Value = resp.json().await?;
    let text = data.get("text").and_then(Value::as_str).unwrap_or("");
    Ok(text.trim().to_string())
}

fn select_audiocpp_asset(assets: &[Value]) -> Option<(String, String)> {
    let want_suffix = if cfg!(windows) {
        "windows-x64-cpu-portable.zip"
    } else if cfg!(target_os = "linux") {
        "ubuntu-x64-cpu-portable.tar.gz"
    } else {
        return None;
    };
    for asset in assets {
        let name = asset.get("name").and_then(Value::as_str).unwrap_or("");
        let url = asset
            .get("browser_download_url")
            .and_then(Value::as_str)
            .unwrap_or("");
        if name.ends_with(want_suffix) && !url.is_empty() {
            return Some((name.to_string(), url.to_string()));
        }
    }
    None
}

fn select_whisper_asset(assets: &[Value]) -> Option<(String, String)> {
    let is_arm = cfg!(target_arch = "aarch64");
    let has_cuda = has_nvidia_gpu();

    let candidates: &[&str] = if cfg!(windows) {
        if is_arm {
            &["whisper-bin-win-cpu-arm64.zip"]
        } else if has_cuda {
            &["whisper-cublas-12", "whisper-bin-x64.zip"]
        } else {
            &["whisper-bin-x64.zip"]
        }
    } else if cfg!(target_os = "linux") {
        if is_arm {
            &["whisper-bin-ubuntu-arm64.tar.gz"]
        } else {
            &["whisper-bin-ubuntu-x64.tar.gz"]
        }
    } else {
        &[]
    };

    for &pattern in candidates {
        for asset in assets {
            let name = asset.get("name").and_then(Value::as_str).unwrap_or("");
            let url = asset
                .get("browser_download_url")
                .and_then(Value::as_str)
                .unwrap_or("");
            if (name == pattern || (pattern.starts_with("whisper-cublas") && name.starts_with(pattern) && name.ends_with("-bin-x64.zip")))
                && !url.is_empty()
            {
                return Some((name.to_string(), url.to_string()));
            }
        }
    }
    None
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DownloadProgress {
    received: u64,
    total: Option<u64>,
    bytes_per_second: u64,
}

impl DownloadProgress {
    pub fn fraction(self) -> Option<f32> {
        self.total
            .map(|total| (self.received as f32 / total as f32).min(1.0))
    }

    pub fn label(self) -> String {
        let received = self.received as f64 / 1_000_000.0;
        match self.total {
            Some(total) => format!("{received:.1} of {:.1} MB", total as f64 / 1_000_000.0),
            None => format!("{received:.1} MB received"),
        }
    }

    pub fn transfer_label(self) -> String {
        let mut label = self.label();
        if self.bytes_per_second > 0 {
            label.push_str(&format!(
                " · {:.1} MB/s",
                self.bytes_per_second as f64 / 1_000_000.0
            ));
            if let Some(total) = self.total {
                let seconds = total
                    .saturating_sub(self.received)
                    .div_ceil(self.bytes_per_second);
                if seconds > 0 {
                    let remaining = if seconds < 60 {
                        format!("{seconds}s left")
                    } else if seconds < 3600 {
                        format!("{}m left", seconds.div_ceil(60))
                    } else {
                        format!("{:.1}h left", seconds as f64 / 3600.0)
                    };
                    label.push_str(&format!(" · {remaining}"));
                }
            }
        }
        label
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstallStatus {
    Checking,
    DownloadingEngine(DownloadProgress),
    ExtractingEngine,
    DownloadingModel(DownloadProgress),
    Finalizing,
    Ready,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DownloadCommand {
    Run,
    Pause,
    Cancel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DownloadOutcome {
    Complete,
    Paused,
    Canceled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstallTarget {
    Engine(LocalEngineKind),
    Model(String),
}

impl InstallTarget {
    pub fn model_id(&self) -> Option<&str> {
        match self {
            Self::Model(id) => Some(id),
            Self::Engine(_) => None,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Engine(engine) => engine.label(),
            Self::Model(id) => find_local_whisper_spec(id).map_or(id.as_str(), |spec| spec.name()),
        }
    }
}

impl InstallStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Checking => "Checking requirements",
            Self::DownloadingEngine(_) => "Downloading the speech engine",
            Self::ExtractingEngine => "Installing the speech engine",
            Self::DownloadingModel(_) => "Downloading the speech model",
            Self::Finalizing => "Finishing setup",
            Self::Ready => "Ready",
        }
    }

    pub fn transfer(self) -> Option<DownloadProgress> {
        match self {
            Self::DownloadingEngine(progress) | Self::DownloadingModel(progress) => Some(progress),
            _ => None,
        }
    }
}

fn download_sidecar(dest: &Path, suffix: &str) -> PathBuf {
    let mut path = dest.as_os_str().to_owned();
    path.push(suffix);
    path.into()
}

fn lock_download(dest: &Path) -> Result<fs::File> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(download_sidecar(dest, ".download-lock"))?;
    file.try_lock()
        .context("Another process is downloading these files")?;
    Ok(file)
}

pub fn discard_download(paths: &AppPaths, target: &InstallTarget) -> Result<()> {
    let Some(model_id) = target.model_id() else {
        return Ok(());
    };
    let spec = find_local_whisper_spec(model_id).context("Unknown speech model")?;
    let dest = paths.whisper_models_root.join(spec.file);
    let _lock = lock_download(&dest)?;
    for suffix in [".part", ".etag"] {
        match fs::remove_file(download_sidecar(&dest, suffix)) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        }
    }
    Ok(())
}

fn resumed_total(value: &str, offset: u64) -> Option<u64> {
    let (range, total) = value.strip_prefix("bytes ")?.split_once('/')?;
    let (start, end) = range.split_once('-')?;
    let total = total.parse::<u64>().ok()?;
    if start.parse::<u64>().ok()? == offset
        && end.parse::<u64>().ok()?.checked_add(1)? == total
        && offset < total
    {
        Some(total)
    } else {
        None
    }
}

/// Removes unused partial files for engine archives, including on cancellation.
struct TemporaryDownload(PathBuf);

impl Drop for TemporaryDownload {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

struct EngineStagingDirectory(PathBuf);

impl Drop for EngineStagingDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

async fn download_file<F>(
    url: &str,
    dest: &Path,
    bearer_token: Option<&str>,
    resumable: bool,
    mut on_progress: F,
) -> Result<()>
where
    F: FnMut(DownloadProgress),
{
    let _lock = resumable.then(|| lock_download(dest)).transpose()?;
    let temporary = (!resumable)
        .then(|| TemporaryDownload(dest.with_extension(format!("{}.part", uuid::Uuid::new_v4()))));
    let partial = temporary
        .as_ref()
        .map_or_else(|| download_sidecar(dest, ".part"), |file| file.0.clone());
    let etag_path = download_sidecar(dest, ".etag");
    let saved_etag = resumable
        .then(|| fs::read_to_string(&etag_path).ok())
        .flatten()
        .filter(|tag| tag.starts_with('"') && tag.ends_with('"'));
    let mut offset = if saved_etag.is_some() {
        fs::metadata(&partial).map_or(0, |info| info.len())
    } else {
        0
    };
    let client = reqwest::Client::builder()
        .user_agent("AgenteDictate/0.1")
        .connect_timeout(Duration::from_secs(15))
        .build()?;
    let make_request = |offset| {
        let mut request = client.get(url);
        if let Some(tok) = bearer_token.filter(|t| !t.trim().is_empty()) {
            request = request.bearer_auth(tok.trim());
        }
        if let Some(tag) = saved_etag.as_ref().filter(|_| offset > 0) {
            request = request
                .header(reqwest::header::RANGE, format!("bytes={offset}-"))
                .header(reqwest::header::IF_RANGE, tag);
        }
        request
    };
    let send = |request: reqwest::RequestBuilder| async {
        tokio::time::timeout(Duration::from_secs(60), request.send())
            .await
            .context("Timed out waiting for the download to start")?
            .map_err(anyhow::Error::from)
    };
    let mut response = send(make_request(offset)).await?;
    let mut range_total = None;
    if offset > 0 {
        range_total = response
            .headers()
            .get(reqwest::header::CONTENT_RANGE)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| resumed_total(value, offset))
            .filter(|_| response.status() == reqwest::StatusCode::PARTIAL_CONTENT)
            .filter(|_| {
                response
                    .headers()
                    .get(reqwest::header::ETAG)
                    .is_none_or(|value| value.to_str().ok() == saved_etag.as_deref())
            });
        if range_total.is_none() {
            offset = 0;
            if matches!(
                response.status(),
                reqwest::StatusCode::PARTIAL_CONTENT | reqwest::StatusCode::RANGE_NOT_SATISFIABLE
            ) {
                response = send(make_request(0)).await?;
            }
        }
    }
    let status = response.status();
    if status.as_u16() == 401 || status.as_u16() == 403 {
        bail!("Download denied (HTTP {status}). For a gated Hugging Face model, check your Hugging Face token in Settings and verify access to the model.");
    }
    if status != reqwest::StatusCode::OK && range_total.is_none() {
        bail!("Download failed (HTTP {status}): {url}");
    }
    let total = range_total.or_else(|| response.content_length().filter(|&size| size > 0));
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    if resumable && offset == 0 {
        // Invalidate the old validator before touching its bytes.
        match fs::remove_file(&etag_path) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        }
    }
    let mut file = if offset > 0 {
        fs::OpenOptions::new().append(true).open(&partial)?
    } else {
        fs::File::create(&partial)?
    };
    if resumable && offset == 0 {
        if let Some(tag) = response
            .headers()
            .get(reqwest::header::ETAG)
            .and_then(|value| value.to_str().ok())
            .filter(|tag| tag.starts_with('"') && tag.ends_with('"'))
        {
            fs::write(&etag_path, tag)?;
        }
    }
    let mut received = offset;
    let started = Instant::now();
    let mut last_report = started;
    let report = |received, bytes_per_second| DownloadProgress {
        received,
        total,
        bytes_per_second,
    };
    on_progress(report(received, 0));
    let mut stream = response.bytes_stream();

    while let Some(chunk) = tokio::time::timeout(Duration::from_secs(60), stream.next())
        .await
        .context("The download stopped receiving data")?
    {
        let bytes = chunk?;
        file.write_all(&bytes)?;
        received += bytes.len() as u64;
        if last_report.elapsed() >= Duration::from_millis(125) {
            let rate = ((received - offset) as f64 / started.elapsed().as_secs_f64()) as u64;
            on_progress(report(received, rate));
            last_report = Instant::now();
        }
    }
    if received == 0 || total.is_some_and(|size| received != size) {
        bail!("The download ended before the full file was received.");
    }
    file.flush()?;
    drop(file);
    fs::rename(&partial, dest)?;
    on_progress(report(received, 0));
    if resumable {
        let _ = fs::remove_file(etag_path);
    }
    Ok(())
}

fn extract_archive(archive_path: &Path, dest_dir: &Path) -> Result<()> {
    fs::create_dir_all(dest_dir)?;
    let name = archive_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    if cfg!(windows) || name.ends_with(".zip") {
        let mut tar = Command::new("tar");
        tar.arg("-xf").arg(archive_path).arg("-C").arg(dest_dir);
        hide_console_window(&mut tar);
        let status = tar.status();
        if status.is_ok_and(|s| s.success()) {
            return Ok(());
        }
        // Fallback to PowerShell Expand-Archive on Windows if system tar failed
        let mut powershell = Command::new("powershell");
        powershell
            .env("ZOV_ENGINE_ARCHIVE", archive_path)
            .env("ZOV_ENGINE_DESTINATION", dest_dir)
            .args([
                "-NoProfile",
                "-Command",
                "Expand-Archive -LiteralPath $env:ZOV_ENGINE_ARCHIVE -DestinationPath $env:ZOV_ENGINE_DESTINATION -Force",
            ]);
        hide_console_window(&mut powershell);
        let ps = powershell
            .status()
            .context("Failed to extract the speech engine archive")?;
        if !ps.success() {
            bail!("Expand-Archive failed for {}", archive_path.display());
        }
    } else {
        let status = Command::new("tar")
            .args([
                "-xzf",
                &archive_path.to_string_lossy(),
                "-C",
                &dest_dir.to_string_lossy(),
            ])
            .status()
            .context("Failed to extract tar.gz archive")?;
        if !status.success() {
            bail!("tar extraction failed for {}", archive_path.display());
        }
    }
    Ok(())
}

/// Install an engine or model with the same progress and cancellation controls.
pub async fn install_controlled<F>(
    paths: &AppPaths,
    cfg: &DictateConfig,
    target: &InstallTarget,
    mut commands: tokio::sync::watch::Receiver<DownloadCommand>,
    on_status: F,
) -> Result<DownloadOutcome>
where
    F: FnMut(InstallStatus),
{
    let stop = async {
        loop {
            let command = *commands.borrow_and_update();
            if command != DownloadCommand::Run {
                return command;
            }
            if commands.changed().await.is_err() {
                return DownloadCommand::Pause;
            }
        }
    };
    let command = tokio::select! {
        biased;
        command = stop => command,
        result = install(paths, cfg, target, on_status) => {
            return result.map(|()| DownloadOutcome::Complete);
        }
    };
    if command == DownloadCommand::Pause {
        Ok(DownloadOutcome::Paused)
    } else {
        discard_download(paths, target)?;
        Ok(DownloadOutcome::Canceled)
    }
}

pub async fn setup_local_whisper<F>(
    paths: &AppPaths,
    cfg: &DictateConfig,
    model_id: &str,
    on_status: F,
) -> Result<()>
where
    F: FnMut(InstallStatus),
{
    install(
        paths,
        cfg,
        &InstallTarget::Model(model_id.to_owned()),
        on_status,
    )
    .await
}

async fn install<F>(
    paths: &AppPaths,
    cfg: &DictateConfig,
    target: &InstallTarget,
    mut on_status: F,
) -> Result<()>
where
    F: FnMut(InstallStatus),
{
    on_status(InstallStatus::Checking);
    match target {
        InstallTarget::Engine(engine) => {
            ensure_speech_engine(paths, cfg, *engine, &mut on_status).await?;
            on_status(InstallStatus::Finalizing);
        }
        InstallTarget::Model(model_id) => {
            let spec = find_local_whisper_spec(model_id)
                .with_context(|| format!("Unknown local model '{model_id}'"))?;
            ensure_speech_engine(paths, cfg, spec.engine, &mut on_status).await?;
            if check_local_whisper_model(paths, spec.id).is_none() {
                let destination = paths.whisper_models_root.join(spec.file);
                let token = cfg.resolve_hf_token();
                let bearer = (!token.is_empty()).then_some(token.as_str());
                download_file(spec.url, &destination, bearer, true, |progress| {
                    on_status(InstallStatus::DownloadingModel(progress));
                })
                .await?;
            }
            on_status(InstallStatus::Finalizing);
            if check_local_whisper_model(paths, spec.id).is_none() {
                bail!("The installed model could not be found. Check where it was downloaded, then retry.");
            }
        }
    }
    on_status(InstallStatus::Ready);
    Ok(())
}

async fn ensure_speech_engine(
    paths: &AppPaths,
    cfg: &DictateConfig,
    engine: LocalEngineKind,
    on_status: &mut impl FnMut(InstallStatus),
) -> Result<()> {
    if resolve_speech_engine(paths, cfg, engine).is_some() {
        return Ok(());
    }
    let _lock = lock_download(&paths.cache_root.join(engine.label()))?;
    if resolve_speech_engine(paths, cfg, engine).is_some() {
        return Ok(());
    }
    let releases_url = match engine {
        LocalEngineKind::WhisperCpp => RELEASES_URL,
        LocalEngineKind::AudioCpp => AUDIOCPP_RELEASES_URL,
    };
    let client = reqwest::Client::builder()
        .user_agent("AgenteDictate/0.1")
        .timeout(Duration::from_secs(30))
        .build()?;
    let releases: Vec<Value> = client
        .get(releases_url)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let (tag, asset_name, asset_url) = releases
        .iter()
        .find_map(|release| {
            let assets = release.get("assets")?.as_array()?;
            let (name, url) = match engine {
                LocalEngineKind::WhisperCpp => select_whisper_asset(assets)?,
                LocalEngineKind::AudioCpp => select_audiocpp_asset(assets)?,
            };
            let tag = release
                .get("tag_name")
                .and_then(Value::as_str)
                .unwrap_or("latest");
            Some((tag, name, url))
        })
        .with_context(|| {
            format!(
                "No prebuilt {} binary found for this OS or architecture",
                engine.label()
            )
        })?;

    let archive = TemporaryDownload(paths.cache_root.join(asset_name));
    download_file(&asset_url, &archive.0, None, false, |progress| {
        on_status(InstallStatus::DownloadingEngine(progress));
    })
    .await?;

    on_status(InstallStatus::ExtractingEngine);
    let destination = match engine {
        LocalEngineKind::WhisperCpp => paths.whisper_bin_root.join(tag),
        LocalEngineKind::AudioCpp => paths.audio_bin_root.join("cpu"),
    };
    let destination = if destination.exists() {
        download_sidecar(&destination, &format!("-{}", uuid::Uuid::new_v4()))
    } else {
        destination
    };
    let staging_path = paths
        .cache_root
        .join(format!("engine-unpack-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&staging_path)?;
    let staging = EngineStagingDirectory(staging_path);
    extract_archive(&archive.0, &staging.0)?;
    let binary_names = match engine {
        LocalEngineKind::WhisperCpp => WHISPER_CLI_NAMES,
        LocalEngineKind::AudioCpp => &[AUDIOCPP_SERVER_NAME],
    };
    let binary = find_binary_in_dir(&staging.0, binary_names).with_context(|| {
        format!(
            "The archive did not contain a {} executable.",
            engine.label()
        )
    })?;
    if binary.strip_prefix(&staging.0)?.components().count() > 3 {
        bail!(
            "The {} archive has an unsupported folder layout.",
            engine.label()
        );
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::rename(&staging.0, &destination)
        .context("Could not publish the installed speech engine")?;
    if resolve_speech_engine(paths, cfg, engine).is_none() {
        bail!(
            "Setup finished but the {} executable could not be found.",
            engine.label()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::TcpListener;

    async fn serve_download(
        response: &'static [u8],
    ) -> (String, tokio::sync::oneshot::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (headers_tx, headers_rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut stream = BufReader::new(stream);
            let mut line = String::new();
            let mut headers = String::new();
            loop {
                line.clear();
                if stream.read_line(&mut line).await.unwrap() == 0 || line == "\r\n" {
                    break;
                }
                headers.push_str(&line);
            }
            let _ = headers_tx.send(headers);
            stream.get_mut().write_all(response).await.unwrap();
            stream.get_mut().shutdown().await.unwrap();
        });
        (format!("http://{addr}/model"), headers_rx)
    }

    #[tokio::test]
    async fn reports_bytes_when_the_server_does_not_send_a_total() {
        let (url, _) = serve_download(
            b"HTTP/1.1 200 OK\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nmodel\r\n0\r\n\r\n",
        ).await;
        let dir = std::env::temp_dir().join(format!("zov-download-{}", uuid::Uuid::new_v4()));
        let target = dir.join("model.bin");
        let mut reports = Vec::new();
        let result = download_file(&url, &target, None, false, |progress| {
            reports.push(progress)
        })
        .await;
        let saved = fs::read(&target);
        let _ = fs::remove_file(&target);
        let _ = fs::remove_dir(&dir);

        result.unwrap();
        assert_eq!(saved.unwrap(), b"model");
        assert_eq!(reports.first().unwrap().received, 0);
        assert_eq!(reports.last().unwrap().received, 5);
        assert_eq!(reports.last().unwrap().fraction(), None);
    }

    #[tokio::test]
    async fn failed_download_keeps_the_existing_file_and_cleans_up_the_partial() {
        let (url, _) = serve_download(
            b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 100\r\n\r\nincomplete",
        )
        .await;
        let dir = std::env::temp_dir().join(format!("zov-download-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&dir).unwrap();
        let target = dir.join("model.bin");
        fs::write(&target, b"existing model").unwrap();

        let result = download_file(&url, &target, None, false, |_| {}).await;
        let saved = fs::read(&target).unwrap();
        fs::remove_file(&target).unwrap();
        let cleanup = fs::remove_dir(&dir);

        assert!(result.is_err());
        assert_eq!(saved, b"existing model");
        cleanup.unwrap();
    }

    #[tokio::test]
    async fn model_download_resumes_only_for_matching_ranges() {
        for (response, expected) in [
            (b"HTTP/1.1 206 Partial Content\r\nConnection: close\r\nContent-Length: 3\r\nContent-Range: bytes 2-4/5\r\nETag: \"v1\"\r\n\r\ndel".as_slice(), b"model".as_slice()),
            (b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 5\r\nETag: \"v2\"\r\n\r\nfresh".as_slice(), b"fresh".as_slice()),
        ] {
            let (url, headers) = serve_download(response).await;
            let dir = std::env::temp_dir().join(format!("zov-resume-{}", uuid::Uuid::new_v4()));
            fs::create_dir(&dir).unwrap();
            let target = dir.join("model.bin");
            fs::write(download_sidecar(&target, ".part"), b"mo").unwrap();
            fs::write(download_sidecar(&target, ".etag"), "\"v1\"").unwrap();
            let mut reports = Vec::new();
            let result = download_file(&url, &target, None, true, |progress| reports.push(progress)).await;
            let saved = fs::read(&target);
            let _ = fs::remove_file(&target);
            fs::remove_file(download_sidecar(&target, ".download-lock")).unwrap();
            let cleanup = fs::remove_dir(&dir);

            result.unwrap();
            assert_eq!(saved.unwrap(), expected);
            assert_eq!(reports.last().unwrap().fraction(), Some(1.0));
            let headers = headers.await.unwrap().to_ascii_lowercase();
            assert!(headers.contains("range: bytes=2-"));
            assert!(headers.contains("if-range: \"v1\""));
            cleanup.unwrap();
        }
    }

    #[test]
    fn range_response_must_cover_the_remaining_file_exactly() {
        assert_eq!(resumed_total("bytes 2-4/5", 2), Some(5));
        assert_eq!(resumed_total("bytes 1-4/5", 2), None);
        assert_eq!(resumed_total("bytes 2-3/5", 2), None);
        assert_eq!(resumed_total("bytes 2-4/*", 2), None);
    }

    #[test]
    fn cancel_discards_the_partial_without_deleting_the_installed_model() {
        let dir = std::env::temp_dir().join(format!("zov-cancel-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&dir).unwrap();
        let mut paths = AppPaths::resolve();
        paths.whisper_models_root = dir.clone();
        let spec = find_local_whisper_spec("small_q5").unwrap();
        let target = dir.join(spec.file);
        fs::write(&target, b"installed").unwrap();
        let partial = download_sidecar(&target, ".part");
        let validator = download_sidecar(&target, ".etag");
        fs::write(&partial, b"partial").unwrap();
        fs::write(&validator, "\"v1\"").unwrap();

        discard_download(&paths, &InstallTarget::Model(spec.id.into())).unwrap();

        assert!(!partial.exists());
        assert!(!validator.exists());
        assert_eq!(fs::read(&target).unwrap(), b"installed");
        fs::remove_file(&target).unwrap();
        fs::remove_file(download_sidecar(&target, ".download-lock")).unwrap();
        fs::remove_dir(&dir).unwrap();
    }

    #[tokio::test]
    async fn engine_only_setup_reuses_a_runtime_without_creating_a_model_directory() {
        let dir = std::env::temp_dir().join(format!("zov-engine-{}", uuid::Uuid::new_v4()));
        let runtime_dir = dir.join("runtime");
        fs::create_dir_all(&runtime_dir).unwrap();
        let binary = runtime_dir.join(AUDIOCPP_SERVER_NAME);
        fs::write(&binary, b"test executable").unwrap();
        let mut paths = AppPaths::resolve();
        paths.shared_audio_bin_roots = vec![runtime_dir.clone()];
        paths.whisper_models_root = dir.join("models");
        paths.cache_root = dir.join("cache");
        let target = InstallTarget::Engine(LocalEngineKind::AudioCpp);
        let (_command, commands) = tokio::sync::watch::channel(DownloadCommand::Run);
        let mut progress = Vec::new();

        let result = install_controlled(
            &paths,
            &DictateConfig::default(),
            &target,
            commands,
            |stage| {
                progress.push(stage);
            },
        )
        .await;

        assert_eq!(result.unwrap(), DownloadOutcome::Complete);
        assert_eq!(
            progress,
            [
                InstallStatus::Checking,
                InstallStatus::Finalizing,
                InstallStatus::Ready
            ]
        );
        assert!(!paths.whisper_models_root.exists());
        assert!(!paths.cache_root.exists());
        fs::remove_file(binary).unwrap();
        fs::remove_dir(runtime_dir).unwrap();
        fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn parses_whisper_cli_timestamps_and_strips_diagnostics() {
        let sample = "\
whisper_model_load: loading model
[00:00.000 --> 00:02.100]   Hello from Agente Dictate.
[00:02.100 --> 00:04.000]  Fast native Rust speech to text!
ggml_metal_free: deallocating
";
        assert_eq!(
            parse_whisper_output(sample),
            "Hello from Agente Dictate. Fast native Rust speech to text!"
        );
    }

    #[test]
    fn builds_valid_multipart_form_body() {
        let (ct, body) = build_multipart_form(
            &[("model", "gpt-4o-transcribe"), ("language", "en")],
            "file",
            "clip.wav",
            "audio/wav",
            b"RIFF....WAVE",
        );
        assert!(ct.starts_with("multipart/form-data; boundary="));
        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains("name=\"model\"\r\n\r\ngpt-4o-transcribe"));
        assert!(body_str.contains("filename=\"clip.wav\""));
        assert!(body_str.contains("RIFF....WAVE"));
    }
}
