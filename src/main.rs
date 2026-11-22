mod audio;
mod catalog;
mod config;
mod engine;
mod gpu_ui;
mod inject;
mod style;
mod tray;
mod ui;
mod vad;
mod wav;

use crate::audio::{list_input_devices, start_recording, ActiveRecording};
use crate::catalog::{
    find_commercial_provider, find_local_whisper_spec, LocalEngineKind, COMMERCIAL_STT_PROVIDERS,
    COMMERCIAL_TTS_CATALOG, LOCAL_WHISPER_CATALOG,
};
use crate::config::{load_history, save_history, AppPaths, DictateConfig, HistoryEntry, HotkeyMode};
use crate::engine::{
    check_local_whisper_model, has_nvidia_gpu, installed_local_models, is_local_engine_ready,
    resolve_audiocpp_binary, resolve_whisper_binary, setup_local_whisper, stop_whisper_server,
    transcribe_wav, TranscriptionOutput,
};
use crate::gpu_ui::{DrawList, GpuContext, GpuWindowRenderer, TextEngine};
use crate::inject::{
    configure_overlay_window_no_activate, copy_to_clipboard, deliver_transcript,
    paste_from_clipboard, set_overlay_visible_no_activate, spawn_global_hotkey_listener,
    type_unicode_text, GlobalHotkeyEvent,
};
use crate::style::{palette, set_theme, toggle_theme, Rgba, Theme};
use crate::tray::{create_winit_window_icon, spawn_tray_icon, TrayEvent};
use crate::ui::{
    filtered_stt_choices, hit_test, render_main_window, render_overlay_pill, AppView,
    DictationPhase, HitTarget, HitZone, Modal, ModelFilterTab, ProviderField, ScrollList,
    UiSnapshot,
};
use crate::vad::HISTORY_BARS;
use crate::wav::load_wav_normalized;
use anyhow::Result;
use chrono::Utc;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::{CursorIcon, ResizeDirection, Window, WindowId, WindowLevel};

#[derive(Parser)]
#[command(
    name = "enio",
    version,
    about = "Enio — Native GPU-accelerated desktop dictation app with floating system overlay (Non-Whisper + Whisper + Hugging Face + 15 Commercial STTs)"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Show status of local whisper.cpp / audio.cpp runtimes, GPU, HF token, and commercial STT keys
    Status,
    /// Diagnostic check for microphone capture (cpal / WASAPI)
    DiagMic,
    /// List all local (Non-Whisper + Whisper) models, Hugging Face, and commercial STT/TTS models
    Models,
    /// One-step download & install of the local engine and model
    Setup {
        #[arg(default_value = "small_q5")]
        model: String,
    },
    /// Transcribe a WAV file from disk using local or commercial/Hugging Face providers
    Transcribe {
        file: PathBuf,
        #[arg(short, long)]
        backend: Option<String>,
        #[arg(short, long)]
        model: Option<String>,
    },
}

enum WorkerMessage {
    InstallProgress { label: String, progress: u8 },
    InstallDone(Result<String, String>),
    TranscriptionDone {
        duration_secs: f32,
        is_realtime_clause: bool,
        target_hwnd: Option<isize>,
        result: Result<TranscriptionOutput, String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let paths = AppPaths::resolve();
    let mut cfg = DictateConfig::load(&paths);

    if cfg.theme == "light" {
        set_theme(Theme::Light);
    } else {
        set_theme(Theme::Dark);
    }

    let rt = tokio::runtime::Runtime::new()?;

    match cli.command {
        Some(Commands::Status) => {
            print_status(&paths, &cfg);
            return Ok(());
        }
        Some(Commands::DiagMic) => {
            eprintln!("[diag] calling list_input_devices()...");
            let devs = list_input_devices();
            eprintln!("[diag] devices = {devs:?}");
            eprintln!("[diag] calling start_recording(None, false, false)...");
            match start_recording(None, false, false) {
                Ok(rec) => {
                    eprintln!("[diag] recording started on: {}", rec.device_name);
                    std::thread::sleep(Duration::from_millis(600));
                    let (chunks, samples, peak_rms) = rec.diag_stats();
                    eprintln!(
                        "[diag] captured chunks={chunks}, samples={samples}, peak_rms={peak_rms:.5}"
                    );
                    let done = rec.finish();
                    eprintln!(
                        "[diag] finished ok: {} bytes",
                        done.map(|d| d.wav_bytes.len()).unwrap_or(0)
                    );
                }
                Err(e) => eprintln!("[diag] start_recording err: {e}"),
            }
            return Ok(());
        }
        Some(Commands::Models) => {
            print_models(&paths, &cfg);
            return Ok(());
        }
        Some(Commands::Setup { model }) => {
            println!("Setting up local engine + model '{model}'…");
            rt.block_on(setup_local_whisper(&paths, &cfg, &model, |msg, pct| {
                println!("[{pct:>3}%] {msg}");
            }))?;
            cfg.backend = "local".to_string();
            cfg.local_model = model;
            cfg.save(&paths);
            println!("Setup complete.");
            return Ok(());
        }
        Some(Commands::Transcribe {
            file,
            backend,
            model,
        }) => {
            if let Some(b) = backend {
                cfg.backend = b;
            }
            if let Some(m) = model {
                if cfg.backend == "local" {
                    cfg.local_model = m;
                } else {
                    cfg.commercial_model = m;
                }
            }
            let wav_bytes = load_wav_normalized(&file)?;
            let out = rt.block_on(transcribe_wav(&paths, &cfg, &wav_bytes))?;
            println!("{}", out.text);
            stop_whisper_server();
            return Ok(());
        }
        None => {}
    }

    let event_loop = EventLoop::new()?;
    let mut app = DictateDesktopApp::new(paths, cfg, rt.handle().clone());
    let res = event_loop.run_app(&mut app);
    stop_whisper_server();
    res?;
    Ok(())
}

fn print_status(paths: &AppPaths, cfg: &DictateConfig) {
    let whisper_bin = resolve_whisper_binary(paths, cfg);
    let audio_bin = resolve_audiocpp_binary(paths);
    let installed = installed_local_models(paths);
    let microphones = list_input_devices();
    let hf_ready = !cfg.resolve_hf_token().is_empty();
    println!("Enio — System Status");
    println!("  Data folder      : {}", paths.root.display());
    println!("  Active engine    : {}", cfg.active_summary());
    println!("  Hotkey mode      : {}", cfg.hotkey_mode.label());
    println!("  Language         : {}", cfg.language);
    println!("  Delivery mode    : {}", cfg.delivery_mode.label());
    println!(
        "  HF Token         : {}",
        if hf_ready {
            "configured"
        } else {
            "not set (Settings -> Hugging Face Token or $HF_TOKEN)"
        }
    );
    println!(
        "  whisper.cpp bin  : {}",
        whisper_bin
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "not installed".to_string())
    );
    println!(
        "  audio.cpp bin    : {}",
        audio_bin
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "not installed".to_string())
    );
    println!(
        "  NVIDIA CUDA GPU  : {}",
        if has_nvidia_gpu() { "detected" } else { "no" }
    );
    println!(
        "  Local models     : {}",
        if installed.is_empty() {
            "none installed".to_string()
        } else {
            installed.join(", ")
        }
    );
    println!(
        "  Microphones ({})  : {}",
        microphones.len(),
        if microphones.is_empty() {
            "none".to_string()
        } else {
            microphones.join(" | ")
        }
    );
}

fn print_models(paths: &AppPaths, cfg: &DictateConfig) {
    println!("=== On-Device Local Speech Models (Non-Whisper + Whisper) ===");
    for spec in LOCAL_WHISPER_CATALOG {
        let installed = check_local_whisper_model(paths, spec.id).is_some();
        let status = if installed { "installed" } else { "downloadable" };
        println!(
            "  {:<22} {:>4} MB  [{:<12}]  {}",
            spec.id, spec.size_mb, status, spec.label
        );
    }
    println!("\n=== Hugging Face & Commercial Cloud STT Models ===");
    for prov in COMMERCIAL_STT_PROVIDERS {
        let key_state = if cfg.has_api_key(prov.id) {
            "key ready"
        } else {
            "set key"
        };
        println!("  {} ({}) [{}]:", prov.name, prov.id, key_state);
        for m in prov.models {
            println!("    - {:<36} {}", m.id, m.label);
        }
    }
    println!("\n=== Commercial Cloud Text-to-Speech (TTS) Models ===");
    for tts in COMMERCIAL_TTS_CATALOG {
        println!(
            "  - {:<12} {:<26} {}",
            tts.provider_id, tts.model_id, tts.label
        );
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GPU-ACCELERATED DUAL-WINDOW DESKTOP APPLICATION
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct DictateDesktopApp {
    paths: AppPaths,
    config: DictateConfig,
    rt_handle: tokio::runtime::Handle,
    // Single shared wgpu GPU context across both windows (prevents DX12/Vulkan 0xc0000005)
    gpu_ctx: Option<GpuContext>,
    // Main frameless Agente-style window
    main_window: Option<Arc<Window>>,
    main_renderer: Option<GpuWindowRenderer>,
    main_draw: DrawList,
    main_text: TextEngine,
    // Always-on-top floating speech overlay pill
    overlay_window: Option<Arc<Window>>,
    overlay_renderer: Option<GpuWindowRenderer>,
    overlay_draw: DrawList,
    overlay_text: TextEngine,
    overlay_visible: bool,
    // Navigation & UI state
    view: AppView,
    model_tab: ModelFilterTab,
    model_filter: String,
    model_search_focused: bool,
    model_dropdown_open: bool,
    caret_blink_at: Instant,
    last_caret_on: bool,
    model_selected: usize,
    model_scroll: usize,
    hits: Vec<HitZone>,
    mouse_pos: (f32, f32),
    last_hovered: Option<HitTarget>,
    hover_started_at: Instant,
    pressed_target: Option<HitTarget>,
    dragging_scrollbar: Option<(ScrollList, f32, f32, usize, usize)>,
    last_cursor: CursorIcon,
    last_titlebar_click: Option<Instant>,
    modifiers: ModifiersState,
    // Background workers, system tray & global hotkey
    hotkey_rx: mpsc::Receiver<GlobalHotkeyEvent>,
    tray_rx: mpsc::Receiver<TrayEvent>,
    worker_tx: mpsc::Sender<WorkerMessage>,
    worker_rx: mpsc::Receiver<WorkerMessage>,
    available_mics: Vec<String>,
    should_exit: bool,
    history: Vec<HistoryEntry>,
    selected_history: usize,
    history_scroll: usize,
    phase: DictationPhase,
    modal: Modal,
    status_toast: Option<String>,
    active_recording: Option<ActiveRecording>,
    target_hwnd: Option<isize>,
    live_preview: String,
    audio_levels: [f32; 15],
    history_bars: Vec<f32>,
    vad_speaking: bool,
    elapsed_secs: u64,
    active_mic_name: String,
}

impl DictateDesktopApp {
    fn new(
        paths: AppPaths,
        config: DictateConfig,
        rt_handle: tokio::runtime::Handle,
    ) -> Self {
        let hotkey_rx = spawn_global_hotkey_listener();
        let tray_rx = spawn_tray_icon();
        let (worker_tx, worker_rx) = mpsc::channel();
        let history = load_history(&paths);
        let available_mics = list_input_devices();
        let active_mic_name = config
            .audio_device
            .clone()
            .filter(|d| !d.trim().is_empty())
            .or_else(|| available_mics.first().cloned())
            .unwrap_or_else(|| "Default Microphone".to_string());

        Self {
            paths,
            config,
            rt_handle,
            gpu_ctx: None,
            main_window: None,
            main_renderer: None,
            main_draw: DrawList::new(1.0),
            main_text: TextEngine::new(),
            overlay_window: None,
            overlay_renderer: None,
            overlay_draw: DrawList::new(1.0),
            overlay_text: TextEngine::new(),
            overlay_visible: false,
            view: AppView::Dictate,
            model_tab: ModelFilterTab::All,
            model_filter: String::new(),
            model_search_focused: false,
            model_dropdown_open: false,
            caret_blink_at: Instant::now(),
            last_caret_on: true,
            model_selected: 0,
            model_scroll: 0,
            hits: Vec::new(),
            mouse_pos: (-1.0, -1.0),
            last_hovered: None,
            hover_started_at: Instant::now(),
            pressed_target: None,
            dragging_scrollbar: None,
            last_cursor: CursorIcon::Default,
            last_titlebar_click: None,
            modifiers: ModifiersState::default(),
            hotkey_rx,
            tray_rx,
            worker_tx,
            worker_rx,
            available_mics,
            should_exit: false,
            history,
            selected_history: 0,
            history_scroll: 0,
            phase: DictationPhase::Idle,
            modal: Modal::None,
            status_toast: None,
            active_recording: None,
            target_hwnd: None,
            live_preview: String::new(),
            audio_levels: [0.0; 15],
            history_bars: vec![0.0; HISTORY_BARS],
            vad_speaking: false,
            elapsed_secs: 0,
            active_mic_name,
        }
    }

    fn apply_scrollbar_y(
        &mut self,
        list: ScrollList,
        track_y: f32,
        track_h: f32,
        total: usize,
        visible: usize,
        mouse_y: f32,
    ) {
        if total <= visible || track_h <= 1.0 {
            return;
        }
        let max_scroll = total.saturating_sub(visible);
        let thumb_h = (track_h * (visible as f32 / total as f32)).clamp(28.0, track_h);
        let usable = (track_h - thumb_h).max(1.0);
        let ratio = ((mouse_y - track_y - thumb_h * 0.5) / usable).clamp(0.0, 1.0);
        let next = (ratio * max_scroll as f32).round() as usize;
        match list {
            ScrollList::Models => self.model_scroll = next.min(max_scroll),
            ScrollList::History => self.history_scroll = next.min(max_scroll),
        }
    }

    fn request_redraws(&self) {
        if let Some(win) = &self.main_window {
            win.request_redraw();
        }
        if self.overlay_visible {
            if let Some(ov) = &self.overlay_window {
                ov.request_redraw();
            }
        }
    }

    fn set_overlay_visible(&mut self, visible: bool) {
        if self.overlay_visible == visible {
            return;
        }
        self.overlay_visible = visible;
        if let Some(ov) = &self.overlay_window {
            set_overlay_visible_no_activate(ov, visible);
            if visible {
                ov.request_redraw();
            }
        }
    }

    fn start_local_install(&mut self, model_id: String) {
        self.phase = DictationPhase::Installing {
            label: format!("Installing {model_id}"),
            progress: 0,
        };
        let tx = self.worker_tx.clone();
        let paths_clone = self.paths.clone();
        let cfg_clone = self.config.clone();
        self.rt_handle.spawn(async move {
            let tx_p = tx.clone();
            let res = setup_local_whisper(&paths_clone, &cfg_clone, &model_id, move |lbl, pct| {
                let _ = tx_p.send(WorkerMessage::InstallProgress {
                    label: lbl,
                    progress: pct,
                });
            })
            .await
            .map(|()| model_id)
            .map_err(|e| e.to_string());
            let _ = tx.send(WorkerMessage::InstallDone(res));
        });
        self.request_redraws();
    }

    fn dispatch_transcription(
        &self,
        wav_bytes: Vec<u8>,
        duration_secs: f32,
        is_realtime_clause: bool,
        target_hwnd: Option<isize>,
    ) {
        let tx = self.worker_tx.clone();
        let paths_clone = self.paths.clone();
        let cfg_clone = self.config.clone();
        self.rt_handle.spawn(async move {
            let res = transcribe_wav(&paths_clone, &cfg_clone, &wav_bytes)
                .await
                .map_err(|e| e.to_string());
            let _ = tx.send(WorkerMessage::TranscriptionDone {
                duration_secs,
                is_realtime_clause,
                target_hwnd,
                result: res,
            });
        });
    }

    fn start_capture(&mut self, target_hwnd: Option<isize>, show_overlay: bool) {
        if self.active_recording.is_some()
            || matches!(
                self.phase,
                DictationPhase::Transcribing | DictationPhase::Installing { .. }
            )
        {
            return;
        }

        if self.config.backend == "local" && !is_local_engine_ready(&self.paths, &self.config) {
            self.start_local_install(self.config.local_model.clone());
            return;
        }

        let realtime = self.config.hotkey_mode == HotkeyMode::RealTimeStreaming;
        match start_recording(
            self.config.audio_device.as_deref(),
            self.config.vad_auto_stop,
            realtime,
        ) {
            Ok(rec) => {
                self.active_mic_name = rec.device_name.clone();
                self.active_recording = Some(rec);
                self.target_hwnd = target_hwnd;
                self.live_preview.clear();
                self.phase = DictationPhase::Recording;
                self.status_toast = Some("Listening…".to_string());
                if show_overlay {
                    self.set_overlay_visible(true);
                }
            }
            Err(err) => {
                self.phase = DictationPhase::Error(err.to_string());
                self.status_toast = Some(format!("Mic error: {err}"));
            }
        }
        self.request_redraws();
    }

    fn stop_capture_and_transcribe(&mut self) {
        if let Some(rec) = self.active_recording.take() {
            if let Some(done) = rec.finish() {
                self.phase = DictationPhase::Transcribing;
                self.dispatch_transcription(
                    done.wav_bytes,
                    done.duration_secs,
                    false,
                    self.target_hwnd,
                );
            } else {
                self.phase = DictationPhase::Idle;
                self.set_overlay_visible(false);
            }
            self.request_redraws();
        }
    }

    fn toggle_recording(&mut self, target_hwnd: Option<isize>, show_overlay: bool) {
        if self.active_recording.is_some() {
            self.stop_capture_and_transcribe();
        } else {
            self.start_capture(target_hwnd, show_overlay);
        }
    }

    fn open_provider_config_modal(&mut self, provider_id: String) {
        let spec = find_commercial_provider(&provider_id);
        let pname = spec
            .map(|p| p.name.to_string())
            .unwrap_or_else(|| provider_id.clone());
        let is_self_hosted = provider_id == "custom";
        let saved = self.config.providers.get(&provider_id);
        let api_key = if provider_id == "huggingface" {
            self.config.resolve_hf_token()
        } else {
            saved.map(|p| p.api_key.clone()).unwrap_or_default()
        };
        let base_url = saved
            .map(|p| p.base_url.clone())
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| {
                if is_self_hosted {
                    spec.map(|p| p.default_base_url.to_string())
                        .unwrap_or_else(|| "http://127.0.0.1:8000/v1".to_string())
                } else {
                    String::new()
                }
            });
        let model_id = saved
            .map(|p| p.model.clone())
            .filter(|m| !m.trim().is_empty())
            .unwrap_or_else(|| {
                if self.config.backend == provider_id && !self.config.commercial_model.is_empty() {
                    self.config.commercial_model.clone()
                } else {
                    spec.map(|p| p.default_model.to_string()).unwrap_or_default()
                }
            });
        self.model_dropdown_open = false;
        self.caret_blink_at = Instant::now();
        self.last_caret_on = true;
        self.modal = Modal::ProviderConfig {
            provider_id,
            provider_name: pname,
            is_self_hosted,
            base_url,
            model_id,
            api_key,
            active_field: if is_self_hosted {
                ProviderField::BaseUrl
            } else {
                ProviderField::ApiKey
            },
            reveal_key: false,
        };
        self.request_redraws();
    }

    fn open_confirm_download_modal(&mut self, model_id: String) {
        let spec = find_local_whisper_spec(&model_id);
        let label = spec
            .map(|s| s.label.to_string())
            .unwrap_or_else(|| model_id.clone());
        let size_mb = spec.map(|s| s.size_mb).unwrap_or(190);
        let engine_name = spec
            .map(|s| match s.engine {
                LocalEngineKind::AudioCpp => "audio.cpp",
                LocalEngineKind::WhisperCpp => "whisper.cpp",
            })
            .unwrap_or("whisper.cpp")
            .to_string();
        self.model_dropdown_open = false;
        self.modal = Modal::ConfirmDownload {
            model_id,
            label,
            size_mb,
            engine_name,
        };
        self.request_redraws();
    }

    fn pick_model_index(&mut self, index: usize) {
        let choices = filtered_stt_choices(
            &self.paths,
            &self.config,
            self.model_tab,
            &self.model_filter,
        );
        let Some(choice) = choices.get(index).cloned() else {
            return;
        };
        self.model_selected = index;

        if choice.is_local {
            let mut probe_cfg = self.config.clone();
            probe_cfg.backend = "local".to_string();
            probe_cfg.local_model = choice.model_id.clone();

            if !is_local_engine_ready(&self.paths, &probe_cfg) {
                self.open_confirm_download_modal(choice.model_id);
            } else {
                self.config.backend = "local".to_string();
                self.config.local_model = choice.model_id.clone();
                self.config.save(&self.paths);
                stop_whisper_server();
                self.status_toast = Some(format!("Switched to {}", choice.model_id));
                self.view = AppView::Dictate;
            }
        } else {
            self.config.backend = choice.backend.clone();
            self.config.commercial_model = choice.model_id.clone();
            self.config.save(&self.paths);
            if !self.config.has_api_key(&choice.backend) {
                self.open_provider_config_modal(choice.backend);
            } else {
                self.status_toast = Some(format!("Switched to {}", choice.model_id));
                self.view = AppView::Dictate;
            }
        }
        self.request_redraws();
    }

    fn execute_hit(&mut self, _event_loop: &ActiveEventLoop, target: HitTarget) {
        match target {
            HitTarget::DragWindow => {
                let now = Instant::now();
                let is_double = self
                    .last_titlebar_click
                    .is_some_and(|t| now.duration_since(t) < Duration::from_millis(320));
                if is_double {
                    self.last_titlebar_click = None;
                    if let Some(win) = &self.main_window {
                        win.set_maximized(!win.is_maximized());
                    }
                } else {
                    self.last_titlebar_click = Some(now);
                    if let Some(win) = &self.main_window {
                        let _ = win.drag_window();
                    }
                }
                return;
            }
            HitTarget::MinimizeWindow => {
                if let Some(win) = &self.main_window {
                    win.set_minimized(true);
                }
                return;
            }
            HitTarget::MaximizeWindow => {
                if let Some(win) = &self.main_window {
                    win.set_maximized(!win.is_maximized());
                }
                return;
            }
            HitTarget::CloseWindow => {
                // Close hides to the System Tray (Activity Bar) so global dictation stays active
                if let Some(win) = &self.main_window {
                    win.set_visible(false);
                }
                return;
            }
            HitTarget::SwitchView(v) => {
                self.view = v;
                self.model_dropdown_open = false;
            }
            HitTarget::ToggleRecord => {
                self.model_dropdown_open = false;
                self.toggle_recording(None, false);
            }
            HitTarget::ToggleModelDropdown => {
                self.model_dropdown_open = !self.model_dropdown_open;
            }
            HitTarget::CloseModelDropdown => {
                self.model_dropdown_open = false;
            }
            HitTarget::SelectDropdownLocal(model_id) => {
                self.model_dropdown_open = false;
                self.config.backend = "local".to_string();
                self.config.local_model = model_id.clone();
                self.config.save(&self.paths);
                stop_whisper_server();
                self.status_toast = Some(format!("Switched to {model_id}"));
            }
            HitTarget::SelectDropdownCloud { backend, model_id } => {
                self.model_dropdown_open = false;
                self.config.backend = backend;
                self.config.commercial_model = model_id.clone();
                self.config.save(&self.paths);
                self.status_toast = Some(format!("Switched to {model_id}"));
            }
            HitTarget::InstallLocalEngine => {
                self.open_confirm_download_modal(self.config.local_model.clone());
            }
            HitTarget::CycleLanguage => {
                self.config.cycle_language();
                self.config.save(&self.paths);
                self.status_toast = Some(format!("Language: {}", self.config.language));
            }
            HitTarget::CycleHotkeyMode => {
                self.config.hotkey_mode = self.config.hotkey_mode.next();
                self.config.save(&self.paths);
                self.status_toast = Some(self.config.hotkey_mode.label().to_string());
            }
            HitTarget::RefreshMicrophones => {
                self.available_mics = list_input_devices();
                if self.available_mics.is_empty() {
                    self.active_mic_name = "No microphone detected".to_string();
                    self.status_toast = Some(
                        "No microphone found — check Windows Microphone privacy settings"
                            .to_string(),
                    );
                } else {
                    if self.config.audio_device.is_none() {
                        self.active_mic_name = self.available_mics[0].clone();
                    }
                    self.status_toast = Some(format!(
                        "Found {} microphone(s)",
                        self.available_mics.len()
                    ));
                }
            }
            HitTarget::CycleMicrophone => {
                if self.available_mics.is_empty() {
                    self.available_mics = list_input_devices();
                }
                if !self.available_mics.is_empty() {
                    let cur = self
                        .available_mics
                        .iter()
                        .position(|m| m == &self.active_mic_name)
                        .unwrap_or(0);
                    let next = (cur + 1) % self.available_mics.len();
                    let chosen = self.available_mics[next].clone();
                    self.active_mic_name = chosen.clone();
                    self.config.audio_device = Some(chosen.clone());
                    self.config.save(&self.paths);
                    self.status_toast = Some(format!("Microphone: {chosen}"));
                } else {
                    self.status_toast = Some("No microphones detected".to_string());
                }
            }
            HitTarget::ToggleVad => {
                self.config.vad_auto_stop = !self.config.vad_auto_stop;
                self.config.save(&self.paths);
            }
            HitTarget::CycleDelivery => {
                self.config.delivery_mode = self.config.delivery_mode.next();
                self.config.save(&self.paths);
                self.status_toast = Some(self.config.delivery_mode.label().to_string());
            }
            HitTarget::ToggleTheme => {
                let next = toggle_theme();
                self.config.theme = match next {
                    Theme::Dark => "dark".to_string(),
                    Theme::Light => "light".to_string(),
                };
                self.config.save(&self.paths);
            }
            HitTarget::OpenProviderConfig(pid) => self.open_provider_config_modal(pid),
            HitTarget::FocusProviderField(field) => {
                if let Modal::ProviderConfig { active_field, .. } = &mut self.modal {
                    *active_field = field;
                    self.caret_blink_at = Instant::now();
                    self.last_caret_on = true;
                }
            }
            HitTarget::ToggleRevealKey => {
                if let Modal::ProviderConfig { reveal_key, .. } = &mut self.modal {
                    *reveal_key = !*reveal_key;
                }
            }
            HitTarget::FocusModelSearch => {
                self.model_search_focused = true;
                self.caret_blink_at = Instant::now();
                self.last_caret_on = true;
            }
            HitTarget::ClearModelSearch => {
                self.model_filter.clear();
                self.model_selected = 0;
                self.model_scroll = 0;
                self.model_search_focused = true;
                self.caret_blink_at = Instant::now();
                self.last_caret_on = true;
            }
            HitTarget::Scrollbar {
                list,
                track_y,
                track_h,
                total,
                visible,
            } => {
                self.apply_scrollbar_y(
                    list,
                    track_y as f32,
                    track_h as f32,
                    total,
                    visible,
                    self.mouse_pos.1,
                );
            }
            HitTarget::SetModelTab(tab) => {
                self.model_tab = tab;
                self.model_selected = 0;
                self.model_scroll = 0;
            }
            HitTarget::PickModel(idx) => {
                self.pick_model_index(idx);
            }
            HitTarget::ActivateLocalModel(model_id) => {
                self.config.backend = "local".to_string();
                self.config.local_model = model_id.clone();
                self.config.save(&self.paths);
                stop_whisper_server();
                self.status_toast = Some(format!("Switched to local model: {model_id}"));
            }
            HitTarget::SelectHistory(idx) => {
                if idx < self.history.len() {
                    self.selected_history = idx;
                }
            }
            HitTarget::CopyHistory(idx) => {
                if let Some(entry) = self.history.get(idx) {
                    self.selected_history = idx;
                    match copy_to_clipboard(&entry.text) {
                        Ok(()) => self.status_toast = Some("Copied to clipboard".to_string()),
                        Err(e) => self.status_toast = Some(format!("Clipboard error: {e}")),
                    }
                }
            }
            HitTarget::TypeHistory(idx) => {
                if let Some(entry) = self.history.get(idx) {
                    self.selected_history = idx;
                    let _ = copy_to_clipboard(&entry.text);
                    match type_unicode_text(&entry.text) {
                        Ok(()) => self.status_toast = Some("Typed into active window".to_string()),
                        Err(e) => self.status_toast = Some(format!("Type error: {e}")),
                    }
                }
            }
            HitTarget::RequestDeleteHistory(idx) => {
                if let Some(entry) = self.history.get(idx) {
                    let preview = entry.text.trim().to_string();
                    self.modal = Modal::ConfirmDeleteClip { index: idx, preview };
                }
            }
            HitTarget::SaveProviderConfig => {
                if let Modal::ProviderConfig {
                    provider_id,
                    provider_name,
                    base_url,
                    model_id,
                    api_key,
                    ..
                } = &self.modal
                {
                    let pid = provider_id.clone();
                    let pname = provider_name.clone();
                    self.config.set_provider_config(
                        &pid,
                        api_key.clone(),
                        base_url.clone(),
                        model_id.clone(),
                    );
                    self.config.save(&self.paths);
                    self.status_toast = Some(format!("Saved {pname} configuration"));
                    self.modal = Modal::None;
                }
            }
            HitTarget::ConfirmDownloadModel => {
                if let Modal::ConfirmDownload { model_id, .. } = &self.modal {
                    let mid = model_id.clone();
                    self.modal = Modal::None;
                    self.config.backend = "local".to_string();
                    self.config.local_model = mid.clone();
                    self.config.save(&self.paths);
                    stop_whisper_server();
                    self.view = AppView::Dictate;
                    self.start_local_install(mid);
                }
            }
            HitTarget::ConfirmDeleteClip => {
                if let Modal::ConfirmDeleteClip { index, .. } = &self.modal {
                    let idx = *index;
                    if idx < self.history.len() {
                        self.history.remove(idx);
                        if self.selected_history >= self.history.len() && self.selected_history > 0
                        {
                            self.selected_history -= 1;
                        }
                        save_history(&self.paths, &self.history);
                        self.status_toast = Some("Deleted clip".to_string());
                    }
                    self.modal = Modal::None;
                }
            }
            HitTarget::CloseModal => {
                self.modal = Modal::None;
            }
        }
        self.request_redraws();
    }

    fn process_background_events(&mut self) -> bool {
        let mut changed = false;

        let mut vad_auto_triggered = false;
        let mut realtime_clause = None;
        if let Some(rec) = self.active_recording.as_ref() {
            self.elapsed_secs = rec.elapsed_secs();
            let (levels, hist, speaking) = rec.snapshot_waveform();
            self.audio_levels = levels;
            self.history_bars = hist;
            self.vad_speaking = speaking;
            if let Some(clause) = rec.take_completed_clause() {
                realtime_clause = Some(clause);
            }
            if rec.should_stop.load(Ordering::Relaxed) {
                vad_auto_triggered = true;
            }
            changed = true;
        } else {
            self.audio_levels = [0.0; 15];
            self.vad_speaking = false;
            self.elapsed_secs = 0;
            if matches!(self.phase, DictationPhase::Transcribing) {
                changed = true;
            }
        }

        if let Some(clause) = realtime_clause {
            self.dispatch_transcription(
                clause.wav_bytes,
                clause.duration_secs,
                true,
                self.target_hwnd,
            );
            changed = true;
        }

        if vad_auto_triggered {
            self.stop_capture_and_transcribe();
            changed = true;
        }

        while let Ok(ev) = self.tray_rx.try_recv() {
            changed = true;
            match ev {
                TrayEvent::ShowMainWindow => {
                    if let Some(win) = &self.main_window {
                        win.set_visible(true);
                        win.set_minimized(false);
                        win.focus_window();
                        win.request_redraw();
                    }
                }
                TrayEvent::ToggleDictation => {
                    self.toggle_recording(None, true);
                }
                TrayEvent::Quit => {
                    self.should_exit = true;
                }
            }
        }

        while let Ok(ev) = self.hotkey_rx.try_recv() {
            changed = true;
            match ev {
                GlobalHotkeyEvent::Pressed { target_hwnd } => {
                    if self.active_recording.is_some() {
                        self.stop_capture_and_transcribe();
                    } else {
                        self.start_capture(Some(target_hwnd), true);
                    }
                }
                GlobalHotkeyEvent::Released { held_ms } => {
                    if self.active_recording.is_some() {
                        match self.config.hotkey_mode {
                            HotkeyMode::HoldToTalk => self.stop_capture_and_transcribe(),
                            HotkeyMode::SmartHoldOrTap => {
                                if held_ms >= 350 {
                                    self.stop_capture_and_transcribe();
                                }
                            }
                            HotkeyMode::RealTimeStreaming => {}
                        }
                    }
                }
            }
        }

        while let Ok(msg) = self.worker_rx.try_recv() {
            changed = true;
            match msg {
                WorkerMessage::InstallProgress { label, progress } => {
                    self.phase = DictationPhase::Installing { label, progress };
                }
                WorkerMessage::InstallDone(Ok(model_id)) => {
                    self.phase = DictationPhase::Idle;
                    self.set_overlay_visible(false);
                    self.status_toast = Some(format!("Installed {model_id}"));
                }
                WorkerMessage::InstallDone(Err(err)) => {
                    self.phase = DictationPhase::Error(err.clone());
                    self.set_overlay_visible(false);
                    self.status_toast = Some(format!("Setup failed: {err}"));
                }
                WorkerMessage::TranscriptionDone {
                    duration_secs,
                    is_realtime_clause,
                    target_hwnd,
                    result,
                } => match result {
                    Ok(out) => {
                        if !is_realtime_clause {
                            self.phase = DictationPhase::Idle;
                            self.set_overlay_visible(false);
                        }
                        if !out.text.trim().is_empty() {
                            let to_inject = if is_realtime_clause {
                                format!("{} ", out.text.trim())
                            } else {
                                out.text.clone()
                            };
                            self.live_preview = out.text.clone();
                            let delivery_note = match deliver_transcript(
                                &to_inject,
                                self.config.delivery_mode,
                                target_hwnd,
                            ) {
                                Ok(note) => note,
                                Err(e) => format!("Copied to clipboard ({e})"),
                            };
                            let entry = HistoryEntry {
                                id: uuid::Uuid::new_v4().to_string(),
                                text: out.text,
                                created_at: Utc::now(),
                                duration_secs,
                                backend: out.backend,
                                model: out.model,
                                elapsed_ms: out.elapsed_ms,
                            };
                            self.history.insert(0, entry);
                            self.selected_history = 0;
                            self.history_scroll = 0;
                            save_history(&self.paths, &self.history);
                            self.status_toast =
                                Some(format!("{delivery_note} ({}ms)", out.elapsed_ms));
                        } else if !is_realtime_clause {
                            self.status_toast = Some(
                                "No speech detected — check Microphone level in Settings"
                                    .to_string(),
                            );
                        }
                    }
                    Err(err) => {
                        self.phase = DictationPhase::Error(err.clone());
                        if !is_realtime_clause {
                            self.set_overlay_visible(false);
                        }
                        self.status_toast = Some(err);
                    }
                },
            }
        }

        changed
    }
}

impl ApplicationHandler for DictateDesktopApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.main_window.is_some() {
            return;
        }
        // 1. Custom Frameless Main Application Window
        let main_attrs = Window::default_attributes()
            .with_title("Enio")
            .with_decorations(false)
            .with_window_icon(create_winit_window_icon())
            .with_inner_size(LogicalSize::new(980.0, 640.0))
            .with_min_inner_size(LogicalSize::new(720.0, 460.0));

        if let Ok(window) = event_loop.create_window(main_attrs) {
            let win = Arc::new(window);
            if let Ok((ctx, renderer)) = GpuContext::new(Arc::clone(&win)) {
                self.gpu_ctx = Some(ctx);
                self.main_renderer = Some(renderer);
            }
            self.main_window = Some(win);
        }

        // 2. Floating Always-On-Top Speech Overlay Pill (hidden until Ctrl+Shift+Space)
        let overlay_w = 420.0_f64;
        let overlay_h = 74.0_f64;
        let mut ov_attrs = Window::default_attributes()
            .with_title("Enio Overlay")
            .with_decorations(false)
            .with_transparent(true)
            .with_resizable(false)
            .with_visible(false)
            .with_window_level(WindowLevel::AlwaysOnTop)
            .with_inner_size(LogicalSize::new(overlay_w, overlay_h));

        if let Some(monitor) = event_loop.primary_monitor() {
            let msize = monitor.size();
            let mpos = monitor.position();
            let scale = monitor.scale_factor();
            let phys_w = (overlay_w * scale) as i32;
            let phys_h = (overlay_h * scale) as i32;
            let x = mpos.x + ((msize.width as i32) - phys_w) / 2;
            let y = mpos.y + (msize.height as i32) - phys_h - (84.0 * scale) as i32;
            ov_attrs = ov_attrs.with_position(PhysicalPosition::new(x, y));
        }

        if let Ok(ov_win) = event_loop.create_window(ov_attrs) {
            configure_overlay_window_no_activate(&ov_win);
            self.overlay_window = Some(Arc::new(ov_win));
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.process_background_events() {
            self.request_redraws();
        }
        if self.should_exit {
            event_loop.exit();
            return;
        }
        let hover_animating =
            self.last_hovered.is_some() && self.hover_started_at.elapsed() < Duration::from_millis(120);
        if hover_animating {
            self.request_redraws();
        }
        if (self.view == AppView::Models && self.model_search_focused)
            || matches!(self.modal, Modal::ProviderConfig { .. })
        {
            let caret_on = (self.caret_blink_at.elapsed().as_millis() / 530) % 2 == 0;
            if caret_on != self.last_caret_on {
                self.last_caret_on = caret_on;
                self.request_redraws();
            }
        }
        let poll_ms = if self.active_recording.is_some()
            || hover_animating
            || matches!(
                self.phase,
                DictationPhase::Transcribing | DictationPhase::Installing { .. }
            ) {
            16 // 60 fps live waveform & hover ease-out
        } else {
            50
        };
        event_loop.set_control_flow(ControlFlow::WaitUntil(
            Instant::now() + Duration::from_millis(poll_ms),
        ));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let is_overlay = self
            .overlay_window
            .as_ref()
            .is_some_and(|w| w.id() == window_id);

        if is_overlay {
            match event {
                WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                    if let (Some(ctx), Some(r)) = (&self.gpu_ctx, &mut self.overlay_renderer) {
                        r.resize(ctx, size.width, size.height);
                    }
                }
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button: MouseButton::Left,
                    ..
                } => {
                    if self.active_recording.is_some() {
                        self.stop_capture_and_transcribe();
                    }
                }
                WindowEvent::RedrawRequested => {
                    if !self.overlay_visible {
                        return;
                    }
                    if let (Some(ctx), Some(ov)) = (&self.gpu_ctx, &self.overlay_window) {
                        let size = ov.inner_size();
                        if size.width == 0 || size.height == 0 {
                            return;
                        }
                        if self.overlay_renderer.is_none() {
                            if let Ok(renderer) = ctx.create_window_renderer(Arc::clone(ov), true) {
                                self.overlay_renderer = Some(renderer);
                            }
                        }
                        if let Some(r) = &mut self.overlay_renderer {
                            let scale = ov.scale_factor() as f32;
                            let logical_w = size.width as f32 / scale;
                            let logical_h = size.height as f32 / scale;
                            self.overlay_draw.clear(scale);
                            render_overlay_pill(
                                &mut self.overlay_draw,
                                &mut self.overlay_text,
                                logical_w,
                                logical_h,
                                &self.config,
                                &self.phase,
                                self.elapsed_secs,
                                self.audio_levels,
                                &self.live_preview,
                            );
                            let _ = r.render(
                                ctx,
                                Rgba::transparent(),
                                &self.overlay_draw,
                                &mut self.overlay_text,
                            );
                        }
                    }
                }
                _ => {}
            }
            return;
        }

        // Main window events
        let scale = self
            .main_window
            .as_ref()
            .map(|w| w.scale_factor() as f32)
            .unwrap_or(1.0);

        match event {
            WindowEvent::CloseRequested => {
                // Hide to the Activity Bar / System Tray instead of quitting
                if let Some(win) = &self.main_window {
                    win.set_visible(false);
                }
            }
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    if let (Some(ctx), Some(renderer)) = (&self.gpu_ctx, &mut self.main_renderer) {
                        renderer.resize(ctx, size.width, size.height);
                    }
                    self.request_redraws();
                }
            }
            WindowEvent::ModifiersChanged(mods) => {
                self.modifiers = mods.state();
            }
            WindowEvent::CursorMoved { position, .. } => {
                let lx = position.x as f32 / scale;
                let ly = position.y as f32 / scale;
                self.mouse_pos = (lx, ly);

                if let Some((list, track_y, track_h, total, visible)) = self.dragging_scrollbar {
                    self.apply_scrollbar_y(list, track_y, track_h, total, visible, ly);
                    self.request_redraws();
                    return;
                }

                let (win_w, win_h, is_max) = self
                    .main_window
                    .as_ref()
                    .map(|w| {
                        let sz = w.inner_size();
                        (sz.width as f32 / scale, sz.height as f32 / scale, w.is_maximized())
                    })
                    .unwrap_or((980.0, 640.0, false));

                let resize_dir = if !is_max {
                    edge_resize_direction(lx, ly, win_w, win_h)
                } else {
                    None
                };
                let hovered = if resize_dir.is_some() {
                    None
                } else {
                    hit_test(&self.hits, lx, ly)
                };

                let desired_cursor = if let Some(dir) = resize_dir {
                    resize_cursor(dir)
                } else if hovered.as_ref().is_some_and(HitTarget::is_text_input) {
                    CursorIcon::Text
                } else if hovered.as_ref().is_some_and(HitTarget::is_pointer) {
                    CursorIcon::Pointer
                } else {
                    CursorIcon::Default
                };

                if desired_cursor != self.last_cursor {
                    self.last_cursor = desired_cursor;
                    if let Some(win) = &self.main_window {
                        win.set_cursor(desired_cursor);
                    }
                }

                // Only repaint when hover target actually changes (zero wasted redraws)
                if hovered != self.last_hovered {
                    self.last_hovered = hovered;
                    self.hover_started_at = Instant::now();
                    self.request_redraws();
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if let Some(win) = &self.main_window {
                    if !win.is_maximized() {
                        let sz = win.inner_size();
                        let lw = sz.width as f32 / scale;
                        let lh = sz.height as f32 / scale;
                        if let Some(dir) =
                            edge_resize_direction(self.mouse_pos.0, self.mouse_pos.1, lw, lh)
                        {
                            let _ = win.drag_resize_window(dir);
                            return;
                        }
                    }
                }
                if let Some(target) = hit_test(&self.hits, self.mouse_pos.0, self.mouse_pos.1) {
                    let keep_search_focus = matches!(
                        target,
                        HitTarget::FocusModelSearch | HitTarget::ClearModelSearch
                    );
                    if !keep_search_focus && self.model_search_focused {
                        self.model_search_focused = false;
                    }
                    match target {
                        HitTarget::DragWindow => {
                            self.execute_hit(event_loop, target);
                        }
                        HitTarget::FocusModelSearch => {
                            self.model_search_focused = true;
                            self.caret_blink_at = Instant::now();
                            self.last_caret_on = true;
                            self.request_redraws();
                        }
                        HitTarget::FocusProviderField(field) => {
                            if let Modal::ProviderConfig { active_field, .. } = &mut self.modal {
                                *active_field = field;
                                self.caret_blink_at = Instant::now();
                                self.last_caret_on = true;
                                self.request_redraws();
                            }
                        }
                        HitTarget::Scrollbar {
                            list,
                            track_y,
                            track_h,
                            total,
                            visible,
                        } => {
                            self.dragging_scrollbar =
                                Some((list, track_y as f32, track_h as f32, total, visible));
                            self.apply_scrollbar_y(
                                list,
                                track_y as f32,
                                track_h as f32,
                                total,
                                visible,
                                self.mouse_pos.1,
                            );
                            self.request_redraws();
                        }
                        other => {
                            self.pressed_target = Some(other);
                            self.request_redraws();
                        }
                    }
                } else if self.model_search_focused {
                    self.model_search_focused = false;
                    self.request_redraws();
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                if self.dragging_scrollbar.take().is_some() {
                    self.request_redraws();
                }
                if let Some(pressed) = self.pressed_target.take() {
                    let still_over =
                        hit_test(&self.hits, self.mouse_pos.0, self.mouse_pos.1) == Some(pressed.clone());
                    if still_over {
                        self.execute_hit(event_loop, pressed);
                    } else {
                        self.request_redraws();
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let step: isize = match delta {
                    MouseScrollDelta::LineDelta(_, y) => {
                        if y > 0.0 {
                            -1
                        } else if y < 0.0 {
                            1
                        } else {
                            0
                        }
                    }
                    MouseScrollDelta::PixelDelta(pos) => {
                        if pos.y > 0.0 {
                            -1
                        } else if pos.y < 0.0 {
                            1
                        } else {
                            0
                        }
                    }
                };
                if step != 0 {
                    match self.view {
                        AppView::Models => {
                            let total = filtered_stt_choices(
                                &self.paths,
                                &self.config,
                                self.model_tab,
                                &self.model_filter,
                            )
                            .len();
                            let visible = self
                                .hits
                                .iter()
                                .find_map(|h| match h.target {
                                    HitTarget::Scrollbar {
                                        list: ScrollList::Models,
                                        visible,
                                        ..
                                    } => Some(visible),
                                    _ => None,
                                })
                                .unwrap_or(1);
                            let max_scroll = total.saturating_sub(visible);
                            if max_scroll > 0 {
                                self.model_scroll = (self.model_scroll as isize + step * 2)
                                    .clamp(0, max_scroll as isize)
                                    as usize;
                            }
                        }
                        AppView::Dictate => {
                            let total = self.history.len();
                            let visible = self
                                .hits
                                .iter()
                                .find_map(|h| match h.target {
                                    HitTarget::Scrollbar {
                                        list: ScrollList::History,
                                        visible,
                                        ..
                                    } => Some(visible),
                                    _ => None,
                                })
                                .unwrap_or(1);
                            let max_scroll = total.saturating_sub(visible);
                            if max_scroll > 0 {
                                self.history_scroll = (self.history_scroll as isize + step)
                                    .clamp(0, max_scroll as isize)
                                    as usize;
                            }
                        }
                        AppView::Settings => {}
                    }
                    self.request_redraws();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state != ElementState::Pressed {
                    return;
                }
                let ctrl = self.modifiers.control_key();

                if ctrl
                    && matches!(&event.logical_key, Key::Character(c) if c.eq_ignore_ascii_case("t"))
                {
                    self.execute_hit(event_loop, HitTarget::ToggleTheme);
                    return;
                }

                // Active modal keyboard handling (Provider Config, Confirm Download, Confirm Delete)
                match &mut self.modal {
                    Modal::None => {}
                    Modal::ConfirmDownload { .. } => {
                        match &event.logical_key {
                            Key::Named(NamedKey::Escape) => self.modal = Modal::None,
                            Key::Named(NamedKey::Enter) => {
                                self.execute_hit(event_loop, HitTarget::ConfirmDownloadModel);
                                return;
                            }
                            _ => {}
                        }
                        self.request_redraws();
                        return;
                    }
                    Modal::ConfirmDeleteClip { .. } => {
                        match &event.logical_key {
                            Key::Named(NamedKey::Escape) => self.modal = Modal::None,
                            Key::Named(NamedKey::Enter) => {
                                self.execute_hit(event_loop, HitTarget::ConfirmDeleteClip);
                                return;
                            }
                            _ => {}
                        }
                        self.request_redraws();
                        return;
                    }
                    Modal::ProviderConfig {
                        is_self_hosted,
                        base_url,
                        model_id,
                        api_key,
                        active_field,
                        ..
                    } => {
                        let shift = self.modifiers.shift_key();
                        match &event.logical_key {
                            Key::Named(NamedKey::Escape) => self.modal = Modal::None,
                            Key::Named(NamedKey::Enter) => {
                                self.execute_hit(event_loop, HitTarget::SaveProviderConfig);
                                return;
                            }
                            Key::Named(NamedKey::Tab) => {
                                *active_field = if *is_self_hosted {
                                    match (*active_field, shift) {
                                        (ProviderField::BaseUrl, false) => ProviderField::ModelId,
                                        (ProviderField::ModelId, false) => ProviderField::ApiKey,
                                        (ProviderField::ApiKey, false) => ProviderField::BaseUrl,
                                        (ProviderField::BaseUrl, true) => ProviderField::ApiKey,
                                        (ProviderField::ModelId, true) => ProviderField::BaseUrl,
                                        (ProviderField::ApiKey, true) => ProviderField::ModelId,
                                    }
                                } else {
                                    match *active_field {
                                        ProviderField::ApiKey => ProviderField::BaseUrl,
                                        _ => ProviderField::ApiKey,
                                    }
                                };
                                self.caret_blink_at = Instant::now();
                                self.last_caret_on = true;
                            }
                            Key::Named(NamedKey::Backspace) => {
                                let target_buf = match *active_field {
                                    ProviderField::BaseUrl => base_url,
                                    ProviderField::ModelId => model_id,
                                    ProviderField::ApiKey => api_key,
                                };
                                if ctrl {
                                    target_buf.clear();
                                } else {
                                    target_buf.pop();
                                }
                                self.caret_blink_at = Instant::now();
                                self.last_caret_on = true;
                            }
                            Key::Character(c) if ctrl && c.eq_ignore_ascii_case("v") => {
                                if let Some(clip) = paste_from_clipboard() {
                                    let target_buf = match *active_field {
                                        ProviderField::BaseUrl => base_url,
                                        ProviderField::ModelId => model_id,
                                        ProviderField::ApiKey => api_key,
                                    };
                                    target_buf.push_str(clip.trim());
                                    self.caret_blink_at = Instant::now();
                                    self.last_caret_on = true;
                                }
                            }
                            Key::Character(c) if ctrl && c.eq_ignore_ascii_case("u") => {
                                let target_buf = match *active_field {
                                    ProviderField::BaseUrl => base_url,
                                    ProviderField::ModelId => model_id,
                                    ProviderField::ApiKey => api_key,
                                };
                                target_buf.clear();
                                self.caret_blink_at = Instant::now();
                                self.last_caret_on = true;
                            }
                            Key::Character(c) if !ctrl => {
                                let target_buf = match *active_field {
                                    ProviderField::BaseUrl => base_url,
                                    ProviderField::ModelId => model_id,
                                    ProviderField::ApiKey => api_key,
                                };
                                target_buf.push_str(c.as_str());
                                self.caret_blink_at = Instant::now();
                                self.last_caret_on = true;
                            }
                            _ => {}
                        }
                        self.request_redraws();
                        return;
                    }
                }

                // Models page live search & navigation
                if self.view == AppView::Models {
                    let choices = filtered_stt_choices(
                        &self.paths,
                        &self.config,
                        self.model_tab,
                        &self.model_filter,
                    );
                    let total = choices.len();
                    match &event.logical_key {
                        Key::Named(NamedKey::Escape) => {
                            if !self.model_filter.is_empty() {
                                self.model_filter.clear();
                                self.model_selected = 0;
                                self.model_scroll = 0;
                            } else if self.model_search_focused {
                                self.model_search_focused = false;
                            } else {
                                self.view = AppView::Dictate;
                            }
                        }
                        Key::Named(NamedKey::ArrowUp) => {
                            if self.model_selected > 0 {
                                self.model_selected -= 1;
                                if self.model_selected < self.model_scroll {
                                    self.model_scroll = self.model_selected;
                                }
                            }
                        }
                        Key::Named(NamedKey::ArrowDown) => {
                            if self.model_selected + 1 < total {
                                self.model_selected += 1;
                                if self.model_selected >= self.model_scroll + 8 {
                                    self.model_scroll = self.model_selected.saturating_sub(7);
                                }
                            }
                        }
                        Key::Named(NamedKey::PageUp) => {
                            self.model_scroll = self.model_scroll.saturating_sub(8);
                            self.model_selected = self.model_selected.saturating_sub(8);
                        }
                        Key::Named(NamedKey::PageDown) if total > 0 => {
                            self.model_scroll = (self.model_scroll + 8).min(total.saturating_sub(1));
                            self.model_selected = (self.model_selected + 8).min(total - 1);
                        }
                        Key::Named(NamedKey::Backspace) => {
                            if ctrl {
                                self.model_filter.clear();
                            } else {
                                self.model_filter.pop();
                            }
                            self.model_search_focused = true;
                            self.caret_blink_at = Instant::now();
                            self.last_caret_on = true;
                            self.model_selected = 0;
                            self.model_scroll = 0;
                        }
                        Key::Named(NamedKey::Enter) => {
                            self.pick_model_index(self.model_selected);
                            return;
                        }
                        Key::Character(c) if ctrl && c.eq_ignore_ascii_case("v") => {
                            if let Some(clip) = paste_from_clipboard() {
                                self.model_filter.push_str(clip.trim());
                                self.model_search_focused = true;
                                self.caret_blink_at = Instant::now();
                                self.last_caret_on = true;
                                self.model_selected = 0;
                                self.model_scroll = 0;
                            }
                        }
                        Key::Character(c) if ctrl && c.eq_ignore_ascii_case("u") => {
                            self.model_filter.clear();
                            self.model_search_focused = true;
                            self.caret_blink_at = Instant::now();
                            self.last_caret_on = true;
                            self.model_selected = 0;
                            self.model_scroll = 0;
                        }
                        Key::Character(c) if !ctrl => {
                            self.model_filter.push_str(c.as_str());
                            self.model_search_focused = true;
                            self.caret_blink_at = Instant::now();
                            self.last_caret_on = true;
                            self.model_selected = 0;
                            self.model_scroll = 0;
                        }
                        _ => {}
                    }
                    self.request_redraws();
                    return;
                }

                // Dictate & Settings shortcuts
                match &event.logical_key {
                    Key::Named(NamedKey::Space) => self.toggle_recording(None, false),
                    Key::Named(NamedKey::Escape) => {
                        if self.model_dropdown_open {
                            self.model_dropdown_open = false;
                            self.request_redraws();
                        } else if let Some(rec) = self.active_recording.take() {
                            drop(rec);
                            self.phase = DictationPhase::Idle;
                            self.set_overlay_visible(false);
                            self.request_redraws();
                        } else if self.view != AppView::Dictate {
                            self.view = AppView::Dictate;
                            self.request_redraws();
                        }
                    }
                    Key::Named(NamedKey::ArrowUp) if self.view == AppView::Dictate => {
                        if self.selected_history > 0 {
                            self.selected_history -= 1;
                            if self.selected_history < self.history_scroll {
                                self.history_scroll = self.selected_history;
                            }
                            self.request_redraws();
                        }
                    }
                    Key::Named(NamedKey::ArrowDown) if self.view == AppView::Dictate => {
                        if self.selected_history + 1 < self.history.len() {
                            self.selected_history += 1;
                            if self.selected_history >= self.history_scroll + 4 {
                                self.history_scroll = self.selected_history.saturating_sub(3);
                            }
                            self.request_redraws();
                        }
                    }
                    Key::Named(NamedKey::Enter) if self.view == AppView::Dictate => {
                        self.execute_hit(event_loop, HitTarget::CopyHistory(self.selected_history));
                    }
                    _ => {}
                }
            }
            WindowEvent::RedrawRequested => {
                self.process_background_events();
                if let (Some(ctx), Some(win), Some(renderer)) = (
                    &self.gpu_ctx,
                    &self.main_window,
                    &mut self.main_renderer,
                ) {
                    let s = win.scale_factor() as f32;
                    let size = win.inner_size();
                    let logical_w = size.width as f32 / s;
                    let logical_h = size.height as f32 / s;

                    let raw_t = (self.hover_started_at.elapsed().as_secs_f32() / 0.11).clamp(0.0, 1.0);
                    let hover_t = 1.0 - (1.0 - raw_t).powi(3); // cubic ease-out

                    self.main_draw.clear(s);
                    let snap = UiSnapshot {
                        width: logical_w,
                        height: logical_h,
                        view: self.view,
                        model_tab: self.model_tab,
                        model_filter: &self.model_filter,
                        model_search_focused: self.model_search_focused,
                        model_dropdown_open: self.model_dropdown_open,
                        caret_visible: self.last_caret_on,
                        model_selected: self.model_selected,
                        model_scroll: self.model_scroll,
                        config: &self.config,
                        paths: &self.paths,
                        phase: &self.phase,
                        elapsed_secs: self.elapsed_secs,
                        history_bars: &self.history_bars,
                        vad_speaking: self.vad_speaking,
                        live_preview: &self.live_preview,
                        history: &self.history,
                        selected_history: self.selected_history,
                        history_scroll: self.history_scroll,
                        status_toast: self.status_toast.as_deref(),
                        modal: &self.modal,
                        active_mic_name: &self.active_mic_name,
                        mouse_pos: self.mouse_pos,
                        hover_t,
                        pressed_target: self.pressed_target.as_ref(),
                        dragging_scrollbar: self.dragging_scrollbar.map(|(l, ..)| l),
                    };
                    self.hits =
                        render_main_window(&mut self.main_draw, &mut self.main_text, &snap);
                    let _ = renderer.render(
                        ctx,
                        palette().bg,
                        &self.main_draw,
                        &mut self.main_text,
                    );
                }
            }
            _ => {}
        }
    }
}

fn edge_resize_direction(x: f32, y: f32, w: f32, h: f32) -> Option<ResizeDirection> {
    let margin = 6.0;
    let corner = 12.0;
    let left = x <= margin;
    let right = x >= w - margin;
    let top = y <= margin;
    let bottom = y >= h - margin;

    if (left && y <= corner) || (top && x <= corner) {
        return Some(ResizeDirection::NorthWest);
    }
    if (right && y <= corner) || (top && x >= w - corner) {
        return Some(ResizeDirection::NorthEast);
    }
    if (left && y >= h - corner) || (bottom && x <= corner) {
        return Some(ResizeDirection::SouthWest);
    }
    if (right && y >= h - corner) || (bottom && x >= w - corner) {
        return Some(ResizeDirection::SouthEast);
    }
    if left {
        return Some(ResizeDirection::West);
    }
    if right {
        return Some(ResizeDirection::East);
    }
    if top {
        return Some(ResizeDirection::North);
    }
    if bottom {
        return Some(ResizeDirection::South);
    }
    None
}

fn resize_cursor(dir: ResizeDirection) -> CursorIcon {
    match dir {
        ResizeDirection::East | ResizeDirection::West => CursorIcon::EwResize,
        ResizeDirection::North | ResizeDirection::South => CursorIcon::NsResize,
        ResizeDirection::NorthWest | ResizeDirection::SouthEast => CursorIcon::NwseResize,
        ResizeDirection::NorthEast | ResizeDirection::SouthWest => CursorIcon::NeswResize,
    }
}
