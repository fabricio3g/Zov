mod audio;
mod catalog;
mod config;
mod engine;
mod gpu_ui;
mod icon_motion;
mod inject;
mod style;
mod tray;
mod ui;
mod vad;
mod wav;

use crate::audio::{list_input_devices, start_recording, ActiveRecording, CompletedRecording};
use crate::catalog::{
    find_commercial_provider, find_local_whisper_spec, UnifiedSttChoice, COMMERCIAL_STT_PROVIDERS,
    COMMERCIAL_TTS_CATALOG, LOCAL_WHISPER_CATALOG,
};
use crate::config::{load_history, save_history, AppPaths, DictateConfig, HistoryEntry, HotkeyMode};
use crate::engine::{
    check_local_whisper_model, discard_download, has_nvidia_gpu, install_controlled,
    installed_local_models, is_local_engine_ready, resolve_audiocpp_binary, resolve_whisper_binary,
    setup_local_whisper, stop_whisper_server, transcribe_wav, DownloadCommand, DownloadOutcome,
    InstallStatus, InstallTarget, LocalInventory, TranscriptionOutput,
};
use crate::gpu_ui::{DrawList, GpuContext, GpuWindowRenderer, TextEngine};
use crate::inject::{
    configure_overlay_window_no_activate, copy_to_clipboard, deliver_transcript, is_native_window,
    paste_from_clipboard, set_overlay_visible_no_activate, spawn_global_hotkey_listener,
    GlobalHotkeyEvent,
};
use crate::style::{palette, set_theme, toggle_theme, window_border_color, Rgba, Theme};
use crate::tray::{create_winit_window_icon, spawn_tray_icon, TrayEvent};
use crate::ui::{
    filtered_stt_choices, hit_test, render_main_window, render_overlay_pill, AppView, ComposerMenu,
    DictationPhase, DownloadJob, DownloadState, HitTarget, HitZone, Modal, ModelFilterTab,
    ProviderField, ScrollList, UiSnapshot,
};
use crate::vad::HISTORY_BARS;
use crate::wav::load_wav_normalized;
use anyhow::Result;
use chrono::Utc;
use clap::{Parser, Subcommand};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::{CursorIcon, ResizeDirection, Window, WindowId, WindowLevel};

const APP_NAME: &str = "Zov";
const FRAME_INTERVAL: Duration = Duration::from_nanos(16_666_667);
const CARET_INTERVAL_MS: u128 = 530;

fn hover_progress(elapsed: Duration) -> f32 {
    let progress = (elapsed.as_secs_f32() / 0.11).min(1.0);
    1.0 - (1.0 - progress).powi(3)
}

struct WakeSender<T> {
    sender: mpsc::Sender<T>,
    event_loop: EventLoopProxy<()>,
}

impl<T> WakeSender<T> {
    fn send(&self, message: T) -> Result<(), mpsc::SendError<T>> {
        self.sender.send(message)?;
        let _ = self.event_loop.send_event(());
        Ok(())
    }
}

fn event_channel<T>(event_loop: EventLoopProxy<()>) -> (Arc<WakeSender<T>>, mpsc::Receiver<T>) {
    let (sender, receiver) = mpsc::channel();
    (Arc::new(WakeSender { sender, event_loop }), receiver)
}

#[derive(Parser)]
#[command(
    name = "enio",
    version,
    about = "Zov — Native desktop dictation with on-device and cloud transcription"
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
    InstallProgress(InstallStatus),
    InstallDone {
        target: InstallTarget,
        result: Result<DownloadOutcome, String>,
        previous_selection: (String, String, String),
    },
    TranscriptionDone {
        session_id: uuid::Uuid,
        duration_secs: f32,
        is_realtime_clause: bool,
        result: Result<TranscriptionOutput, String>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CaptureSource {
    Window,
    Tray,
    Hotkey(isize),
}

impl CaptureSource {
    fn release_stops(self, mode: HotkeyMode, held_ms: u64) -> bool {
        matches!(self, Self::Hotkey(_))
            && (mode == HotkeyMode::HoldToTalk
                || (mode == HotkeyMode::SmartHoldOrTap && held_ms >= 350))
    }
}

#[cfg(test)]
mod capture_tests {
    use super::*;

    #[test]
    fn only_hotkey_started_recordings_stop_on_hotkey_release() {
        assert!(!CaptureSource::Window.release_stops(HotkeyMode::HoldToTalk, 550));
        assert!(!CaptureSource::Tray.release_stops(HotkeyMode::SmartHoldOrTap, 550));
        let hotkey = CaptureSource::Hotkey(1);
        assert!(hotkey.release_stops(HotkeyMode::HoldToTalk, 80));
        assert!(!hotkey.release_stops(HotkeyMode::SmartHoldOrTap, 349));
        assert!(hotkey.release_stops(HotkeyMode::SmartHoldOrTap, 350));
        assert!(!hotkey.release_stops(HotkeyMode::RealTimeStreaming, 550));
    }
}

struct DictationSession {
    id: uuid::Uuid,
    config: DictateConfig,
    source: CaptureSource,
    queue: VecDeque<(CompletedRecording, bool)>,
    error: Option<String>,
}

struct Transition<T> {
    key: Option<T>,
    since: Option<Instant>,
}

impl<T> Default for Transition<T> {
    fn default() -> Self {
        Self {
            key: None,
            since: None,
        }
    }
}

impl<T: PartialEq> Transition<T> {
    fn update(&mut self, key: T, now: Instant) {
        if self.key.as_ref() != Some(&key) {
            self.since = self.key.as_ref().map(|_| now);
            self.key = Some(key);
        }
    }

    fn value(&self, now: Instant, duration_ms: f32) -> f32 {
        let t = self.since.map_or(1.0, |since| {
            (now.saturating_duration_since(since).as_secs_f32() * 1000.0 / duration_ms).min(1.0)
        });
        1.0 - (1.0 - t).powi(3)
    }
}

struct UiTransitions {
    enabled: bool,
    clock: Instant,
    page: Transition<AppView>,
    popup: Transition<Option<ComposerMenu>>,
    modal: Transition<std::mem::Discriminant<Modal>>,
    dictation: Transition<(std::mem::Discriminant<DictationPhase>, bool)>,
    download: Transition<f32>,
    download_from: f32,
}

impl UiTransitions {
    fn new() -> Self {
        Self {
            enabled: interface_animations_enabled(),
            clock: Instant::now(),
            page: Transition::default(),
            popup: Transition::default(),
            modal: Transition::default(),
            dictation: Transition::default(),
            download: Transition::default(),
            download_from: 0.0,
        }
    }

    fn animating(&self, now: Instant) -> bool {
        self.enabled
            && [
                self.page.since,
                self.popup.since,
                self.modal.since,
                self.dictation.since,
                self.download.since,
            ]
            .into_iter()
            .flatten()
            .any(|started| now.saturating_duration_since(started) < Duration::from_millis(255))
    }

    fn update_download(&mut self, fraction: Option<f32>, now: Instant) -> Option<f32> {
        let Some(fraction) = fraction else {
            self.download = Transition::default();
            self.download_from = 0.0;
            return None;
        };
        let previous = self.download.key.unwrap_or(fraction);
        if fraction < previous {
            self.download = Transition::default();
        }
        if self.download.key != Some(fraction) {
            self.download_from = self.download_from
                + (previous - self.download_from) * self.download.value(now, 200.0);
            self.download.update(fraction, now);
        }
        Some(if self.enabled {
            self.download_from + (fraction - self.download_from) * self.download.value(now, 200.0)
        } else {
            fraction
        })
    }
}

#[cfg(windows)]
fn interface_animations_enabled() -> bool {
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        SystemParametersInfoW, SPI_GETCLIENTAREAANIMATION,
    };
    let mut enabled = 1i32;
    unsafe {
        SystemParametersInfoW(
            SPI_GETCLIENTAREAANIMATION,
            0,
            (&mut enabled as *mut i32).cast(),
            0,
        );
    }
    enabled != 0
}

#[cfg(not(windows))]
fn interface_animations_enabled() -> bool {
    true
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
            rt.block_on(setup_local_whisper(&paths, &cfg, &model, |status| {
                if let Some(progress) = status.transfer() {
                    println!("{} · {}", status.label(), progress.label());
                } else {
                    println!("{}", status.label());
                }
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
    let mut app =
        DictateDesktopApp::new(paths, cfg, rt.handle().clone(), event_loop.create_proxy());
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
    println!("{APP_NAME} — System Status");
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

struct ModelChoices {
    tab: ModelFilterTab,
    filter: String,
    backend: String,
    local_model: String,
    commercial_model: String,
    rows: Vec<UnifiedSttChoice>,
}

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
    overlay_hide_at: Option<Instant>,
    // Navigation & UI state
    view: AppView,
    model_tab: ModelFilterTab,
    model_filter: String,
    model_search_focused: bool,
    composer_menu: Option<ComposerMenu>,
    caret_blink_at: Instant,
    last_caret_on: bool,
    model_selected: usize,
    model_scroll: usize,
    inventory: LocalInventory,
    model_choices: Option<ModelChoices>,
    settings_scroll: usize,
    hits: Vec<HitZone>,
    mouse_pos: (f32, f32),
    last_hovered: Option<HitTarget>,
    leaving_hover: Option<(HitTarget, f32)>,
    hover_started_at: Instant,
    transitions: UiTransitions,
    next_frame_at: Instant,
    pressed_target: Option<HitTarget>,
    dragging_scrollbar: Option<(ScrollList, f32, f32, usize, usize)>,
    last_cursor: CursorIcon,
    last_titlebar_click: Option<Instant>,
    modifiers: ModifiersState,
    // Background workers, system tray & global hotkey
    hotkey_rx: mpsc::Receiver<GlobalHotkeyEvent>,
    tray_rx: mpsc::Receiver<TrayEvent>,
    worker_tx: Arc<WakeSender<WorkerMessage>>,
    worker_rx: mpsc::Receiver<WorkerMessage>,
    download: Option<DownloadJob>,
    recent_downloads: Vec<InstallTarget>,
    download_command: Option<tokio::sync::watch::Sender<DownloadCommand>>,
    available_mics: Vec<String>,
    should_exit: bool,
    history: Vec<HistoryEntry>,
    selected_history: usize,
    history_scroll: usize,
    phase: DictationPhase,
    modal: Modal,
    status_toast: Option<String>,
    active_recording: Option<ActiveRecording>,
    session: Option<DictationSession>,
    pending_transcription: Option<uuid::Uuid>,
    live_preview: String,
    audio_levels: [f32; 15],
    history_bars: Vec<f32>,
    vad_speaking: bool,
    elapsed_secs: u64,
}

impl DictateDesktopApp {
    fn new(
        paths: AppPaths,
        config: DictateConfig,
        rt_handle: tokio::runtime::Handle,
        wake: EventLoopProxy<()>,
    ) -> Self {
        let hotkey_rx = spawn_global_hotkey_listener(wake.clone());
        let tray_rx = spawn_tray_icon(wake.clone());
        let (worker_tx, worker_rx) = event_channel(wake);
        let history = load_history(&paths);
        let available_mics = list_input_devices();
        let inventory = LocalInventory::scan(&paths, &config);
        let main_text = TextEngine::new();
        let overlay_text = main_text.for_other_window();
        Self {
            paths,
            config,
            rt_handle,
            gpu_ctx: None,
            main_window: None,
            main_renderer: None,
            main_draw: DrawList::new(1.0),
            main_text,
            overlay_window: None,
            overlay_renderer: None,
            overlay_draw: DrawList::new(1.0),
            overlay_text,
            overlay_visible: false,
            overlay_hide_at: None,
            view: AppView::Dictate,
            model_tab: ModelFilterTab::All,
            model_filter: String::new(),
            model_search_focused: false,
            composer_menu: None,
            caret_blink_at: Instant::now(),
            last_caret_on: true,
            model_selected: 0,
            model_scroll: 0,
            inventory,
            model_choices: None,
            settings_scroll: 0,
            hits: Vec::new(),
            mouse_pos: (-1.0, -1.0),
            last_hovered: None,
            leaving_hover: None,
            hover_started_at: Instant::now(),
            transitions: UiTransitions::new(),
            next_frame_at: Instant::now(),
            pressed_target: None,
            dragging_scrollbar: None,
            last_cursor: CursorIcon::Default,
            last_titlebar_click: None,
            modifiers: ModifiersState::default(),
            hotkey_rx,
            tray_rx,
            worker_tx,
            worker_rx,
            download: None,
            recent_downloads: Vec::new(),
            download_command: None,
            available_mics,
            should_exit: false,
            history,
            selected_history: 0,
            history_scroll: 0,
            phase: DictationPhase::Idle,
            modal: Modal::None,
            status_toast: None,
            active_recording: None,
            session: None,
            pending_transcription: None,
            live_preview: String::new(),
            audio_levels: [0.0; 15],
            history_bars: vec![0.0; HISTORY_BARS],
            vad_speaking: false,
            elapsed_secs: 0,
        }
    }

    fn refresh_inventory(&mut self) {
        let inventory = LocalInventory::scan(&self.paths, &self.config);
        if inventory != self.inventory {
            self.inventory = inventory;
            self.model_choices = None;
        }
    }

    fn sync_model_choices(&mut self) {
        if self.model_choices.as_ref().is_some_and(|cache| {
            cache.tab == self.model_tab
                && cache.filter == self.model_filter
                && cache.backend == self.config.backend
                && cache.local_model == self.config.local_model
                && cache.commercial_model == self.config.commercial_model
        }) {
            return;
        }
        self.model_choices = Some(ModelChoices {
            tab: self.model_tab,
            filter: self.model_filter.clone(),
            backend: self.config.backend.clone(),
            local_model: self.config.local_model.clone(),
            commercial_model: self.config.commercial_model.clone(),
            rows: filtered_stt_choices(
                &self.inventory,
                &self.config,
                self.model_tab,
                &self.model_filter,
            ),
        });
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
        let thumb_h = (track_h * (visible as f32 / total as f32))
            .max(28.0)
            .min(track_h);
        let usable = (track_h - thumb_h).max(1.0);
        let ratio = ((mouse_y - track_y - thumb_h * 0.5) / usable).clamp(0.0, 1.0);
        let next = (ratio * max_scroll as f32).round() as usize;
        match list {
            ScrollList::Models => self.model_scroll = next.min(max_scroll),
            ScrollList::History => self.history_scroll = next.min(max_scroll),
            ScrollList::Settings => self.settings_scroll = next,
            ScrollList::Transcript => {
                if let Modal::Transcript { scroll, .. } = &mut self.modal {
                    *scroll = next;
                }
            }
        }
    }

    fn scroll_limit(hits: &[HitZone], list: ScrollList) -> usize {
        hits.iter()
            .find_map(|zone| match zone.target {
                HitTarget::Scrollbar {
                    list: target,
                    total,
                    visible,
                    ..
                } if target == list => Some(total.saturating_sub(visible)),
                _ => None,
            })
            .unwrap_or(0)
    }

    fn scroll_transcript(&mut self, step: isize) {
        if let Modal::Transcript { scroll, .. } = &mut self.modal {
            let limit = Self::scroll_limit(&self.hits, ScrollList::Transcript);
            *scroll = scroll.saturating_add_signed(step).min(limit);
            self.request_redraws();
        }
    }

    fn request_redraws(&self) {
        if let Some(win) = self
            .main_window
            .as_ref()
            .filter(|win| win.is_visible() != Some(false) && win.is_minimized() != Some(true))
        {
            win.request_redraw();
        }
        if self.overlay_visible {
            if let Some(ov) = &self.overlay_window {
                ov.request_redraw();
            }
        }
    }

    fn update_hover(&mut self, target: Option<HitTarget>) {
        if self.last_hovered != target {
            let previous = std::mem::replace(&mut self.last_hovered, target);
            self.leaving_hover = previous
                .map(|previous| (previous, hover_progress(self.hover_started_at.elapsed())));
            self.hover_started_at = Instant::now();
            self.request_redraws();
        }
    }

    fn set_overlay_visible(&mut self, visible: bool) {
        self.overlay_hide_at = None;
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

    fn start_local_install(&mut self, target: InstallTarget) {
        if self.download_command.is_some() {
            return;
        }
        let progress = self
            .download
            .as_ref()
            .filter(|job| job.target == target)
            .and_then(|job| job.progress);
        self.status_toast = None;
        self.download = Some(DownloadJob {
            target: target.clone(),
            state: DownloadState::Running(InstallStatus::Checking),
            progress,
            stopping: false,
        });
        let (command, commands) = tokio::sync::watch::channel(DownloadCommand::Run);
        self.download_command = Some(command);
        let tx = self.worker_tx.clone();
        let paths_clone = self.paths.clone();
        let cfg_clone = self.config.clone();
        let previous_selection = (
            cfg_clone.backend.clone(),
            cfg_clone.local_model.clone(),
            cfg_clone.commercial_model.clone(),
        );
        self.rt_handle.spawn(async move {
            let tx_p = tx.clone();
            let result =
                install_controlled(&paths_clone, &cfg_clone, &target, commands, move |status| {
                    let _ = tx_p.send(WorkerMessage::InstallProgress(status));
                })
                .await
                .map_err(|e| e.to_string());
            let _ = tx.send(WorkerMessage::InstallDone {
                target,
                result,
                previous_selection,
            });
        });
        self.request_redraws();
    }

    fn download_is_pending(&self) -> bool {
        self.download.as_ref().is_some_and(|job| {
            matches!(job.state, DownloadState::Running(_) | DownloadState::Paused)
        })
    }

    fn control_download(&mut self, command: DownloadCommand) {
        let Some(job) = &mut self.download else {
            return;
        };
        let allowed = match command {
            DownloadCommand::Pause => job.can_pause(),
            DownloadCommand::Cancel => job.can_cancel(),
            DownloadCommand::Run => false,
        };
        if !allowed {
            return;
        }
        if let Some(sender) = &self.download_command {
            job.stopping = sender.send(command).is_ok();
        } else if command == DownloadCommand::Cancel {
            job.state = match discard_download(&self.paths, &job.target) {
                Ok(()) => {
                    job.progress = None;
                    DownloadState::Canceled
                }
                Err(err) => DownloadState::Failed(err.to_string()),
            };
        }
    }

    fn dispatch_next_transcription(&mut self) {
        if self.pending_transcription.is_some() {
            return;
        }
        let Some(session) = &mut self.session else {
            return;
        };
        let Some((recording, is_realtime_clause)) = session.queue.pop_front() else {
            return;
        };
        let session_id = session.id;
        self.pending_transcription = Some(session_id);
        let tx = self.worker_tx.clone();
        let paths_clone = self.paths.clone();
        let cfg_clone = session.config.clone();
        self.rt_handle.spawn(async move {
            let res = transcribe_wav(&paths_clone, &cfg_clone, &recording.wav_bytes)
                .await
                .map_err(|e| e.to_string());
            let _ = tx.send(WorkerMessage::TranscriptionDone {
                session_id,
                duration_secs: recording.duration_secs,
                is_realtime_clause,
                result: res,
            });
        });
    }

    fn start_capture(&mut self, source: CaptureSource) {
        if self.session.is_some() {
            return;
        }

        let local = self.config.backend == "local";
        let ready = if local {
            is_local_engine_ready(&self.paths, &self.config)
        } else {
            self.config.has_api_key(&self.config.backend)
        };
        if !ready {
            if source == CaptureSource::Window {
                if local {
                    self.open_confirm_download_modal(self.config.local_model.clone());
                } else {
                    self.open_provider_config_modal(self.config.backend.clone(), None);
                }
            } else {
                self.phase = DictationPhase::SetupRequired;
                self.set_overlay_visible(true);
                self.overlay_hide_at = Some(Instant::now() + Duration::from_secs(4));
                self.request_redraws();
            }
            return;
        }

        self.status_toast = None;
        let realtime = self.config.hotkey_mode == HotkeyMode::RealTimeStreaming;
        match start_recording(
            self.config.audio_device.as_deref(),
            self.config.vad_auto_stop,
            realtime,
        ) {
            Ok(rec) => {
                self.active_recording = Some(rec);
                self.session = Some(DictationSession {
                    id: uuid::Uuid::new_v4(),
                    config: self.config.clone(),
                    source,
                    queue: VecDeque::new(),
                    error: None,
                });
                self.live_preview.clear();
                self.phase = DictationPhase::Recording;
                self.set_overlay_visible(source != CaptureSource::Window);
            }
            Err(err) => {
                self.phase = DictationPhase::Error(format!("Microphone unavailable. {err}"));
                if source != CaptureSource::Window {
                    self.set_overlay_visible(true);
                    self.overlay_hide_at = Some(Instant::now() + Duration::from_secs(4));
                }
            }
        }
        self.request_redraws();
    }

    fn stop_capture_and_transcribe(&mut self) {
        if let Some(mut rec) = self.active_recording.take() {
            let session = self.session.as_mut().expect("recording has a session");
            rec.stop_capture();
            while let Some(clause) = rec.take_completed_clause() {
                session.queue.push_back((clause, true));
            }
            if let Some(done) = rec.finish() {
                session.queue.push_back((done, false));
            }
            self.phase = DictationPhase::Transcribing;
            self.dispatch_next_transcription();
            self.finish_session_if_drained();
            self.request_redraws();
        }
    }

    fn finish_session_if_drained(&mut self) {
        if self.active_recording.is_some()
            || self.session.as_ref().is_none_or(|session| {
                self.pending_transcription == Some(session.id) || !session.queue.is_empty()
            })
        {
            return;
        }
        let error = self.session.take().and_then(|session| session.error);
        self.phase = error.map_or(DictationPhase::Idle, DictationPhase::Error);
        if self.live_preview.is_empty() && matches!(self.phase, DictationPhase::Idle) {
            self.status_toast =
                Some("No speech captured. Try speaking closer to the microphone.".into());
        }
        if self.overlay_visible {
            let seconds = if matches!(self.phase, DictationPhase::Error(_)) {
                4
            } else {
                3
            };
            self.overlay_hide_at = Some(Instant::now() + Duration::from_secs(seconds));
        }
    }

    fn cancel_recording(&mut self) {
        self.active_recording = None;
        self.session = None;
        self.phase = DictationPhase::Idle;
        let message = if self.live_preview.is_empty() {
            "Dictation canceled"
        } else {
            "Dictation stopped. Previously transcribed phrases are in history."
        };
        self.status_toast = Some(message.into());
        self.set_overlay_visible(false);
        self.request_redraws();
    }

    fn toggle_recording(&mut self, source: CaptureSource) {
        if self.active_recording.is_some() {
            self.stop_capture_and_transcribe();
        } else {
            self.start_capture(source);
        }
    }

    fn open_provider_config_modal(&mut self, provider_id: String, select_model: Option<String>) {
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
        let model_id = select_model.clone().unwrap_or_else(|| {
            self.config
                .resolve_commercial_auth(&provider_id)
                .map(|(_, _, model)| model)
                .unwrap_or_default()
        });
        self.composer_menu = None;
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
            select_model,
        };
        self.request_redraws();
    }

    fn open_confirm_download_modal(&mut self, model_id: String) {
        self.open_install_confirmation(InstallTarget::Model(model_id));
    }

    fn open_install_confirmation(&mut self, target: InstallTarget) {
        if self.download_is_pending() {
            if self
                .download
                .as_ref()
                .is_some_and(|job| job.target != target)
            {
                self.status_toast = Some("Finish or cancel the current download first.".into());
            }
            self.composer_menu = None;
            self.modal = Modal::Downloads;
            self.request_redraws();
            return;
        }
        if target
            .model_id()
            .is_some_and(|id| find_local_whisper_spec(id).is_none())
        {
            self.status_toast = Some("Select a supported speech model from Models.".into());
            self.view = AppView::Models;
            self.request_redraws();
            return;
        }
        self.refresh_inventory();
        if let InstallTarget::Engine(engine) = target {
            if self.inventory.has_engine(engine) {
                self.status_toast = Some(format!("{} is already installed.", engine.label()));
                self.request_redraws();
                return;
            }
        }
        self.composer_menu = None;
        self.modal = Modal::ConfirmDownload(target);
        self.request_redraws();
    }

    fn pick_model_index(&mut self, index: usize) {
        self.sync_model_choices();
        let Some(choice) = self
            .model_choices
            .as_ref()
            .and_then(|cache| cache.rows.get(index))
            .cloned()
        else {
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
                self.status_toast = Some(format!("Switched to {}", choice.model_id));
                self.view = AppView::Dictate;
            }
        } else if self.config.has_api_key(&choice.backend) {
            self.config.backend = choice.backend;
            self.config.commercial_model = choice.model_id.clone();
            self.config.save(&self.paths);
            self.status_toast = Some(format!("Switched to {}", choice.model_id));
            self.view = AppView::Dictate;
        } else {
            self.open_provider_config_modal(choice.backend, Some(choice.model_id));
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
                if v == AppView::Models {
                    self.refresh_inventory();
                }
                self.view = v;
                self.composer_menu = None;
                if self.modal == Modal::Downloads {
                    self.modal = Modal::None;
                }
            }
            HitTarget::ToggleRecord => {
                self.composer_menu = None;
                self.toggle_recording(CaptureSource::Window);
            }
            HitTarget::CancelRecording => {
                if self.session.is_some() {
                    self.cancel_recording();
                }
            }
            HitTarget::ToggleComposerMenu(menu) => {
                self.composer_menu = (self.composer_menu != Some(menu)).then_some(menu);
            }
            HitTarget::CloseComposerMenu => {
                self.composer_menu = None;
            }
            HitTarget::SelectDropdownLocal(model_id) => {
                self.composer_menu = None;
                self.config.backend = "local".to_string();
                self.config.local_model = model_id.clone();
                self.config.save(&self.paths);
                self.status_toast = Some(format!("Switched to {model_id}"));
            }
            HitTarget::SelectDropdownCloud { backend, model_id } => {
                self.composer_menu = None;
                self.config.backend = backend;
                self.config.commercial_model = model_id.clone();
                self.config.save(&self.paths);
                self.status_toast = Some(format!("Switched to {model_id}"));
            }
            HitTarget::DownloadLocalModel(id) => self.open_confirm_download_modal(id),
            HitTarget::InstallEngine(engine) => {
                self.open_install_confirmation(InstallTarget::Engine(engine))
            }
            HitTarget::OpenEngineSettings => {
                self.refresh_inventory();
                self.settings_scroll = 0;
                self.composer_menu = None;
                self.modal = Modal::None;
                self.view = AppView::Settings;
            }
            HitTarget::ToggleDownloads => {
                self.composer_menu = None;
                self.modal = if self.modal == Modal::Downloads {
                    Modal::None
                } else {
                    Modal::Downloads
                };
            }
            HitTarget::PauseDownload => self.control_download(DownloadCommand::Pause),
            HitTarget::CancelDownload => self.control_download(DownloadCommand::Cancel),
            HitTarget::ResumeDownload => {
                if let Some(job) = self.download.as_ref().filter(|job| {
                    matches!(job.state, DownloadState::Paused | DownloadState::Failed(_))
                }) {
                    self.start_local_install(job.target.clone());
                }
            }
            HitTarget::CycleLanguage | HitTarget::SetLanguage(_) => {
                if let HitTarget::SetLanguage(code) = target {
                    self.config.language = code.into();
                    self.composer_menu = None;
                } else {
                    self.config.cycle_language();
                }
                self.config.save(&self.paths);
                self.status_toast = Some(format!(
                    "Language · {}",
                    DictateConfig::language_label(&self.config.language)
                ));
            }
            HitTarget::CycleHotkeyMode => {
                self.config.hotkey_mode = self.config.hotkey_mode.next();
                self.config.save(&self.paths);
                self.status_toast = Some(self.config.hotkey_mode.label().to_string());
            }
            HitTarget::RefreshMicrophones => {
                self.available_mics = list_input_devices();
                if self.available_mics.is_empty() {
                    self.status_toast = Some(
                        "No microphone found — check Windows Microphone privacy settings"
                            .to_string(),
                    );
                } else {
                    self.status_toast =
                        Some(format!("Found {} microphone(s)", self.available_mics.len()));
                }
            }
            HitTarget::CycleMicrophone => {
                if self.available_mics.is_empty() {
                    self.available_mics = list_input_devices();
                }
                if !self.available_mics.is_empty() {
                    let current = self.config.audio_device.as_ref().and_then(|device| {
                        self.available_mics.iter().position(|mic| mic == device)
                    });
                    let next = current.map_or(0, |index| index + 1);
                    self.config.audio_device = self.available_mics.get(next).cloned();
                    self.config.save(&self.paths);
                    let chosen = self
                        .config
                        .audio_device
                        .as_deref()
                        .unwrap_or("System default");
                    self.status_toast = Some(format!("Microphone · {chosen}"));
                } else {
                    self.status_toast = Some("No microphones detected".to_string());
                }
            }
            HitTarget::ToggleVad => {
                self.config.vad_auto_stop = !self.config.vad_auto_stop;
                self.config.save(&self.paths);
            }
            HitTarget::CycleDelivery | HitTarget::SetDelivery(_) => {
                self.config.delivery_mode = if let HitTarget::SetDelivery(mode) = target {
                    self.composer_menu = None;
                    mode
                } else {
                    self.config.delivery_mode.next()
                };
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
            HitTarget::OpenProviderConfig(pid) => self.open_provider_config_modal(pid, None),
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
                self.status_toast = Some(format!("Switched to local model: {model_id}"));
            }
            HitTarget::OpenHistory(id) => {
                if let Some(idx) = self.history.iter().position(|entry| entry.id == id) {
                    self.selected_history = idx;
                    self.modal = Modal::Transcript {
                        id,
                        scroll: 0,
                        copied: false,
                    };
                }
            }
            HitTarget::CopyHistory(id) => {
                if let Some(idx) = self.history.iter().position(|entry| entry.id == id) {
                    self.selected_history = idx;
                    let result = copy_to_clipboard(&self.history[idx].text);
                    if let Modal::Transcript { copied, .. } = &mut self.modal {
                        *copied = result.is_ok();
                        if result.is_err() {
                            self.modal = Modal::None;
                        }
                    }
                    self.status_toast = Some(
                        result
                            .map_or_else(|err| err.to_string(), |()| "Copied to clipboard".into()),
                    );
                }
            }
            HitTarget::RequestDeleteHistory(id) => {
                if let Some(entry) = self.history.iter().find(|entry| entry.id == id) {
                    let preview = entry.text.trim().to_string();
                    self.modal = Modal::ConfirmDeleteClip { id, preview };
                }
            }
            HitTarget::SaveProviderConfig => {
                if let Modal::ProviderConfig {
                    provider_id,
                    provider_name,
                    base_url,
                    model_id,
                    api_key,
                    select_model,
                    ..
                } = &self.modal
                {
                    let pid = provider_id.clone();
                    let pname = provider_name.clone();
                    if let Some(selected) = select_model {
                        self.config.backend = pid.clone();
                        self.config.commercial_model = selected.clone();
                        self.view = AppView::Dictate;
                    }
                    self.config.set_provider_config(
                        &pid,
                        api_key.clone(),
                        base_url.clone(),
                        model_id.clone(),
                    );
                    self.model_choices = None;
                    self.config.save(&self.paths);
                    self.status_toast = Some(format!("Saved {pname} configuration"));
                    self.modal = Modal::None;
                }
            }
            HitTarget::ConfirmInstall => {
                if let Modal::ConfirmDownload(target) = &self.modal {
                    let target = target.clone();
                    if self.download_is_pending() {
                        self.modal = Modal::Downloads;
                    } else {
                        self.modal = Modal::None;
                        self.start_local_install(target);
                    }
                }
            }
            HitTarget::ConfirmDeleteClip => {
                if let Modal::ConfirmDeleteClip { id, .. } = &self.modal {
                    if let Some(idx) = self.history.iter().position(|entry| &entry.id == id) {
                        self.history.remove(idx);
                        self.main_text.clear_layouts();
                        if idx < self.selected_history {
                            self.selected_history -= 1;
                        }
                        if self.selected_history >= self.history.len() && self.selected_history > 0
                        {
                            self.selected_history -= 1;
                        }
                        self.history_scroll = self
                            .history_scroll
                            .min(self.history.len().saturating_sub(1));
                        save_history(&self.paths, &self.history);
                        self.status_toast = Some("Deleted clip".to_string());
                    }
                    self.modal = Modal::None;
                }
            }
            HitTarget::CloseModal => {
                self.modal = Modal::None;
            }
            HitTarget::KeepModalOpen => {}
        }
        self.request_redraws();
    }

    fn process_background_events(&mut self, sample_audio: bool) -> bool {
        let mut changed = false;
        if self
            .overlay_hide_at
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            self.set_overlay_visible(false);
            changed = true;
        }

        let mut vad_auto_triggered = false;
        if sample_audio {
            if let Some(rec) = self.active_recording.as_ref() {
                let (elapsed, levels, speaking) = rec.snapshot_waveform(&mut self.history_bars);
                self.elapsed_secs = elapsed;
                self.audio_levels = levels;
                self.vad_speaking = speaking;
                if let Some(session) = &mut self.session {
                    while let Some(clause) = rec.take_completed_clause() {
                        session.queue.push_back((clause, true));
                    }
                }
                vad_auto_triggered = rec.should_stop.load(Ordering::Relaxed);
            }
        }
        if self.active_recording.is_none() {
            self.audio_levels = [0.0; 15];
            self.vad_speaking = false;
            self.elapsed_secs = 0;
        }

        self.dispatch_next_transcription();

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
                    self.toggle_recording(CaptureSource::Tray);
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
                    self.toggle_recording(CaptureSource::Hotkey(target_hwnd));
                }
                GlobalHotkeyEvent::Released { held_ms } => {
                    if self.session.as_ref().is_some_and(|session| {
                        session
                            .source
                            .release_stops(session.config.hotkey_mode, held_ms)
                    }) {
                        self.stop_capture_and_transcribe();
                    }
                }
            }
        }

        while let Ok(msg) = self.worker_rx.try_recv() {
            changed = true;
            match msg {
                WorkerMessage::InstallProgress(status) => {
                    if let Some(job) = &mut self.download {
                        job.state = DownloadState::Running(status);
                        if let Some(progress) = status.transfer() {
                            job.progress = Some(progress);
                        }
                    }
                }
                WorkerMessage::InstallDone {
                    target,
                    result,
                    previous_selection,
                } => {
                    self.download_command = None;
                    self.refresh_inventory();
                    let Some(job) = &mut self.download else {
                        continue;
                    };
                    job.stopping = false;
                    job.state = match result {
                        Ok(DownloadOutcome::Complete) => {
                            let (backend, local_model, commercial_model) = previous_selection;
                            if let InstallTarget::Model(model_id) = &target {
                                if self.config.backend == backend
                                    && self.config.local_model == local_model
                                    && self.config.commercial_model == commercial_model
                                {
                                    self.config.backend = "local".into();
                                    self.config.local_model = model_id.clone();
                                    self.config.save(&self.paths);
                                }
                            }
                            self.recent_downloads.retain(|previous| previous != &target);
                            self.recent_downloads.insert(0, target);
                            self.recent_downloads.truncate(3);
                            DownloadState::Completed
                        }
                        Ok(DownloadOutcome::Paused) => DownloadState::Paused,
                        Ok(DownloadOutcome::Canceled) => {
                            job.progress = None;
                            DownloadState::Canceled
                        }
                        Err(err) => DownloadState::Failed(err),
                    };
                }
                WorkerMessage::TranscriptionDone {
                    session_id,
                    duration_secs,
                    is_realtime_clause,
                    result,
                } => {
                    self.pending_transcription = None;
                    let Some(session) = self
                        .session
                        .as_mut()
                        .filter(|session| session.id == session_id)
                    else {
                        self.dispatch_next_transcription();
                        continue;
                    };
                    match result {
                        Ok(out) if !out.text.trim().is_empty() => {
                            let target_hwnd = match session.source {
                                CaptureSource::Hotkey(hwnd)
                                    if hwnd != 0
                                        && !self
                                            .main_window
                                            .as_ref()
                                            .is_some_and(|win| is_native_window(win, hwnd)) =>
                                {
                                    Some(hwnd)
                                }
                                _ => None,
                            };
                            let delivery_note = deliver_transcript(
                                &out.text,
                                session.config.delivery_mode,
                                target_hwnd,
                                is_realtime_clause,
                            )
                            .unwrap_or_else(|err| format!("Saved to history. {err}"));
                            self.live_preview = out.text.clone();
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
                            if self.history_scroll > 0 || self.modal != Modal::None {
                                self.selected_history =
                                    (self.selected_history + 1).min(self.history.len() - 1);
                                if self.history_scroll > 0 {
                                    self.history_scroll += 1;
                                }
                            } else {
                                self.selected_history = 0;
                            }
                            save_history(&self.paths, &self.history);
                            self.status_toast = Some(delivery_note);
                        }
                        Ok(_) => {}
                        Err(err) => session.error = Some(err),
                    }
                    self.dispatch_next_transcription();
                    self.finish_session_if_drained();
                }
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
            .with_title(APP_NAME)
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
            .with_title(format!("{APP_NAME} Overlay"))
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
        let now = Instant::now();
        let frame_due = now >= self.next_frame_at;
        let mut redraw = self.process_background_events(frame_due);
        if self.should_exit {
            event_loop.exit();
            return;
        }
        let main_visible = self
            .main_window
            .as_ref()
            .is_some_and(|win| win.is_visible() != Some(false) && win.is_minimized() != Some(true));
        let visual_animating = main_visible
            && self.transitions.enabled
            && (self.transitions.animating(now)
                || (self.last_hovered.is_some()
                    && self.hover_started_at.elapsed()
                        < Duration::from_millis(icon_motion::HOVER_MS))
                || (self.leaving_hover.is_some()
                    && self.hover_started_at.elapsed() < Duration::from_millis(125))
                || self.download.as_ref().is_some_and(DownloadJob::is_running));
        let needs_frames = self.active_recording.is_some()
            || visual_animating
            || (self.overlay_visible && self.phase == DictationPhase::Transcribing);
        let mut deadline = self.overlay_hide_at;
        if needs_frames {
            if frame_due {
                let next = self.next_frame_at + FRAME_INTERVAL;
                self.next_frame_at = if next > now {
                    next
                } else {
                    now + FRAME_INTERVAL
                };
                redraw = true;
            }
            deadline = Some(deadline.map_or(self.next_frame_at, |at| at.min(self.next_frame_at)));
        }
        if main_visible
            && ((self.view == AppView::Models && self.model_search_focused)
                || matches!(self.modal, Modal::ProviderConfig { .. }))
        {
            let elapsed_ms = now
                .saturating_duration_since(self.caret_blink_at)
                .as_millis();
            let caret_on = (elapsed_ms / CARET_INTERVAL_MS) % 2 == 0;
            if caret_on != self.last_caret_on {
                self.last_caret_on = caret_on;
                redraw = true;
            }
            let blink_at = now
                + Duration::from_millis(
                    (CARET_INTERVAL_MS - elapsed_ms % CARET_INTERVAL_MS) as u64,
                );
            deadline = Some(deadline.map_or(blink_at, |at| at.min(blink_at)));
        }
        if redraw {
            self.request_redraws();
        }
        event_loop.set_control_flow(deadline.map_or(ControlFlow::Wait, ControlFlow::WaitUntil));
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
                    } else if self.overlay_hide_at.is_some() {
                        self.set_overlay_visible(false);
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
                                self.session
                                    .as_ref()
                                    .map_or(&self.config, |session| &session.config),
                                self.session.as_ref().is_some_and(|session| {
                                    matches!(session.source, CaptureSource::Hotkey(_))
                                }),
                                &self.phase,
                                self.elapsed_secs,
                                self.audio_levels,
                                &self.live_preview,
                                self.status_toast.as_deref(),
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
            WindowEvent::Focused(focused) => {
                if focused {
                    self.refresh_inventory();
                }
                self.request_redraws();
            }
            WindowEvent::CursorLeft { .. } => {
                self.mouse_pos = (-1.0, -1.0);
                self.update_hover(None);
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

                self.update_hover(hovered);
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
                    if !matches!(self.modal, Modal::None) {
                        self.scroll_transcript(step * 3);
                        return;
                    }
                    if self.composer_menu.is_some() {
                        return;
                    }
                    match self.view {
                        AppView::Models => {
                            self.sync_model_choices();
                            let total = self
                                .model_choices
                                .as_ref()
                                .map_or(0, |cache| cache.rows.len());
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
                        AppView::Settings => {
                            let limit = Self::scroll_limit(&self.hits, ScrollList::Settings);
                            self.settings_scroll = self
                                .settings_scroll
                                .saturating_add_signed(step * 44)
                                .min(limit);
                        }
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
                    && !event.repeat
                    && matches!(self.modal, Modal::None | Modal::Downloads)
                    && matches!(&event.logical_key, Key::Character(c) if c.eq_ignore_ascii_case("j"))
                {
                    self.execute_hit(event_loop, HitTarget::ToggleDownloads);
                    return;
                }

                if ctrl
                    && matches!(&event.logical_key, Key::Character(c) if c.eq_ignore_ascii_case("t"))
                {
                    self.execute_hit(event_loop, HitTarget::ToggleTheme);
                    return;
                }

                // Active modal keyboard handling (Provider Config, Confirm Download, Confirm Delete)
                match &mut self.modal {
                    Modal::None => {}
                    Modal::Downloads => {
                        if event.logical_key == Key::Named(NamedKey::Escape) {
                            self.modal = Modal::None;
                            self.request_redraws();
                        }
                        return;
                    }
                    Modal::Transcript { id, .. } => {
                        match &event.logical_key {
                            Key::Named(NamedKey::Escape) => self.modal = Modal::None,
                            Key::Named(NamedKey::Enter) => {
                                let id = id.clone();
                                self.execute_hit(event_loop, HitTarget::CopyHistory(id));
                            }
                            Key::Named(NamedKey::ArrowUp) => self.scroll_transcript(-1),
                            Key::Named(NamedKey::ArrowDown) => self.scroll_transcript(1),
                            Key::Named(NamedKey::PageUp) => self.scroll_transcript(-10),
                            Key::Named(NamedKey::PageDown) => self.scroll_transcript(10),
                            _ => {}
                        }
                        self.request_redraws();
                        return;
                    }
                    Modal::ConfirmDownload(_) => {
                        match &event.logical_key {
                            Key::Named(NamedKey::Escape) => self.modal = Modal::None,
                            Key::Named(NamedKey::Enter) => {
                                self.execute_hit(event_loop, HitTarget::ConfirmInstall);
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
                    self.sync_model_choices();
                    let total = self
                        .model_choices
                        .as_ref()
                        .map_or(0, |cache| cache.rows.len());
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

                if self.composer_menu.is_some() {
                    if event.logical_key == Key::Named(NamedKey::Escape) {
                        self.composer_menu = None;
                        self.request_redraws();
                    }
                    return;
                }

                // Dictate & Settings shortcuts
                match &event.logical_key {
                    Key::Named(NamedKey::Space)
                        if self.view == AppView::Dictate
                            && !event.repeat
                            && self.modifiers.is_empty() =>
                    {
                        self.toggle_recording(CaptureSource::Window);
                    }
                    Key::Named(NamedKey::Escape) => {
                        if self.session.is_some() {
                            self.cancel_recording();
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
                            let id = &self.history[self.selected_history].id;
                            if !self.hits.iter().any(|zone| matches!(&zone.target, HitTarget::OpenHistory(visible) if visible == id)) {
                                self.history_scroll = self.selected_history;
                            }
                            self.request_redraws();
                        }
                    }
                    Key::Named(NamedKey::Enter) if self.view == AppView::Dictate => {
                        if let Some(entry) = self.history.get(self.selected_history) {
                            let target = if ctrl {
                                HitTarget::CopyHistory(entry.id.clone())
                            } else {
                                HitTarget::OpenHistory(entry.id.clone())
                            };
                            self.execute_hit(event_loop, target);
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::RedrawRequested => {
                if self.view == AppView::Models {
                    self.sync_model_choices();
                }
                if let (Some(ctx), Some(win), Some(renderer)) = (
                    &self.gpu_ctx,
                    &self.main_window,
                    &mut self.main_renderer,
                ) {
                    let s = win.scale_factor() as f32;
                    let size = win.inner_size();
                    if size.width == 0 || size.height == 0 {
                        return;
                    }
                    let logical_w = size.width as f32 / s;
                    let logical_h = size.height as f32 / s;

                    let hover_t = if self.transitions.enabled {
                        hover_progress(self.hover_started_at.elapsed())
                    } else {
                        1.0
                    };
                    let now = Instant::now();
                    self.transitions.page.update(self.view, now);
                    self.transitions.popup.update(self.composer_menu, now);
                    self.transitions
                        .modal
                        .update(std::mem::discriminant(&self.modal), now);
                    let setup_visible = self.download.as_ref().is_some_and(|job| {
                        job.target.model_id() == Some(self.config.local_model.as_str())
                            && !matches!(
                                job.state,
                                DownloadState::Completed | DownloadState::Canceled
                            )
                    });
                    self.transitions
                        .dictation
                        .update((std::mem::discriminant(&self.phase), setup_visible), now);
                    let download_fraction = self.transitions.update_download(
                        self.download
                            .as_ref()
                            .and_then(|job| job.progress)
                            .and_then(|progress| progress.fraction()),
                        now,
                    );

                    self.main_draw.clear(s);
                    let snap = UiSnapshot {
                        width: logical_w,
                        height: logical_h,
                        window_border: (!win.is_maximized() && win.fullscreen().is_none())
                            .then(|| window_border_color(win.has_focus())),
                        view: self.view,
                        model_tab: self.model_tab,
                        model_filter: &self.model_filter,
                        model_search_focused: self.model_search_focused,
                        composer_menu: self.composer_menu,
                        caret_visible: self.last_caret_on,
                        model_selected: self.model_selected,
                        model_scroll: self.model_scroll,
                        settings_scroll: self.settings_scroll,
                        config: &self.config,
                        inventory: &self.inventory,
                        model_choices: self
                            .model_choices
                            .as_ref()
                            .map_or(&[], |cache| cache.rows.as_slice()),
                        phase: &self.phase,
                        download: self.download.as_ref(),
                        recent_downloads: &self.recent_downloads,
                        download_fraction,
                        page_t: if self.transitions.enabled {
                            self.transitions.page.value(now, 180.0)
                        } else {
                            1.0
                        },
                        popup_t: if self.transitions.enabled {
                            self.transitions.popup.value(now, 160.0)
                        } else {
                            1.0
                        },
                        modal_t: if self.transitions.enabled {
                            self.transitions.modal.value(now, 220.0)
                        } else {
                            1.0
                        },
                        dictation_t: if self.transitions.enabled {
                            self.transitions.dictation.value(now, 150.0)
                        } else {
                            1.0
                        },
                        elapsed_secs: self.elapsed_secs,
                        history_bars: &self.history_bars,
                        vad_speaking: self.vad_speaking,
                        history: &self.history,
                        selected_history: self.selected_history,
                        history_scroll: self.history_scroll,
                        status_toast: self.status_toast.as_deref(),
                        modal: &self.modal,
                        active_mic_name: if self.available_mics.is_empty() {
                            "No microphone found"
                        } else {
                            self.config
                                .audio_device
                                .as_ref()
                                .filter(|device| self.available_mics.contains(device))
                                .map_or("System default", String::as_str)
                        },
                        mouse_pos: self.mouse_pos,
                        hover_t,
                        leaving_hover: self
                            .leaving_hover
                            .as_ref()
                            .map(|(target, strength)| (target, *strength * (1.0 - hover_t))),
                        icon_hover_ms: if self.transitions.enabled && self.last_hovered.is_some() {
                            self.hover_started_at.elapsed().as_secs_f32() * 1000.0
                        } else {
                            f32::INFINITY
                        },
                        spinner_angle: if self.transitions.enabled {
                            self.transitions.clock.elapsed().subsec_nanos() as f32 * 0.000_000_36
                        } else {
                            0.0
                        },
                        pressed_target: self.pressed_target.as_ref(),
                        dragging_scrollbar: self.dragging_scrollbar.map(|(l, ..)| l),
                    };
                    self.hits =
                        render_main_window(&mut self.main_draw, &mut self.main_text, &snap);
                    if self.view == AppView::Settings && self.modal == Modal::None {
                        let limit = Self::scroll_limit(&self.hits, ScrollList::Settings);
                        if self.settings_scroll > limit {
                            self.settings_scroll = limit;
                            win.request_redraw();
                        }
                    }
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
