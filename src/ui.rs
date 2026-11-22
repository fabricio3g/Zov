//! Agente Code Desktop-inspired native `wgpu` UI layout.
//!
//! - Custom frameless window chrome with draggable title bar and minimal `─ □ ×` controls
//! - 2-pane layered layout: `#171717` left sidebar (`Dictate` / `Models` / `Settings`) + `#1c1c1c` canvas
//! - 28px-radius Composer Sheet with 1px subtle ring, live waveform, and single 34px round mic control
//! - Scannable Models & Settings pages (labels hard left, values/actions hard right)
//! - Borderless always-on-top Floating Speech Overlay Pill for system-wide `Ctrl+Shift+Space` dictation.

use crate::catalog::{
    all_stt_choices, find_commercial_provider, find_local_whisper_spec, LocalEngineKind,
    UnifiedSttChoice, LOCAL_WHISPER_CATALOG,
};
use crate::config::{DeliveryMode, DictateConfig, HistoryEntry, HotkeyMode, LANGUAGES};
use crate::engine::{DownloadProgress, InstallStatus, InstallTarget, LocalInventory};
use crate::gpu_ui::{DrawList, FontWeight, IconId, Rect, TextEngine};
use crate::style::{current_theme, palette, Palette, Rgba, Theme};
use crate::vad::HISTORY_BARS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppView {
    Dictate,
    Models,
    Settings,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelFilterTab {
    Installed,
    All,
    NonWhisper,
    Whisper,
    HuggingFace,
    Commercial,
}

impl ModelFilterTab {
    pub fn label(self) -> &'static str {
        match self {
            Self::Installed => "Installed",
            Self::All => "All",
            Self::NonWhisper => "Non-Whisper",
            Self::Whisper => "Whisper",
            Self::HuggingFace => "Hugging Face",
            Self::Commercial => "Commercial",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DictationPhase {
    Idle,
    Recording,
    Transcribing,
    SetupRequired,
    Error(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DownloadState {
    Running(InstallStatus),
    Paused,
    Failed(String),
    Completed,
    Canceled,
}

pub struct DownloadJob {
    pub target: InstallTarget,
    pub state: DownloadState,
    pub progress: Option<DownloadProgress>,
    pub stopping: bool,
}

impl DownloadJob {
    pub fn is_running(&self) -> bool {
        matches!(self.state, DownloadState::Running(_)) && !self.stopping
    }

    pub fn name(&self) -> &str {
        self.target.name()
    }

    pub fn label(&self) -> &str {
        if self.stopping {
            return "Stopping…";
        }
        match &self.state {
            DownloadState::Running(stage) => stage.label(),
            DownloadState::Paused => "Paused",
            DownloadState::Failed(_) => "Download failed",
            DownloadState::Completed => "Installed",
            DownloadState::Canceled => "Canceled",
        }
    }

    pub fn can_pause(&self) -> bool {
        !self.stopping
            && matches!(
                self.state,
                DownloadState::Running(InstallStatus::DownloadingModel(_))
            )
    }

    pub fn can_cancel(&self) -> bool {
        !self.stopping
            && matches!(
                self.state,
                DownloadState::Running(
                    InstallStatus::Checking
                        | InstallStatus::DownloadingEngine(_)
                        | InstallStatus::DownloadingModel(_)
                ) | DownloadState::Paused
                    | DownloadState::Failed(_)
            )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderField {
    BaseUrl,
    ModelId,
    ApiKey,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Modal {
    None,
    Downloads,
    ProviderConfig {
        provider_id: String,
        provider_name: String,
        is_self_hosted: bool,
        base_url: String,
        model_id: String,
        api_key: String,
        active_field: ProviderField,
        reveal_key: bool,
        select_model: Option<String>,
    },
    ConfirmDownload(InstallTarget),
    ConfirmDeleteClip {
        id: String,
        preview: String,
    },
    Transcript {
        id: String,
        scroll: usize,
        copied: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScrollList {
    Models,
    History,
    Transcript,
    Settings,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComposerMenu {
    Models,
    Language,
    Delivery,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HitTarget {
    DragWindow,
    MinimizeWindow,
    MaximizeWindow,
    CloseWindow,
    SwitchView(AppView),
    ToggleRecord,
    CancelRecording,
    ToggleComposerMenu(ComposerMenu),
    CloseComposerMenu,
    SelectDropdownLocal(String),
    SelectDropdownCloud {
        backend: String,
        model_id: String,
    },
    DownloadLocalModel(String),
    InstallEngine(LocalEngineKind),
    OpenEngineSettings,
    ToggleDownloads,
    PauseDownload,
    ResumeDownload,
    CancelDownload,
    CycleLanguage,
    SetLanguage(&'static str),
    CycleHotkeyMode,
    CycleMicrophone,
    RefreshMicrophones,
    ToggleVad,
    CycleDelivery,
    SetDelivery(DeliveryMode),
    ToggleTheme,
    OpenProviderConfig(String),
    FocusProviderField(ProviderField),
    ToggleRevealKey,
    FocusModelSearch,
    ClearModelSearch,
    SetModelTab(ModelFilterTab),
    PickModel(usize),
    ActivateLocalModel(String),
    OpenHistory(String),
    CopyHistory(String),
    RequestDeleteHistory(String),
    Scrollbar {
        list: ScrollList,
        track_y: i32,
        track_h: i32,
        total: usize,
        visible: usize,
    },
    SaveProviderConfig,
    ConfirmInstall,
    ConfirmDeleteClip,
    KeepModalOpen,
    CloseModal,
}

impl HitTarget {
    pub fn is_text_input(&self) -> bool {
        matches!(self, Self::FocusModelSearch | Self::FocusProviderField(_))
    }

    pub fn is_pointer(&self) -> bool {
        !matches!(
            self,
            Self::DragWindow
                | Self::FocusModelSearch
                | Self::FocusProviderField(_)
                | Self::KeepModalOpen
                | Self::CloseComposerMenu
        )
    }
}

#[derive(Clone, Debug)]
pub struct HitZone {
    pub rect: Rect,
    pub target: HitTarget,
}

pub struct UiSnapshot<'a> {
    pub width: f32,
    pub height: f32,
    pub window_border: Option<Rgba>,
    pub view: AppView,
    pub model_tab: ModelFilterTab,
    pub model_filter: &'a str,
    pub model_search_focused: bool,
    pub composer_menu: Option<ComposerMenu>,
    pub caret_visible: bool,
    pub model_selected: usize,
    pub model_scroll: usize,
    pub settings_scroll: usize,
    pub config: &'a DictateConfig,
    pub inventory: &'a LocalInventory,
    pub model_choices: &'a [UnifiedSttChoice],
    pub phase: &'a DictationPhase,
    pub download: Option<&'a DownloadJob>,
    pub recent_downloads: &'a [InstallTarget],
    pub download_fraction: Option<f32>,
    pub page_t: f32,
    pub popup_t: f32,
    pub modal_t: f32,
    pub dictation_t: f32,
    pub elapsed_secs: u64,
    pub history_bars: &'a [f32],
    pub vad_speaking: bool,
    pub history: &'a [HistoryEntry],
    pub selected_history: usize,
    pub history_scroll: usize,
    pub status_toast: Option<&'a str>,
    pub modal: &'a Modal,
    pub active_mic_name: &'a str,
    pub mouse_pos: (f32, f32),
    pub hover_t: f32,
    pub leaving_hover: Option<(&'a HitTarget, f32)>,
    pub icon_hover_ms: f32,
    pub spinner_angle: f32,
    pub pressed_target: Option<&'a HitTarget>,
    pub dragging_scrollbar: Option<ScrollList>,
}

impl UiSnapshot<'_> {
    fn hover_level(&self, target: &HitTarget, hovered: bool) -> f32 {
        if hovered {
            self.hover_t
        } else {
            self.leaving_hover
                .filter(|(old, _)| *old == target)
                .map_or(0.0, |(_, amount)| amount)
        }
    }

    fn icon_time(&self, hovered: bool) -> f32 {
        if hovered {
            self.icon_hover_ms
        } else {
            f32::INFINITY
        }
    }
}

pub fn filtered_stt_choices(
    inventory: &LocalInventory,
    config: &DictateConfig,
    tab: ModelFilterTab,
    filter: &str,
) -> Vec<UnifiedSttChoice> {
    let q = filter.trim().to_ascii_lowercase();
    let mut list: Vec<UnifiedSttChoice> = all_stt_choices()
        .into_iter()
        .filter(|c| {
            let tab_ok = match tab {
                ModelFilterTab::Installed => {
                    if c.is_local {
                        inventory.contains(&c.model_id)
                    } else {
                        config.has_api_key(&c.backend)
                    }
                }
                ModelFilterTab::All => true,
                ModelFilterTab::NonWhisper => {
                    if c.is_local {
                        LOCAL_WHISPER_CATALOG
                            .iter()
                            .find(|m| m.id == c.model_id)
                            .is_some_and(|m| m.engine == LocalEngineKind::AudioCpp)
                    } else {
                        c.label.contains("Non-Whisper")
                            || matches!(
                                c.backend.as_str(),
                                "deepgram"
                                    | "elevenlabs"
                                    | "mistral"
                                    | "soniox"
                                    | "gemini"
                                    | "assemblyai"
                                    | "speechmatics"
                                    | "revai"
                            )
                    }
                }
                ModelFilterTab::Whisper => {
                    c.is_local
                        && LOCAL_WHISPER_CATALOG
                            .iter()
                            .find(|m| m.id == c.model_id)
                            .is_some_and(|m| m.engine == LocalEngineKind::WhisperCpp)
                }
                ModelFilterTab::HuggingFace => c.backend == "huggingface",
                ModelFilterTab::Commercial => !c.is_local && c.backend != "huggingface",
            };
            if !tab_ok {
                return false;
            }
            q.is_empty()
                || c.label.to_ascii_lowercase().contains(&q)
                || c.provider_label.to_ascii_lowercase().contains(&q)
                || c.model_id.to_ascii_lowercase().contains(&q)
                || c.backend.to_ascii_lowercase().contains(&q)
                || c.detail.to_ascii_lowercase().contains(&q)
        })
        .collect();

    // Sort already-downloaded local models to the very top so they are immediately selectable
    list.sort_by_key(|c| {
        let is_active = if c.is_local {
            config.backend == "local" && config.local_model == c.model_id
        } else {
            config.backend == c.backend && config.commercial_model == c.model_id
        };
        let is_installed = c.is_local && inventory.contains(&c.model_id);
        if is_active {
            0
        } else if is_installed {
            1
        } else if !c.is_local && config.has_api_key(&c.backend) {
            2
        } else {
            3
        }
    });
    list
}

pub fn hit_test(hits: &[HitZone], x: f32, y: f32) -> Option<HitTarget> {
    hits.iter()
        .rev()
        .find(|z| z.rect.contains(x, y))
        .map(|z| z.target.clone())
}

/// Thin scannable scrollbar matching Agente Code Desktop:
/// subtle track, brighter thumb on hover, accent thumb while dragging;
/// click jumps and drag scrubs smoothly.
#[allow(clippy::too_many_arguments)]
fn draw_scrollbar(
    dl: &mut DrawList,
    hits: &mut Vec<HitZone>,
    pal: &Palette,
    list: ScrollList,
    track_x: f32,
    track_y: f32,
    track_h: f32,
    scroll: usize,
    visible: usize,
    total: usize,
    mouse: (f32, f32),
    dragging: Option<ScrollList>,
) {
    if total <= visible || visible == 0 || track_h < 24.0 {
        return;
    }
    let hit_rect = Rect::new(track_x - 4.0, track_y, 14.0, track_h);
    let is_dragging = dragging == Some(list);
    let is_hovered = hit_rect.contains(mouse.0, mouse.1);

    let visual_w = if is_dragging || is_hovered { 6.0 } else { 4.5 };
    let visual_x = track_x + (6.0 - visual_w) * 0.5;
    let track_rect = Rect::new(visual_x, track_y, visual_w, track_h);

    if is_hovered || is_dragging {
        dl.fill_rect(track_rect, visual_w * 0.5, pal.bg_hover.with_alpha(90));
    }

    let max_scroll = total.saturating_sub(visible).max(1);
    let ratio = scroll.min(max_scroll) as f32 / max_scroll as f32;
    let thumb_h = (track_h * (visible as f32 / total as f32))
        .max(28.0)
        .min(track_h);
    let thumb_y = track_y + (track_h - thumb_h) * ratio;
    let thumb_col = if is_dragging {
        pal.accent
    } else if is_hovered {
        pal.fg_muted
    } else {
        pal.fg_dim.with_alpha(135)
    };
    dl.fill_rect(
        Rect::new(visual_x, thumb_y, visual_w, thumb_h),
        visual_w * 0.5,
        thumb_col,
    );

    hits.push(HitZone {
        rect: hit_rect,
        target: HitTarget::Scrollbar {
            list,
            track_y: track_y.round() as i32,
            track_h: track_h.round() as i32,
            total,
            visible,
        },
    });
}

#[allow(clippy::too_many_arguments)]
fn draw_chip(
    dl: &mut DrawList,
    te: &mut TextEngine,
    hits: &mut Vec<HitZone>,
    pal: &Palette,
    x: f32,
    y: f32,
    height: f32,
    max_width: f32,
    label: &str,
    active: bool,
    target: HitTarget,
    mouse: (f32, f32),
    hover_t: f32,
    pressed: Option<&HitTarget>,
) -> f32 {
    let weight = if active {
        FontWeight::Bold
    } else {
        FontWeight::Semibold
    };
    let text_w = te.measure(label, 12.5, weight);
    let w = (text_w + 18.0).clamp(44.0, max_width);
    let is_pressed = pressed.is_some_and(|p| p == &target);
    let dy = if is_pressed { 1.0 } else { 0.0 };
    let rect = Rect::new(x, y + dy, w, height);
    let hovered = rect.contains(mouse.0, mouse.1);
    let ht = if hovered { hover_t } else { 0.0 };

    let base_fill = if active {
        pal.segment_active
    } else {
        pal.bg_hover.with_alpha(0)
    };
    let hover_fill = if active {
        pal.bg_active
    } else if is_pressed {
        pal.bg_active
    } else {
        pal.bg_hover
    };
    let fill = base_fill.lerp(hover_fill, ht);

    let base_fg = if active { pal.fg } else { pal.fg_muted };
    let hover_fg = pal.fg;
    let fg = base_fg.lerp(hover_fg, ht);

    if fill.0[3] > 0 {
        dl.fill_rect(rect, 6.0, fill);
    }
    dl.draw_text_centered(te, rect, 12.5, fg, weight, label);
    hits.push(HitZone {
        rect: Rect::new(x, y, w, height),
        target,
    });
    w
}

/// Draw an SVG icon mathematically centered inside `rect`.
fn draw_centered_icon(
    dl: &mut DrawList,
    te: &mut TextEngine,
    icon: IconId,
    rect: Rect,
    icon_size: f32,
    color: Rgba,
) {
    let ix = rect.x + (rect.w - icon_size) * 0.5;
    let iy = rect.y + (rect.h - icon_size) * 0.5;
    dl.draw_icon(te, icon, ix, iy, icon_size, color);
}

/// Draw a sleek floating tooltip pill anchored above (or below if near top) `anchor`.
fn draw_tooltip(
    dl: &mut DrawList,
    te: &mut TextEngine,
    pal: &Palette,
    win_w: f32,
    win_h: f32,
    anchor: Rect,
    text: &str,
) {
    let tw = te.measure(text, 11.5, FontWeight::Semibold);
    let pw = tw + 18.0;
    let ph = 24.0;
    let mut px = anchor.x + (anchor.w - pw) * 0.5;
    px = px.clamp(8.0, (win_w - pw - 8.0).max(8.0));
    let mut py = anchor.y - ph - 6.0;
    if py < 6.0 {
        py = (anchor.y + anchor.h + 6.0).min((win_h - ph - 6.0).max(6.0));
    }
    let rect = Rect::new(px, py, pw, ph);
    dl.fill_rect_bordered(rect, 6.0, pal.bg_popover, 1.0, pal.ring);
    dl.draw_text_centered(te, rect, 11.5, pal.fg, FontWeight::Semibold, text);
}

#[allow(clippy::too_many_arguments)]
fn draw_modal_button(
    dl: &mut DrawList,
    te: &mut TextEngine,
    hits: &mut Vec<HitZone>,
    pal: &Palette,
    x: f32,
    y: f32,
    height: f32,
    label: &str,
    tone: ModalBtnTone,
    target: HitTarget,
    mouse: (f32, f32),
    hover_t: f32,
    pressed: Option<&HitTarget>,
) -> f32 {
    draw_labeled_button(
        dl, te, hits, pal, x, y, height, label, tone, target, mouse, hover_t, pressed, None,
    )
}

const BUTTON_ICON_SPAN: f32 = 21.0;

#[allow(clippy::too_many_arguments)]
fn draw_labeled_button(
    dl: &mut DrawList,
    te: &mut TextEngine,
    hits: &mut Vec<HitZone>,
    pal: &Palette,
    x: f32,
    y: f32,
    height: f32,
    label: &str,
    tone: ModalBtnTone,
    target: HitTarget,
    mouse: (f32, f32),
    hover_t: f32,
    pressed: Option<&HitTarget>,
    icon: Option<(IconId, f32)>,
) -> f32 {
    let weight = match tone {
        ModalBtnTone::Secondary => FontWeight::Semibold,
        ModalBtnTone::Primary | ModalBtnTone::Danger => FontWeight::Bold,
    };
    let text_w = te.measure(label, 12.5, weight);
    let icon_span = if icon.is_some() { BUTTON_ICON_SPAN } else { 0.0 };
    let w = (text_w + 28.0 + icon_span).max(84.0);
    let is_pressed = pressed.is_some_and(|p| p == &target);
    let dy = if is_pressed { 1.0 } else { 0.0 };
    let rect = Rect::new(x, y + dy, w, height);
    let hovered = rect.contains(mouse.0, mouse.1);
    let ht = if hovered { hover_t } else { 0.0 };

    let (base_fill, hover_fill, base_fg, hover_fg) = match tone {
        // High-contrast monochrome primary CTA (solid white + black text in dark, black + white in light)
        ModalBtnTone::Primary => (
            pal.primary,
            pal.primary_hover,
            pal.primary_fg,
            pal.primary_fg,
        ),
        ModalBtnTone::Danger => (
            pal.error.with_alpha(34),
            pal.error,
            pal.error,
            Rgba::rgb(26, 26, 26),
        ),
        ModalBtnTone::Secondary => (
            pal.bg_hover.with_alpha(0),
            if is_pressed { pal.bg_active } else { pal.bg_hover },
            pal.fg_muted,
            pal.fg,
        ),
    };
    let fill = base_fill.lerp(hover_fill, ht);
    let fg = base_fg.lerp(hover_fg, ht);
    if fill.0[3] > 0 {
        dl.fill_rect(rect, 6.0, fill);
    }
    if let Some((icon, elapsed)) = icon {
        let left = rect.x + (rect.w - text_w - icon_span) * 0.5;
        dl.draw_animated_icon(
            te, icon, Rect::new(left, rect.y, 15.0, rect.h), 15.0, fg,
            if hovered { elapsed } else { f32::INFINITY },
        );
        dl.draw_text_vcentered(
            te, left + icon_span, rect.y + rect.h * 0.5, 12.5, fg, weight, None, label,
        );
    } else {
        dl.draw_text_centered(te, rect, 12.5, fg, weight, label);
    }
    hits.push(HitZone {
        rect: Rect::new(x, y, w, height),
        target,
    });
    w
}

#[derive(Clone, Copy)]
enum ModalBtnTone {
    Primary,
    Danger,
    Secondary,
}

fn downloads_button_rect(width: f32) -> Rect {
    Rect::new(width - 199.0, 4.0, 30.0, 30.0)
}

fn draw_floating_surface(dl: &mut DrawList, pal: &Palette, rect: Rect, radius: f32) {
    dl.drop_shadow(rect, radius, 9.0, 5.0, pal.popover_shadow);
    dl.fill_rect_bordered(rect, radius, pal.bg_popover, 1.0, pal.popover_border);
}

fn draw_download_bar(dl: &mut DrawList, pal: &Palette, track: Rect, fraction: Option<f32>) {
    if let Some(fraction) = fraction {
        dl.fill_rect(track, track.h * 0.5, pal.bg_active);
        if fraction > 0.0 {
            let fill = Rect::new(track.x, track.y, track.w * fraction, track.h);
            dl.fill_rect(fill, track.h * 0.5, pal.fg_muted);
        }
    }
}

fn draw_recording_waveform(dl: &mut DrawList, pal: &Palette, rect: Rect, history_bars: &[f32]) {
    if rect.w < 24.0 || history_bars.is_empty() {
        return;
    }
    let num_bars = HISTORY_BARS.min((rect.w / 6.0).floor() as usize).max(12);
    let step = rect.w / num_bars as f32;
    let bar_w = (step * 0.48).clamp(2.0, 3.5);
    let cy = rect.y + rect.h * 0.5;
    let max_h = (rect.h - 8.0).max(8.0);
    let slice_start = history_bars.len().saturating_sub(num_bars);
    for (i, &val) in history_bars[slice_start..].iter().enumerate() {
        let v = val.clamp(0.0, 1.0);
        let bh = (3.5 + v * (max_h - 3.5)).min(max_h);
        let bx = rect.x + i as f32 * step;
        let col = pal
            .accent_soft
            .lerp(pal.accent, (0.25 + v * 1.15).clamp(0.0, 1.0));
        dl.fill_rect(Rect::new(bx, cy - bh * 0.5, bar_w, bh), bar_w * 0.5, col);
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_download_status(
    dl: &mut DrawList,
    te: &mut TextEngine,
    hits: &mut Vec<HitZone>,
    tooltip: &mut Option<(Rect, &'static str)>,
    pal: &Palette,
    snap: &UiSnapshot<'_>,
    rect: Rect,
    job: &DownloadJob,
) {
    let pause =
        job.can_pause()
            .then_some((IconId::Pause, HitTarget::PauseDownload, "Pause download"));
    let resume = match job.state {
        DownloadState::Paused => Some((IconId::Play, HitTarget::ResumeDownload, "Resume download")),
        DownloadState::Failed(_) => {
            Some((IconId::Refresh, HitTarget::ResumeDownload, "Retry download"))
        }
        _ => None,
    };
    let cancel =
        job.can_cancel()
            .then_some((IconId::Close, HitTarget::CancelDownload, "Cancel download"));
    let mut right = rect.x + rect.w;
    for (icon, target, tip) in [cancel, pause.or(resume)].into_iter().flatten() {
        right -= 29.0;
        let button = Rect::new(right, rect.y, 26.0, 26.0);
        let hovered = button.contains(snap.mouse_pos.0, snap.mouse_pos.1);
        if hovered {
            dl.fill_rect(
                button,
                5.0,
                pal.bg_hover.with_alpha((255.0 * snap.hover_t) as u8),
            );
            *tooltip = Some((button, tip));
        }
        dl.draw_animated_icon(
            te,
            icon,
            button,
            13.0,
            if hovered { pal.fg } else { pal.fg_muted },
            snap.icon_time(hovered),
        );
        hits.push(HitZone {
            rect: button,
            target,
        });
    }

    let failed = matches!(job.state, DownloadState::Failed(_));
    dl.draw_text_vcentered(
        te,
        rect.x,
        rect.y + 12.0,
        12.0,
        if failed { pal.error } else { pal.fg },
        FontWeight::Semibold,
        Some(right - rect.x - 4.0),
        job.label(),
    );

    let transfer = match &job.state {
        DownloadState::Running(status) => {
            status.transfer().map(|progress| progress.transfer_label())
        }
        DownloadState::Paused => job.progress.map(|progress| progress.label()),
        _ => None,
    };
    if let Some(transfer) = transfer {
        dl.draw_text_vcentered(
            te,
            rect.x,
            rect.y + 30.0,
            11.0,
            pal.fg_muted,
            FontWeight::Regular,
            Some(rect.w),
            &transfer,
        );
        draw_download_bar(
            dl,
            pal,
            Rect::new(rect.x, rect.y + 40.0, rect.w, 3.0),
            snap.download_fraction,
        );
    } else if let DownloadState::Failed(message) = &job.state {
        dl.draw_text(
            te,
            rect.x,
            rect.y + 24.0,
            11.5,
            pal.fg_muted,
            FontWeight::Regular,
            Some(rect.w),
            if rect.h > 50.0 { 2 } else { 1 },
            message,
        );
    }
}

fn render_downloads_popover(
    dl: &mut DrawList,
    te: &mut TextEngine,
    hits: &mut Vec<HitZone>,
    pal: &Palette,
    snap: &UiSnapshot<'_>,
) {
    let controls: Vec<_> = hits
        .iter()
        .filter(|zone| {
            matches!(
                zone.target,
                HitTarget::ToggleDownloads
                    | HitTarget::MinimizeWindow
                    | HitTarget::MaximizeWindow
                    | HitTarget::CloseWindow
                    | HitTarget::ToggleTheme
            )
        })
        .take(5)
        .cloned()
        .collect();
    hits.clear();
    hits.push(HitZone {
        rect: Rect::new(0.0, 0.0, snap.width, snap.height),
        target: HitTarget::CloseModal,
    });
    hits.extend(controls);

    let earlier: Vec<_> = snap
        .recent_downloads
        .iter()
        .filter(|target| snap.download.is_none_or(|job| &job.target != *target))
        .collect();
    let w = 368.0;
    let earlier_height = if earlier.is_empty() {
        0.0
    } else {
        28.0 + 26.0 * earlier.len() as f32
    };
    let h = if snap.download.is_some() { 126.0 } else { 81.0 } + earlier_height;
    let button = downloads_button_rect(snap.width);
    let panel = Rect::new(button.x + button.w - w, button.y + button.h + 5.0, w, h);
    draw_floating_surface(dl, pal, panel, 10.0);
    hits.push(HitZone {
        rect: panel,
        target: HitTarget::KeepModalOpen,
    });
    let x = panel.x + 16.0;
    dl.draw_text_vcentered(
        te,
        x,
        panel.y + 20.0,
        14.0,
        pal.fg,
        FontWeight::Semibold,
        None,
        "Downloads",
    );

    let browse = Rect::new(panel.x + w - 114.0, panel.y + 8.0, 102.0, 25.0);
    let browse_hover = browse.contains(snap.mouse_pos.0, snap.mouse_pos.1);
    dl.draw_text_centered(
        te,
        browse,
        11.5,
        if browse_hover { pal.fg } else { pal.fg_muted },
        FontWeight::Semibold,
        "Browse models",
    );
    hits.push(HitZone {
        rect: browse,
        target: HitTarget::SwitchView(AppView::Models),
    });

    let mut tooltip = None;
    if let Some(job) = snap.download {
        draw_centered_icon(
            dl,
            te,
            if matches!(job.target, InstallTarget::Engine(_)) {
                IconId::Cpu
            } else {
                IconId::Mic
            },
            Rect::new(x, panel.y + 43.0, 19.0, 18.0),
            13.0,
            pal.fg_muted,
        );
        dl.draw_text_vcentered(
            te,
            x + 25.0,
            panel.y + 52.0,
            13.0,
            pal.fg,
            FontWeight::Semibold,
            Some(w - 58.0),
            job.name(),
        );
        draw_download_status(
            dl,
            te,
            hits,
            &mut tooltip,
            pal,
            snap,
            Rect::new(x, panel.y + 64.0, w - 32.0, 54.0),
            job,
        );
    } else {
        dl.draw_text_vcentered(
            te,
            x,
            panel.y + 56.0,
            12.5,
            pal.fg_muted,
            FontWeight::Regular,
            None,
            "No downloads yet.",
        );
    }
    if !earlier.is_empty() {
        let y = panel.y + panel.h - earlier_height;
        dl.fill_rect(Rect::new(x, y, w - 32.0, 1.0), 0.0, pal.seam);
        dl.draw_text_vcentered(
            te,
            x,
            y + 15.0,
            11.0,
            pal.fg_muted,
            FontWeight::Semibold,
            None,
            "Earlier",
        );
        for (i, target) in earlier.iter().enumerate() {
            let cy = y + 36.0 + i as f32 * 26.0;
            draw_centered_icon(
                dl,
                te,
                IconId::Check,
                Rect::new(x, cy - 8.0, 16.0, 16.0),
                13.0,
                pal.fg_muted,
            );
            dl.draw_text_vcentered(
                te,
                x + 24.0,
                cy,
                12.0,
                pal.fg_prose,
                FontWeight::Regular,
                Some(w - 60.0),
                target.name(),
            );
        }
    }
    if let Some((anchor, label)) = tooltip {
        draw_tooltip(dl, te, pal, snap.width, snap.height, anchor, label);
    }
}

#[cfg(test)]
mod download_ui_tests {
    use super::*;

    #[test]
    fn download_actions_and_speech_engine_setup_are_reachable() {
        let config = DictateConfig::default();
        let inventory = LocalInventory::default();
        let job = DownloadJob {
            target: InstallTarget::Model("small_q5".into()),
            state: DownloadState::Paused,
            progress: None,
            stopping: false,
        };
        let snap = UiSnapshot {
            width: 720.0,
            height: 460.0,
            window_border: Some(Rgba::rgb(77, 77, 77)),
            view: AppView::Models,
            model_tab: ModelFilterTab::All,
            model_filter: "",
            model_search_focused: false,
            composer_menu: None,
            caret_visible: false,
            model_selected: 0,
            model_scroll: 0,
            settings_scroll: 0,
            config: &config,
            inventory: &inventory,
            model_choices: &[],
            phase: &DictationPhase::Idle,
            download: Some(&job),
            recent_downloads: &[],
            download_fraction: None,
            page_t: 1.0,
            popup_t: 1.0,
            modal_t: 0.4,
            dictation_t: 1.0,
            elapsed_secs: 0,
            history_bars: &[],
            vad_speaking: false,
            history: &[],
            selected_history: 0,
            history_scroll: 0,
            status_toast: None,
            modal: &Modal::Downloads,
            active_mic_name: "System default",
            mouse_pos: (-1.0, -1.0),
            hover_t: 1.0,
            leaving_hover: None,
            icon_hover_ms: f32::INFINITY,
            spinner_angle: 0.0,
            pressed_target: None,
            dragging_scrollbar: None,
        };
        let mut draw = DrawList::new(1.0);
        let mut text = TextEngine::new();
        let hits = render_main_window(&mut draw, &mut text, &snap);
        for target in [
            HitTarget::ToggleDownloads,
            HitTarget::CloseWindow,
            HitTarget::ResumeDownload,
            HitTarget::CancelDownload,
        ] {
            let zone = hits.iter().find(|zone| zone.target == target).unwrap();
            let x = zone.rect.x + zone.rect.w * 0.5;
            let y = zone.rect.y + zone.rect.h * 0.5;
            assert_eq!(hit_test(&hits, x, y), Some(target));
        }
        assert_eq!(hit_test(&hits, 50.0, 200.0), Some(HitTarget::CloseModal));
        assert!(draw.instances.iter().any(|shape| shape.params[2] == 2.0));

        draw.clear(1.0);
        let settings = UiSnapshot {
            view: AppView::Settings,
            modal: &Modal::None,
            download: None,
            ..snap
        };
        let hits = render_main_window(&mut draw, &mut text, &settings);
        for engine in [LocalEngineKind::WhisperCpp, LocalEngineKind::AudioCpp] {
            assert!(hits
                .iter()
                .any(|zone| zone.target == HitTarget::InstallEngine(engine)));
        }

        draw.clear(2.0);
        let confirmation = Modal::ConfirmDownload(InstallTarget::Engine(LocalEngineKind::AudioCpp));
        let confirming = UiSnapshot {
            modal: &confirmation,
            ..settings
        };
        let hits = render_main_window(&mut draw, &mut text, &confirming);
        assert!(hits
            .iter()
            .any(|zone| zone.target == HitTarget::ConfirmInstall));
        assert_eq!(draw.instances.last().unwrap().params[1], 1.0);
    }
}

pub fn render_main_window(
    dl: &mut DrawList,
    te: &mut TextEngine,
    snap: &UiSnapshot<'_>,
) -> Vec<HitZone> {
    let pal = palette();
    let mut hits = Vec::with_capacity(96);
    let mut tooltip: Option<(Rect, &'static str)> = None;
    let w = snap.width;
    let h = snap.height;
    let sidebar_w = 224.0_f32.min(w * 0.30);
    let titlebar_h = 38.0;

    // Sidebar and canvas share the window chrome.
    dl.fill_rect(Rect::new(0.0, 0.0, sidebar_w, h), 0.0, pal.bg_sidebar);

    // Draggable custom title bar across the top
    hits.push(HitZone {
        rect: Rect::new(0.0, 0.0, w - 203.0, titlebar_h),
        target: HitTarget::DragWindow,
    });

    let downloads = downloads_button_rect(w);
    let hovered = downloads.contains(snap.mouse_pos.0, snap.mouse_pos.1);
    let download_open = matches!(snap.modal, Modal::Downloads);
    let tint = pal.fg_dim.lerp(
        pal.fg,
        if download_open {
            1.0
        } else {
            snap.hover_level(&HitTarget::ToggleDownloads, hovered)
        },
    );
    if snap.download.is_some_and(DownloadJob::is_running) {
        dl.draw_rotating_icon(
            te,
            IconId::Loader,
            downloads,
            15.0,
            tint,
            snap.spinner_angle,
        );
    } else {
        dl.draw_animated_icon(
            te,
            IconId::Download,
            downloads,
            15.0,
            tint,
            snap.icon_time(hovered),
        );
    }
    if snap.download.is_some_and(|job| {
        matches!(
            job.state,
            DownloadState::Running(_) | DownloadState::Paused | DownloadState::Failed(_)
        )
    }) {
        dl.fill_circle(downloads.x + 23.0, downloads.y + 7.0, 2.5, pal.fg);
    }
    if hovered && !download_open {
        tooltip = Some((downloads, "Downloads (Ctrl+J)"));
    }
    hits.push(HitZone {
        rect: downloads,
        target: HitTarget::ToggleDownloads,
    });

    // Quick theme toggle icon right next to window controls (matching Agente Code top-right chrome)
    {
        let is_dark = current_theme() == Theme::Dark;
        let trect = Rect::new(w - 162.0, 5.0, 28.0, 28.0);
        let thov = trect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
        if thov {
            tooltip = Some((
                trect,
                if is_dark {
                    "Switch to Light theme"
                } else {
                    "Switch to Dark theme"
                },
            ));
        }
        dl.draw_animated_icon(
            te,
            if is_dark { IconId::Sun } else { IconId::Moon },
            trect,
            14.0,
            pal.fg_dim
                .lerp(pal.fg, snap.hover_level(&HitTarget::ToggleTheme, thov)),
            snap.icon_time(thov),
        );
        hits.push(HitZone {
            rect: trect,
            target: HitTarget::ToggleTheme,
        });
    }

    // Window controls (Electron SVG icons: minus, maximize-2, x) on top-right
    let win_btns = [
        (IconId::Minus, HitTarget::MinimizeWindow, "Minimize"),
        (
            IconId::Maximize,
            HitTarget::MaximizeWindow,
            "Maximize / Restore",
        ),
        (IconId::Close, HitTarget::CloseWindow, "Hide to System Tray"),
    ];
    for (i, (icon, target, tip)) in win_btns.into_iter().enumerate() {
        let bx = w - 126.0 + i as f32 * 42.0;
        let rect = Rect::new(bx, 0.0, 42.0, titlebar_h);
        let hov = rect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
        if hov {
            tooltip = Some((rect, tip));
        }
        dl.draw_animated_icon(
            te,
            icon,
            rect,
            14.0,
            pal.fg_dim.lerp(pal.fg, snap.hover_level(&target, hov)),
            snap.icon_time(hov),
        );
        hits.push(HitZone { rect, target });
    }

    // ── 2. Sidebar Navigation ────────────────────────────────────────────────
    let brand_icon = Rect::new(15.0, 12.0, 26.0, 26.0);
    draw_centered_icon(dl, te, IconId::Brand, brand_icon, 23.0, pal.fg);
    dl.draw_text_vcentered(
        te,
        46.0,
        25.0,
        16.0,
        pal.fg,
        FontWeight::Bold,
        None,
        crate::APP_NAME,
    );

    let nav_items = [
        (AppView::Dictate, IconId::Mic, "Dictate"),
        (AppView::Models, IconId::Cpu, "Models"),
    ];
    for (i, (v, icon, label)) in nav_items.into_iter().enumerate() {
        let ny = 56.0 + i as f32 * 34.0;
        let nrect = Rect::new(10.0, ny, sidebar_w - 20.0, 32.0);
        let active = snap.view == v;
        let hov = nrect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
        if active {
            dl.fill_rect(nrect, 6.0, pal.bg_element);
        } else if hov {
            dl.fill_rect(
                nrect,
                6.0,
                pal.bg_hover.with_alpha((255.0 * snap.hover_t) as u8),
            );
        }
        dl.draw_animated_icon(
            te,
            icon,
            Rect::new(22.0, ny + 8.5, 15.0, 15.0),
            15.0,
            if active || hov { pal.fg } else { pal.fg_muted },
            snap.icon_time(hov),
        );
        dl.draw_text_vcentered(
            te,
            45.0,
            ny + 16.0,
            13.5,
            if active || hov { pal.fg } else { pal.fg_muted },
            if active {
                FontWeight::Semibold
            } else {
                FontWeight::Regular
            },
            None,
            label,
        );
        hits.push(HitZone {
            rect: nrect,
            target: HitTarget::SwitchView(v),
        });
    }

    // Sidebar: Installed Local Models quick-switcher
    let installed_ids = &snap.inventory.installed;
    if !installed_ids.is_empty() {
        let sec_y = 140.0;
        dl.draw_text(
            te,
            20.0,
            sec_y,
            11.0,
            pal.fg_dim,
            FontWeight::Semibold,
            None,
            1,
            "Installed Models",
        );
        for (idx, mid) in installed_ids.iter().take(6).enumerate() {
            let my = sec_y + 22.0 + idx as f32 * 28.0;
            if my + 26.0 > h - 130.0 {
                break;
            }
            let mrect = Rect::new(10.0, my, sidebar_w - 20.0, 26.0);
            let is_cur = snap.config.backend == "local" && &snap.config.local_model == mid;
            let hov = mrect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
            if is_cur {
                dl.fill_rect(mrect, 6.0, pal.bg_element);
            } else if hov {
                dl.fill_rect(
                    mrect,
                    6.0,
                    pal.bg_hover.with_alpha((255.0 * snap.hover_t) as u8),
                );
            }
            dl.fill_circle(
                mrect.x + 12.0,
                mrect.y + 13.0,
                3.0,
                if is_cur { pal.accent } else { pal.success },
            );
            dl.draw_text_vcentered(
                te,
                mrect.x + 22.0,
                mrect.y + 13.0,
                12.5,
                if is_cur || hov { pal.fg } else { pal.fg_muted },
                if is_cur {
                    FontWeight::Semibold
                } else {
                    FontWeight::Regular
                },
                Some(mrect.w - 28.0),
                find_local_whisper_spec(mid).map_or(mid.as_str(), |spec| spec.name()),
            );
            hits.push(HitZone {
                rect: mrect,
                target: HitTarget::ActivateLocalModel(mid.clone()),
            });
        }
    }

    // Settings pinned at bottom of left rail (matching Agente Code sidebar layout)
    {
        let sy = h - 114.0;
        let srect = Rect::new(10.0, sy, sidebar_w - 20.0, 32.0);
        let active = snap.view == AppView::Settings;
        let hov = srect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
        if active {
            dl.fill_rect(srect, 6.0, pal.bg_element);
        } else if hov {
            dl.fill_rect(
                srect,
                6.0,
                pal.bg_hover.with_alpha((255.0 * snap.hover_t) as u8),
            );
        }
        dl.draw_animated_icon(
            te,
            IconId::Settings,
            Rect::new(22.0, sy + 8.5, 15.0, 15.0),
            15.0,
            if active || hov { pal.fg } else { pal.fg_muted },
            snap.icon_time(hov),
        );
        dl.draw_text_vcentered(
            te,
            45.0,
            sy + 16.0,
            13.5,
            if active || hov { pal.fg } else { pal.fg_muted },
            if active {
                FontWeight::Semibold
            } else {
                FontWeight::Regular
            },
            None,
            "Settings",
        );
        hits.push(HitZone {
            rect: srect,
            target: HitTarget::SwitchView(AppView::Settings),
        });
    }

    // Sidebar bottom status footer (scannable, quiet)
    let runtime_ready = if snap.config.backend == "local" {
        snap.inventory.is_ready(&snap.config.local_model)
    } else {
        snap.config.has_api_key(&snap.config.backend)
    };
    let summary = if snap.config.backend == "local" {
        find_local_whisper_spec(&snap.config.local_model)
            .map_or(snap.config.local_model.as_str(), |spec| spec.name())
            .to_owned()
    } else {
        snap.config.active_summary()
    };
    let status_rect = Rect::new(10.0, h - 68.0, sidebar_w - 20.0, 53.0);
    let hovered = status_rect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
    dl.fill_rect(
        Rect::new(19.0, status_rect.y - 4.0, sidebar_w - 38.0, 1.0),
        0.0,
        pal.seam,
    );
    if hovered {
        dl.fill_rect(
            status_rect,
            6.0,
            pal.bg_hover.with_alpha((255.0 * snap.hover_t) as u8),
        );
    }
    dl.draw_text_vcentered(
        te,
        20.0,
        status_rect.y + 16.0,
        12.5,
        pal.fg,
        FontWeight::Semibold,
        Some(status_rect.w - 22.0),
        &summary,
    );
    dl.fill_circle(
        23.0,
        status_rect.y + 36.0,
        3.0,
        if runtime_ready {
            pal.success
        } else {
            pal.warning
        },
    );
    dl.draw_text_vcentered(
        te,
        33.0,
        status_rect.y + 36.0,
        11.5,
        pal.fg_muted,
        FontWeight::Regular,
        None,
        if runtime_ready {
            "Model ready"
        } else {
            "Setup required"
        },
    );
    hits.push(HitZone {
        rect: status_rect,
        target: HitTarget::SwitchView(AppView::Models),
    });

    // ── 3. Main Content Area ─────────────────────────────────────────────────
    let content = Rect::new(
        sidebar_w + 1.0,
        titlebar_h,
        w - sidebar_w - 1.0,
        h - titlebar_h
            - if snap.status_toast.is_some() {
                34.0
            } else {
                0.0
            },
    );

    let page_first = dl.instances.len();
    match snap.view {
        AppView::Dictate => render_dictate_view(
            dl,
            te,
            &mut hits,
            &mut tooltip,
            &pal,
            content,
            snap,
            runtime_ready,
        ),
        AppView::Models => render_models_view(dl, te, &mut hits, &mut tooltip, &pal, content, snap),
        AppView::Settings => {
            render_settings_view(dl, te, &mut hits, &mut tooltip, &pal, content, snap)
        }
    }

    dl.fade_from(page_first, 0.62 + 0.38 * snap.page_t);

    if let Some(message) = snap.status_toast {
        let footer = Rect::new(content.x, h - 34.0, content.w, 34.0);
        dl.fill_rect(Rect::new(footer.x, footer.y, footer.w, 1.0), 0.0, pal.seam);
        dl.draw_text_vcentered(
            te,
            footer.x + 18.0,
            footer.y + 16.0,
            12.5,
            pal.fg_muted,
            FontWeight::Regular,
            Some(footer.w - 36.0),
            message,
        );
    }

    // Paint floating icon tooltip on top of main UI when no modal is blocking
    if matches!(snap.modal, Modal::None) {
        if let Some((anchor, tip_text)) = tooltip {
            draw_tooltip(dl, te, &pal, w, h, anchor, tip_text);
        }
    }

    // ── 4. Modals (Provider Config, Download Confirm, Delete Clip Confirm) ───
    let modal_first = dl.instances.len();
    match snap.modal {
        Modal::None => {}
        Modal::Downloads => render_downloads_popover(dl, te, &mut hits, &pal, snap),
        Modal::ProviderConfig {
            provider_id,
            provider_name,
            is_self_hosted,
            base_url,
            model_id,
            api_key,
            active_field,
            reveal_key,
            ..
        } => {
            te.with_uncached_layouts(|te| {
                render_provider_config_modal(
                    dl,
                    te,
                    &mut hits,
                    &pal,
                    snap,
                    provider_id,
                    provider_name,
                    *is_self_hosted,
                    base_url,
                    model_id,
                    api_key,
                    *active_field,
                    *reveal_key,
                );
            });
        }
        Modal::ConfirmDownload(target) => {
            render_confirm_download_modal(dl, te, &mut hits, &pal, snap, target)
        }
        Modal::ConfirmDeleteClip { preview, .. } => {
            render_confirm_delete_modal(dl, te, &mut hits, &pal, snap, preview);
        }
        Modal::Transcript { id, scroll, copied } => {
            if let Some(entry) = snap.history.iter().find(|entry| &entry.id == id) {
                render_transcript_modal(dl, te, &mut hits, &pal, snap, entry, *scroll, *copied);
            }
        }
    }

    dl.fade_from(modal_first, 0.38 + 0.62 * snap.modal_t);
    if let Some(border) = snap.window_border {
        dl.fill_rect_bordered(
            Rect::new(0.0, 0.0, w, h),
            0.0,
            Rgba::transparent(),
            1.0 / dl.scale,
            border,
        );
    }

    hits
}

#[allow(clippy::too_many_arguments)]
fn render_dictate_view(
    dl: &mut DrawList,
    te: &mut TextEngine,
    hits: &mut Vec<HitZone>,
    tooltip: &mut Option<(Rect, &'static str)>,
    pal: &Palette,
    area: Rect,
    snap: &UiSnapshot<'_>,
    runtime_ready: bool,
) {
    let col_w = (area.w - 56.0).clamp(360.0, 720.0);
    let col_x = area.x + (area.w - col_w) * 0.5;
    let is_recording = matches!(snap.phase, DictationPhase::Recording);
    let is_processing = matches!(snap.phase, DictationPhase::Transcribing);

    dl.draw_text(
        te,
        col_x,
        area.y + 19.0,
        25.0,
        pal.fg,
        FontWeight::Bold,
        None,
        1,
        "Dictate",
    );
    let shortcut = Rect::new(col_x + col_w - 147.0, area.y + 20.0, 147.0, 28.0);
    dl.draw_text_centered(
        te,
        shortcut,
        12.0,
        pal.fg_muted,
        FontWeight::Semibold,
        "Ctrl + Shift + Space",
    );

    let pill = Rect::new(col_x, area.y + 63.0, col_w, 118.0);
    dl.fill_rect_bordered(
        pill, 12.0, pal.bg_composer, 1.0,
        if is_recording { pal.accent.with_alpha(110) } else { pal.ring },
    );

    let local = snap.config.backend == "local";
    let card_first = dl.instances.len();
    let setup = snap.download.filter(|job| {
        local && !runtime_ready
            && job.target.model_id() == Some(snap.config.local_model.as_str())
            && !matches!(job.state, DownloadState::Completed | DownloadState::Canceled)
    });
    let (title, detail) = match (snap.phase, setup) {
        (DictationPhase::Recording, _) => {
            let minutes = snap.elapsed_secs / 60;
            let seconds = snap.elapsed_secs % 60;
            let detail = if snap.vad_speaking { "Voice detected" } else { "Microphone is on" };
            (format!("Listening · {minutes}:{seconds:02}"), detail)
        }
        (DictationPhase::Transcribing, _) => (
            "Transcribing…".into(),
            "Turning your audio into text",
        ),
        (_, Some(job)) => (job.name().to_owned(), job.label()),
        (DictationPhase::Error(error), _) => ("Dictation needs attention".into(), error.as_str()),
        _ if !runtime_ready && local => (
            "Set up dictation".into(),
            "Download a speech model to start dictating on this device.",
        ),
        _ if !runtime_ready => (
            "Connect your provider".into(),
            "Add your provider credentials to start dictating.",
        ),
        _ => ("Ready to record".into(), "Record here and get the text on your clipboard."),
    };

    let text_x = pill.x + 17.0;
    let title_y = pill.y + 23.0;
    let title_x = if is_recording {
        dl.fill_circle(text_x + 3.0, title_y, 3.0, pal.accent);
        text_x + 15.0
    } else if is_processing {
        dl.draw_rotating_icon(
            te, IconId::Loader, Rect::new(text_x, title_y - 8.0, 16.0, 16.0),
            14.0, pal.fg_muted, snap.spinner_angle,
        );
        text_x + 23.0
    } else {
        text_x
    };
    let progress = setup.and_then(|job| job.progress).filter(|_| !is_recording && !is_processing);
    let progress_label = progress.map(|progress| progress.label());
    let progress_width = progress_label.as_ref()
        .map_or(0.0, |label| te.measure(label, 11.5, FontWeight::Regular) + 12.0);
    dl.draw_text_vcentered(
        te, title_x, title_y, 15.5, pal.fg, FontWeight::Bold,
        Some(pill.x + pill.w - 17.0 - title_x), &title,
    );
    dl.draw_text(
        te, text_x, pill.y + 34.0, 12.5, pal.fg_muted, FontWeight::Regular,
        Some(pill.w - 34.0 - progress_width), if progress.is_some() { 1 } else { 2 }, detail,
    );
    if let Some(label) = progress_label {
        dl.draw_text(
            te, pill.x + pill.w - 5.0 - progress_width, pill.y + 35.0, 11.5,
            pal.fg_muted, FontWeight::Regular, None, 1, &label,
        );
        draw_download_bar(
            dl, pal, Rect::new(text_x, pill.y + 57.0, pill.w - 34.0, 3.0), snap.download_fraction,
        );
    }

    let bar_y = pill.y + 72.0;
    let mut right = pill.x + pill.w - 12.0;
    let action = match snap.phase {
        DictationPhase::Transcribing => None,
        DictationPhase::Recording => Some(("Stop recording", IconId::Square, HitTarget::ToggleRecord)),
        _ if setup.is_some() => Some(("View download", IconId::Download, HitTarget::ToggleDownloads)),
        _ if !runtime_ready && local => Some((
            "Set up model", IconId::Download, HitTarget::DownloadLocalModel(snap.config.local_model.clone()),
        )),
        _ if !runtime_ready => Some((
            "Configure", IconId::Settings, HitTarget::OpenProviderConfig(snap.config.backend.clone()),
        )),
        _ => Some(("Record", IconId::Mic, HitTarget::ToggleRecord)),
    };
    if let Some((label, icon, target)) = action {
        right -= (te.measure(label, 12.5, FontWeight::Bold) + 28.0 + BUTTON_ICON_SPAN).max(84.0);
        draw_labeled_button(
            dl, te, hits, pal, right, bar_y, 34.0, label, ModalBtnTone::Primary,
            target, snap.mouse_pos, snap.hover_t, snap.pressed_target, Some((icon, snap.icon_hover_ms)),
        );
    }

    let first_menu_hit = hits.len();
    if is_recording {
        let cancel_x = pill.x + 10.0;
        let cancel_w = draw_chip(
            dl, te, hits, pal, cancel_x, bar_y, 34.0, 88.0, "Cancel", false,
            HitTarget::CancelRecording, snap.mouse_pos, snap.hover_t, snap.pressed_target,
        );
        draw_recording_waveform(
            dl, pal, Rect::new(cancel_x + cancel_w + 10.0, bar_y, right - cancel_x - cancel_w - 22.0, 34.0),
            snap.history_bars,
        );
    } else if is_processing {
        let width = (te.measure("Cancel", 12.5, FontWeight::Semibold) + 18.0).max(44.0);
        draw_chip(
            dl, te, hits, pal, right - width, bar_y, 34.0, width, "Cancel", false,
            HitTarget::CancelRecording, snap.mouse_pos, snap.hover_t, snap.pressed_target,
        );
    } else {
        let model = if local {
            find_local_whisper_spec(&snap.config.local_model).map_or(snap.config.local_model.as_str(), |spec| {
                spec.name().split('(').next().unwrap_or(spec.name()).trim()
            })
        } else {
            find_commercial_provider(&snap.config.backend).map_or(snap.config.backend.as_str(), |provider| provider.name)
        };
        let menus = [
            (model, 150.0, ComposerMenu::Models),
            (DictateConfig::language_label(&snap.config.language), 116.0, ComposerMenu::Language),
            (snap.config.delivery_mode.short_label(), 96.0, ComposerMenu::Delivery),
        ].map(|(label, max_width, menu)| {
            let open = snap.composer_menu == Some(menu);
            let label = format!("{} {label}", if open { "▴" } else { "▾" });
            let weight = if open { FontWeight::Bold } else { FontWeight::Semibold };
            let width = (te.measure(&label, 12.5, weight) + 18.0).clamp(44.0, max_width);
            (label, width, menu, open)
        });
        let gaps = 8.0;
        let minimum = 44.0 * menus.len() as f32;
        let desired = menus.iter().map(|(_, width, _, _)| width).sum::<f32>();
        let mut x = pill.x + 10.0;
        let extra = desired - minimum;
        let room = (right - 10.0 - x - gaps - minimum).max(0.0);
        let fit = if extra > 0.0 { (room / extra).min(1.0) } else { 1.0 };
        for (label, width, menu, open) in menus {
            x += draw_chip(
                dl, te, hits, pal, x, bar_y, 34.0, 44.0 + (width - 44.0) * fit,
                &label, open, HitTarget::ToggleComposerMenu(menu),
                snap.mouse_pos, snap.hover_t, snap.pressed_target,
            ) + 4.0;
        }
    }
    let last_menu_hit = hits.len();
    dl.fade_from(card_first, 0.60 + 0.40 * snap.dictation_t);

    // ── Transcript History List (clean edge-aligned rows, no boxes-on-boxes) ─
    let list_header_y = pill.y + pill.h + 26.0;
    dl.draw_text(
        te,
        col_x + 4.0,
        list_header_y,
        13.0,
        pal.fg_dim,
        FontWeight::Bold,
        None,
        1,
        &format!("History ({})", snap.history.len()),
    );

    let mut cur_y = list_header_y + 26.0;
    let bottom = area.y + area.h - 18.0;

    if snap.history.is_empty() {
        dl.draw_text(
            te,
            col_x + 4.0,
            cur_y + 6.0,
            13.5,
            pal.fg_muted,
            FontWeight::Regular,
            Some(col_w - 8.0),
            2,
            "Your transcripts appear here, ready to open or copy.",
        );
    } else {
        let start = snap.history_scroll.min(snap.history.len().saturating_sub(1));
        let list_top_y = cur_y;
        let mut rendered_count = 0usize;
        let row_col_w = if snap.history.len() > 3 {
            col_w - 14.0
        } else {
            col_w
        };
        for (idx, entry) in snap.history.iter().enumerate().skip(start) {
            let text_w = row_col_w - 28.0;
            let lines = te.measure_wrapped_lines(&entry.text, 14.5, text_w, 3);
            let row_h = 34.0 + lines as f32 * 21.0;
            if cur_y + row_h > bottom {
                break;
            }
            rendered_count += 1;

            let rrect = Rect::new(col_x, cur_y, row_col_w, row_h);
            let is_sel = idx == snap.selected_history;
            let is_hov = rrect.contains(snap.mouse_pos.0, snap.mouse_pos.1);

            if is_sel {
                dl.fill_rect_bordered(rrect, 8.0, pal.bg_panel, 1.0, pal.seam);
            } else if is_hov {
                dl.fill_rect(rrect, 8.0, pal.bg_panel.with_alpha(150));
            }
            hits.push(HitZone {
                rect: rrect,
                target: HitTarget::OpenHistory(entry.id.clone()),
            });

            // Transcript body
            dl.draw_text(
                te,
                rrect.x + 14.0,
                rrect.y + 10.0,
                14.5,
                pal.fg_prose,
                FontWeight::Regular,
                Some(text_w),
                3,
                &entry.text,
            );

            // Quiet metadata row underneath
            let meta_y = rrect.y + row_h - 22.0;
            let meta = format!(
                "{}  ·  {:.1}s",
                entry
                    .created_at
                    .with_timezone(&chrono::Local)
                    .format("%b %e, %H:%M"),
                entry.duration_secs,
            );
            dl.draw_text(
                te,
                rrect.x + 14.0,
                meta_y,
                11.5,
                pal.fg_dim,
                FontWeight::Regular,
                None,
                1,
                &meta,
            );

            {
                let mut ax = rrect.x + rrect.w - 8.0;
                for (icon, target, is_del, tip) in [
                    (
                        IconId::Trash,
                        HitTarget::RequestDeleteHistory(entry.id.clone()),
                        true,
                        "Delete transcript",
                    ),
                    (
                        IconId::Copy,
                        HitTarget::CopyHistory(entry.id.clone()),
                        false,
                        "Copy transcript",
                    ),
                ] {
                    let bw = 26.0;
                    ax -= bw;
                    let arect = Rect::new(ax, meta_y - 5.0, bw, 24.0);
                    let ahov = arect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
                    if ahov {
                        dl.fill_rect(arect, 6.0, pal.bg_hover);
                        if snap.composer_menu.is_none() {
                            *tooltip = Some((arect, tip));
                        }
                    }
                    dl.draw_animated_icon(
                        te,
                        icon,
                        arect,
                        14.0,
                        if ahov && is_del {
                            pal.error
                        } else if ahov {
                            pal.fg
                        } else {
                            pal.fg_dim
                        },
                        snap.icon_time(ahov),
                    );
                    hits.push(HitZone { rect: arect, target });
                    ax -= 4.0;
                }
            }

            // Subtle separator line
            dl.fill_rect(
                Rect::new(col_x + 12.0, cur_y + row_h + 2.0, row_col_w - 24.0, 1.0),
                0.0,
                pal.seam,
            );
            cur_y += row_h + 6.0;
        }

        let visible_hist = rendered_count.max(1);
        if snap.history.len() > visible_hist {
            draw_scrollbar(
                dl,
                hits,
                pal,
                ScrollList::History,
                col_x + col_w - 7.0,
                list_top_y,
                (bottom - list_top_y).max(24.0),
                start,
                visible_hist,
                snap.history.len(),
                snap.mouse_pos,
                snap.dragging_scrollbar,
            );
        }
    }

    // Floating Model Selector Dropdown anchored under the composer pill model chip
    if let Some(menu) = snap.composer_menu.filter(|_| !is_recording && !is_processing) {
        let popup_first = dl.instances.len();
        let anchor_x = hits[first_menu_hit..last_menu_hit]
            .iter()
            .find(|zone| zone.target == HitTarget::ToggleComposerMenu(menu))
            .map_or(pill.x + 12.0, |zone| zone.rect.x);
        hits.push(HitZone {
            rect: Rect::new(0.0, 0.0, snap.width, snap.height),
            target: HitTarget::CloseComposerMenu,
        });
        hits.extend_from_within(first_menu_hit..last_menu_hit);
        let y = bar_y + 39.0;
        match menu {
            ComposerMenu::Models => render_model_dropdown(dl, te, hits, pal, snap, anchor_x, y),
            ComposerMenu::Language => {
                let choices: Vec<_> = LANGUAGES
                    .iter()
                    .map(|&code| {
                        (
                            DictateConfig::language_label(code),
                            HitTarget::SetLanguage(code),
                            snap.config.language == code,
                        )
                    })
                    .collect();
                render_composer_choices(dl, te, hits, pal, snap, anchor_x, y, 138.0, 2, &choices);
            }
            ComposerMenu::Delivery => {
                let choices = [
                    DeliveryMode::PasteAndClipboard,
                    DeliveryMode::TypeAndClipboard,
                    DeliveryMode::ClipboardOnly,
                ]
                .map(|mode| {
                    (
                        mode.label(),
                        HitTarget::SetDelivery(mode),
                        snap.config.delivery_mode == mode,
                    )
                });
                render_composer_choices(dl, te, hits, pal, snap, anchor_x, y, 251.0, 1, &choices);
            }
        }
        dl.fade_from(popup_first, 0.40 + 0.60 * snap.popup_t);
    }
}

#[allow(clippy::too_many_arguments)]
fn render_composer_choices(
    dl: &mut DrawList,
    te: &mut TextEngine,
    hits: &mut Vec<HitZone>,
    pal: &Palette,
    snap: &UiSnapshot<'_>,
    x: f32,
    y: f32,
    cell_width: f32,
    columns: usize,
    choices: &[(&str, HitTarget, bool)],
) {
    let rows = choices.len().div_ceil(columns);
    let width = cell_width * columns as f32 + 12.0;
    let x = x.min(snap.width - width - 16.0);
    let panel = Rect::new(x, y, width, rows as f32 * 29.0 + 12.0);
    draw_floating_surface(dl, pal, panel, 10.0);
    for (index, (label, target, selected)) in choices.iter().enumerate() {
        let rect = Rect::new(
            x + 6.0 + (index / rows) as f32 * cell_width,
            y + 6.0 + (index % rows) as f32 * 29.0,
            cell_width,
            28.0,
        );
        if *selected || rect.contains(snap.mouse_pos.0, snap.mouse_pos.1) {
            dl.fill_rect(
                rect,
                6.0,
                if *selected {
                    pal.bg_active
                } else {
                    pal.bg_hover
                },
            );
        }
        dl.draw_text_vcentered(
            te,
            rect.x + 10.0,
            rect.y + 14.0,
            12.5,
            pal.fg,
            FontWeight::Regular,
            Some(rect.w - 20.0),
            label,
        );
        hits.push(HitZone {
            rect,
            target: target.clone(),
        });
    }
}

fn render_model_dropdown(
    dl: &mut DrawList,
    te: &mut TextEngine,
    hits: &mut Vec<HitZone>,
    pal: &Palette,
    snap: &UiSnapshot<'_>,
    anchor_x: f32,
    anchor_y: f32,
) {
    use crate::catalog::COMMERCIAL_STT_PROVIDERS;

    let installed = &snap.inventory.installed;
    let mut ready_cloud: Vec<(&'static str, &'static str, String)> = Vec::new();
    for prov in COMMERCIAL_STT_PROVIDERS {
        if snap.config.has_api_key(prov.id) {
            let mdl = snap
                .config
                .resolve_commercial_auth(prov.id)
                .map(|(_, _, m)| m)
                .unwrap_or_else(|| prov.default_model.to_string());
            ready_cloud.push((prov.id, prov.name, mdl));
            if ready_cloud.len() >= 5 {
                break;
            }
        }
    }

    let dw = 316.0_f32.min(snap.width - 32.0);
    let dx = anchor_x.clamp(12.0, (snap.width - dw - 12.0).max(12.0));

    let row_h = 30.0;
    let mut dh = 14.0;
    if !installed.is_empty() {
        dh += 22.0 + installed.len().min(6) as f32 * row_h;
    }
    if !ready_cloud.is_empty() {
        dh += 24.0 + ready_cloud.len() as f32 * row_h;
    }
    dh += 38.0; // divider + "Browse all models…" footer
    let max_h = (snap.height - anchor_y - 16.0).max(120.0);
    let dh = dh.min(max_h);

    let drect = Rect::new(dx, anchor_y, dw, dh);
    draw_floating_surface(dl, pal, drect, 10.0);

    let mut y = drect.y + 8.0;
    if !installed.is_empty() {
        dl.draw_text(
            te,
            drect.x + 12.0,
            y + 2.0,
            10.5,
            pal.fg_dim,
            FontWeight::Bold,
            None,
            1,
            "INSTALLED LOCAL",
        );
        y += 20.0;
        for mid in installed.iter().take(6) {
            if y + row_h > drect.y + drect.h - 38.0 {
                break;
            }
            let rrect = Rect::new(drect.x + 6.0, y, drect.w - 12.0, row_h - 2.0);
            let is_cur = snap.config.backend == "local" && &snap.config.local_model == mid;
            let hov = rrect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
            if is_cur || hov {
                dl.fill_rect(
                    rrect,
                    7.0,
                    if is_cur { pal.bg_element } else { pal.bg_hover },
                );
            }
            dl.fill_circle(
                rrect.x + 11.0,
                rrect.y + rrect.h * 0.5,
                3.0,
                if is_cur { pal.accent } else { pal.success },
            );
            dl.draw_text_vcentered(
                te,
                rrect.x + 22.0,
                rrect.y + rrect.h * 0.5,
                12.5,
                if is_cur || hov { pal.fg } else { pal.fg_prose },
                if is_cur {
                    FontWeight::Bold
                } else {
                    FontWeight::Semibold
                },
                Some(rrect.w - 72.0),
                mid,
            );
            let tag = if is_cur { "Active" } else { "Local" };
            let tw = te.measure(tag, 11.0, FontWeight::Regular);
            dl.draw_text_vcentered(
                te,
                rrect.x + rrect.w - tw - 10.0,
                rrect.y + rrect.h * 0.5,
                11.0,
                if is_cur { pal.accent } else { pal.fg_dim },
                FontWeight::Regular,
                None,
                tag,
            );
            hits.push(HitZone {
                rect: rrect,
                target: HitTarget::SelectDropdownLocal(mid.clone()),
            });
            y += row_h;
        }
    }

    if !ready_cloud.is_empty() && y + 54.0 < drect.y + drect.h - 34.0 {
        y += 4.0;
        dl.draw_text(
            te,
            drect.x + 12.0,
            y + 2.0,
            10.5,
            pal.fg_dim,
            FontWeight::Bold,
            None,
            1,
            "CLOUD & SELF-HOSTED",
        );
        y += 20.0;
        for (pid, pname, mdl) in ready_cloud {
            if y + row_h > drect.y + drect.h - 38.0 {
                break;
            }
            let rrect = Rect::new(drect.x + 6.0, y, drect.w - 12.0, row_h - 2.0);
            let is_cur = snap.config.backend == pid;
            let hov = rrect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
            if is_cur || hov {
                dl.fill_rect(
                    rrect,
                    7.0,
                    if is_cur { pal.bg_element } else { pal.bg_hover },
                );
            }
            dl.fill_circle(
                rrect.x + 11.0,
                rrect.y + rrect.h * 0.5,
                3.0,
                if is_cur { pal.accent } else { pal.fg_dim },
            );
            let label = format!("{pname} · {mdl}");
            dl.draw_text_vcentered(
                te,
                rrect.x + 22.0,
                rrect.y + rrect.h * 0.5,
                12.5,
                if is_cur || hov { pal.fg } else { pal.fg_prose },
                if is_cur {
                    FontWeight::Bold
                } else {
                    FontWeight::Semibold
                },
                Some(rrect.w - 30.0),
                &label,
            );
            hits.push(HitZone {
                rect: rrect,
                target: HitTarget::SelectDropdownCloud {
                    backend: pid.to_string(),
                    model_id: mdl,
                },
            });
            y += row_h;
        }
    }

    // Bottom divider + "Browse all models & providers…"
    let foot_y = drect.y + drect.h - 34.0;
    dl.fill_rect(Rect::new(drect.x + 8.0, foot_y - 3.0, drect.w - 16.0, 1.0), 0.0, pal.seam);
    let browse_rect = Rect::new(drect.x + 6.0, foot_y, drect.w - 12.0, 28.0);
    let browse_hov = browse_rect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
    if browse_hov {
        dl.fill_rect(browse_rect, 7.0, pal.bg_hover);
    }
    dl.draw_text_vcentered(
        te,
        browse_rect.x + 10.0,
        browse_rect.y + browse_rect.h * 0.5,
        12.5,
        if browse_hov { pal.accent } else { pal.fg_muted },
        FontWeight::Semibold,
        None,
        "Browse all models & providers…",
    );
    hits.push(HitZone {
        rect: browse_rect,
        target: HitTarget::SwitchView(AppView::Models),
    });
}

#[allow(clippy::too_many_arguments)]
fn render_models_view(
    dl: &mut DrawList,
    te: &mut TextEngine,
    hits: &mut Vec<HitZone>,
    tooltip: &mut Option<(Rect, &'static str)>,
    pal: &Palette,
    area: Rect,
    snap: &UiSnapshot<'_>,
) {
    let col_w = (area.w - 56.0).clamp(400.0, 760.0);
    let col_x = area.x + (area.w - col_w) * 0.5;

    dl.draw_text(
        te,
        col_x,
        area.y + 19.0,
        25.0,
        pal.fg,
        FontWeight::Bold,
        None,
        1,
        "Models",
    );
    let label = "Speech engines";
    let width = te.measure(label, 12.5, FontWeight::Semibold) + 28.0;
    draw_modal_button(
        dl,
        te,
        hits,
        pal,
        col_x + col_w - width,
        area.y + 20.0,
        31.0,
        label,
        ModalBtnTone::Secondary,
        HitTarget::OpenEngineSettings,
        snap.mouse_pos,
        snap.hover_t,
        snap.pressed_target,
    );

    let search_rect = Rect::new(col_x, area.y + 62.0, col_w, 34.0);
    let search_hov = search_rect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
    let search_focused = snap.model_search_focused;
    let search_fill = if search_focused {
        pal.bg_composer
    } else if search_hov {
        pal.bg_panel.lerp(pal.bg_composer, snap.hover_t)
    } else {
        pal.bg_panel
    };
    dl.fill_rect(search_rect, 6.0, search_fill);
    if search_focused {
        dl.fill_rect(
            Rect::new(
                col_x + 6.0,
                search_rect.y + search_rect.h - 1.0,
                col_w - 12.0,
                1.0,
            ),
            0.0,
            pal.ring_focus,
        );
    }
    dl.draw_animated_icon(
        te,
        IconId::Search,
        Rect::new(search_rect.x + 11.0, search_rect.y + 9.5, 15.0, 15.0),
        15.0,
        if search_focused {
            pal.fg
        } else if search_hov {
            pal.fg_muted
        } else {
            pal.fg_dim
        },
        snap.icon_time(search_hov),
    );

    let text_x = search_rect.x + 34.0;
    let text_cy = search_rect.y + search_rect.h * 0.5;
    let has_filter = !snap.model_filter.is_empty();
    let max_text_w = search_rect.w - if has_filter { 72.0 } else { 48.0 };

    if has_filter {
        let drawn_w = dl.draw_text_vcentered(
            te,
            text_x,
            text_cy,
            13.5,
            pal.fg,
            FontWeight::Regular,
            Some(max_text_w),
            snap.model_filter,
        );
        if search_focused && snap.caret_visible {
            let cx = (text_x + drawn_w + 1.5).min(text_x + max_text_w);
            dl.fill_rect(Rect::new(cx, text_cy - 8.0, 1.5, 16.0), 0.75, pal.fg);
        }
    } else {
        if search_focused && snap.caret_visible {
            dl.fill_rect(Rect::new(text_x, text_cy - 8.0, 1.5, 16.0), 0.75, pal.fg);
        }
        dl.draw_text_vcentered(
            te,
            text_x + if search_focused { 4.0 } else { 0.0 },
            text_cy,
            13.5,
            pal.fg_dim,
            FontWeight::Regular,
            Some(max_text_w),
            if search_focused {
                "Filter models or providers…"
            } else {
                "Click or type anywhere to filter models or providers…"
            },
        );
    }

    hits.push(HitZone {
        rect: search_rect,
        target: HitTarget::FocusModelSearch,
    });

    if has_filter {
        let clear_rect = Rect::new(
            search_rect.x + search_rect.w - 30.0,
            search_rect.y + 6.0,
            24.0,
            24.0,
        );
        let clear_hov = clear_rect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
        if clear_hov {
            dl.fill_rect(clear_rect, 6.0, pal.bg_hover);
            *tooltip = Some((clear_rect, "Clear filter"));
        }
        dl.draw_animated_icon(
            te,
            IconId::Close,
            clear_rect,
            14.0,
            if clear_hov { pal.fg } else { pal.fg_muted },
            snap.icon_time(clear_hov),
        );
        hits.push(HitZone {
            rect: clear_rect,
            target: HitTarget::ClearModelSearch,
        });
    }

    // Filter tabs (Installed first so downloaded models are 1 click away)
    let tabs = [
        ModelFilterTab::Installed,
        ModelFilterTab::All,
        ModelFilterTab::NonWhisper,
        ModelFilterTab::Whisper,
        ModelFilterTab::HuggingFace,
        ModelFilterTab::Commercial,
    ];
    let mut tx = col_x;
    let mut ty = area.y + 106.0;
    for tab in tabs {
        let weight = if snap.model_tab == tab {
            FontWeight::Bold
        } else {
            FontWeight::Semibold
        };
        let width = (te.measure(tab.label(), 12.5, weight) + 18.0).max(44.0);
        if tx + width > col_x + col_w && tx > col_x {
            tx = col_x;
            ty += 31.0;
        }
        tx += draw_chip(
            dl,
            te,
            hits,
            pal,
            tx,
            ty,
            28.0,
            f32::INFINITY,
            tab.label(),
            snap.model_tab == tab,
            HitTarget::SetModelTab(tab),
            snap.mouse_pos,
            snap.hover_t,
            snap.pressed_target,
        ) + 6.0;
    }

    // Scrollable model list + interactive scrollbar + scannable position indicator
    let choices = snap.model_choices;
    let list_y = ty + 40.0;
    let list_bottom = area.y + area.h - 16.0;
    let list_h = (list_bottom - list_y).max(46.0);
    let row_h = 46.0;
    let max_rows = (list_h / row_h).floor().max(1.0) as usize;
    let max_scroll = choices.len().saturating_sub(max_rows);
    let start = snap.model_scroll.min(max_scroll);
    let has_scrollbar = choices.len() > max_rows;
    let row_w = if has_scrollbar { col_w - 14.0 } else { col_w };

    // Position indicator (e.g. "1–9 of 68") aligned hard right on the page header row
    if !choices.is_empty() {
        let shown_end = (start + max_rows).min(choices.len());
        let range_lbl = format!("{}–{} of {}", start + 1, shown_end, choices.len());
        let rw = te.measure(&range_lbl, 12.0, FontWeight::Regular);
        dl.draw_text(
            te,
            col_x + col_w - rw,
            area.y + 18.0,
            12.0,
            pal.fg_dim,
            FontWeight::Regular,
            None,
            1,
            &range_lbl,
        );
    } else {
        dl.draw_text(
            te,
            col_x + 4.0,
            list_y + 10.0,
            13.5,
            pal.fg_dim,
            FontWeight::Regular,
            None,
            1,
            "No matching models. Try clearing your filter or switching tabs.",
        );
    }

    for (idx, choice) in choices.iter().enumerate().skip(start).take(max_rows) {
        let ry = list_y + (idx - start) as f32 * row_h;
        let rrect = Rect::new(col_x, ry, row_w, row_h - 4.0);
        let is_sel = idx == snap.model_selected;
        let is_hov = rrect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
        let hover_level = snap.hover_level(&HitTarget::PickModel(idx), is_hov);

        let is_active = if choice.is_local {
            snap.config.backend == "local" && snap.config.local_model == choice.model_id
                && snap.inventory.is_ready(&choice.model_id)
        } else {
            snap.config.backend == choice.backend
                && snap.config.commercial_model == choice.model_id
        };

        if is_sel || hover_level > 0.0 {
            dl.fill_rect(
                rrect,
                8.0,
                if is_sel {
                    pal.bg_element
                } else {
                    pal.bg_hover.with_alpha((255.0 * hover_level) as u8)
                },
            );
        }

        if is_active {
            dl.fill_circle(rrect.x + 12.0, rrect.y + 20.0, 4.0, pal.accent);
        }

        let job = snap.download.filter(|job| {
            choice.is_local
                && job.target.model_id() == Some(choice.model_id.as_str())
                && !matches!(
                    job.state,
                    DownloadState::Completed | DownloadState::Canceled
                )
        });
        let transfer_width = (row_w * 0.53).min(335.0);
        let has_cfg_btn = !choice.is_local;
        let text_right_pad = if job.is_some() {
            transfer_width + 34.0
        } else if has_cfg_btn {
            165.0
        } else {
            135.0
        };
        let label = if choice.is_local {
            find_local_whisper_spec(&choice.model_id)
                .map_or(choice.label.as_str(), |spec| spec.name())
        } else {
            &choice.label
        };

        dl.draw_text(
            te,
            rrect.x + 26.0,
            rrect.y + 6.0,
            13.5,
            if is_active || is_sel { pal.fg } else { pal.fg_prose },
            if is_active || is_sel {
                FontWeight::Bold
            } else {
                FontWeight::Semibold
            },
            Some((row_w - text_right_pad).max(120.0)),
            1,
            label,
        );
        dl.draw_text(
            te,
            rrect.x + 26.0,
            rrect.y + 24.0,
            11.5,
            pal.fg_dim,
            FontWeight::Regular,
            Some((row_w - text_right_pad).max(120.0)),
            1,
            &choice.detail,
        );

        hits.push(HitZone {
            rect: rrect,
            target: HitTarget::PickModel(idx),
        });
        if let Some(job) = job {
            draw_download_status(
                dl,
                te,
                hits,
                tooltip,
                pal,
                snap,
                Rect::new(
                    rrect.x + rrect.w - transfer_width - 12.0,
                    ry,
                    transfer_width,
                    45.0,
                ),
                job,
            );
            continue;
        }

        let (badge, bcol) = if is_active {
            ("Active", pal.accent)
        } else if choice.is_local {
            if snap.inventory.contains(&choice.model_id) {
                ("Installed", pal.success)
            } else {
                ("Download", pal.fg_muted)
            }
        } else if snap.config.has_api_key(&choice.backend) {
            ("Ready", pal.success)
        } else {
            ("Configure", pal.warning)
        };
        let bweight = if is_active {
            FontWeight::Bold
        } else {
            FontWeight::Semibold
        };
        let bw = te.measure(badge, 12.0, bweight);
        let badge_right = if has_cfg_btn {
            rrect.x + rrect.w - 44.0
        } else {
            rrect.x + rrect.w - 14.0
        };
        dl.draw_text_vcentered(
            te,
            badge_right - bw,
            rrect.y + rrect.h * 0.5,
            12.0,
            bcol,
            bweight,
            None,
            badge,
        );

        // Centered Configure (Sliders) icon button for Cloud & Self-Hosted rows
        if has_cfg_btn {
            let cfg_rect = Rect::new(
                rrect.x + rrect.w - 34.0,
                rrect.y + (rrect.h - 26.0) * 0.5,
                26.0,
                26.0,
            );
            let cfg_hov = cfg_rect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
            if cfg_hov {
                dl.fill_rect(cfg_rect, 6.0, pal.bg_hover);
                *tooltip = Some((
                    cfg_rect,
                    if choice.backend == "custom" {
                        "Configure Self-Hosted URL, model & key"
                    } else {
                        "Configure API key & endpoint"
                    },
                ));
            }
            dl.draw_animated_icon(
                te,
                IconId::Sliders,
                cfg_rect,
                14.0,
                if cfg_hov { pal.fg } else { pal.fg_dim },
                snap.icon_time(cfg_hov),
            );
            hits.push(HitZone {
                rect: cfg_rect,
                target: HitTarget::OpenProviderConfig(choice.backend.clone()),
            });
        }
    }

    if has_scrollbar {
        draw_scrollbar(
            dl,
            hits,
            pal,
            ScrollList::Models,
            col_x + col_w - 7.0,
            list_y,
            (max_rows as f32 * row_h - 4.0).min(list_h),
            start,
            max_rows,
            choices.len(),
            snap.mouse_pos,
            snap.dragging_scrollbar,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn render_settings_view(
    dl: &mut DrawList,
    te: &mut TextEngine,
    hits: &mut Vec<HitZone>,
    tooltip: &mut Option<(Rect, &'static str)>,
    pal: &Palette,
    area: Rect,
    snap: &UiSnapshot<'_>,
) {
    let col_w = (area.w - 56.0).clamp(400.0, 700.0);
    let col_x = area.x + (area.w - col_w) * 0.5;
    dl.draw_text(
        te,
        col_x,
        area.y + 19.0,
        25.0,
        pal.fg,
        FontWeight::Bold,
        None,
        1,
        "Settings",
    );
    let viewport = Rect::new(area.x, area.y + 60.0, area.w, area.h - 60.0);
    let first_instance = dl.instances.len();
    let first_hit = hits.len();
    let previous_tooltip = *tooltip;
    let offset = snap.settings_scroll as f32;
    let mut y = viewport.y + 6.0 - offset;

    dl.draw_text_vcentered(
        te,
        col_x,
        y + 11.0,
        13.0,
        pal.fg_muted,
        FontWeight::Semibold,
        None,
        "Speech engines",
    );
    y += 27.0;

    let mut draw_row = |dl: &mut DrawList,
                        te: &mut TextEngine,
                        hits: &mut Vec<HitZone>,
                        label: &str,
                        desc: &str,
                        val: &str,
                        highlight: bool,
                        target: Option<HitTarget>| {
        let bweight = if highlight {
            FontWeight::Bold
        } else {
            FontWeight::Semibold
        };
        let btn_w = (te.measure(val, 12.5, bweight) + 28.0).clamp(94.0, col_w * 0.46);
        let bx = col_x + col_w - btn_w;
        let label_w = Some(bx - col_x - 12.0);
        dl.draw_text(
            te,
            col_x,
            y + 2.0,
            14.0,
            pal.fg,
            FontWeight::Semibold,
            label_w,
            1,
            label,
        );
        dl.draw_text(
            te,
            col_x,
            y + 21.0,
            12.0,
            pal.fg_dim,
            FontWeight::Regular,
            label_w,
            1,
            desc,
        );

        let is_pressed = target
            .as_ref()
            .is_some_and(|target| snap.pressed_target == Some(target));
        let dy = if is_pressed { 1.0 } else { 0.0 };
        let brect = Rect::new(bx, y + 4.0 + dy, btn_w, 30.0);
        let hov = target.is_some() && brect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
        let ht = if hov { snap.hover_t } else { 0.0 };
        let base_fill = pal.bg_hover.with_alpha(0);
        let hover_fill = if is_pressed {
            pal.bg_active
        } else {
            pal.bg_hover
        };
        let fill = base_fill.lerp(hover_fill, ht);
        let fg = if target.is_some() {
            pal.fg
        } else {
            pal.fg_muted
        };
        dl.fill_rect(brect, 6.0, fill);
        dl.draw_text_centered(te, brect, 12.5, fg, bweight, val);
        if let Some(target) = target {
            hits.push(HitZone {
                rect: Rect::new(bx, y + 4.0, btn_w, 30.0),
                target,
            });
        }

        y += 48.0;
        dl.fill_rect(Rect::new(col_x, y - 4.0, col_w, 1.0), 0.0, pal.seam);
        y += 6.0;
    };

    for (engine, models) in [
        (LocalEngineKind::WhisperCpp, "Runs Whisper speech models"),
        (
            LocalEngineKind::AudioCpp,
            "Runs Parakeet, Qwen ASR, and other local speech models",
        ),
    ] {
        let download = snap.download.filter(|job| {
            job.target == InstallTarget::Engine(engine)
                && !matches!(
                    job.state,
                    DownloadState::Completed | DownloadState::Canceled
                )
        });
        let (label, action) = if download.is_some() {
            ("View download", Some(HitTarget::ToggleDownloads))
        } else if snap.inventory.has_engine(engine) {
            ("Installed", None)
        } else {
            ("Install engine", Some(HitTarget::InstallEngine(engine)))
        };
        let description = download.map_or(models, |job| job.label());
        draw_row(
            dl,
            te,
            hits,
            engine.label(),
            description,
            label,
            false,
            action,
        );
    }

    draw_row(
        dl,
        te,
        hits,
        "Global shortcut",
        "How Ctrl + Shift + Space starts and stops recording",
        snap.config.hotkey_mode.label().trim_start_matches("Hotkey: "),
        false,
        Some(HitTarget::CycleHotkeyMode),
    );

    draw_row(
        dl,
        te,
        hits,
        "Text Delivery",
        "How the global shortcut delivers text",
        snap.config.delivery_mode.short_label(),
        false,
        Some(HitTarget::CycleDelivery),
    );

    draw_row(
        dl,
        te,
        hits,
        "Spoken Language",
        "Choose the language you usually dictate in",
        DictateConfig::language_label(&snap.config.language),
        snap.config.language != "auto",
        Some(HitTarget::CycleLanguage),
    );

    draw_row(
        dl,
        te,
        hits,
        "Voice Activity Auto-Stop",
        "Stop recording automatically after 600 ms of silence",
        if snap.config.vad_auto_stop { "On" } else { "Off" },
        snap.config.vad_auto_stop,
        Some(HitTarget::ToggleVad),
    );

    // Self-Hosted OpenAI-Compatible Server row
    let (_, custom_url, custom_model) = snap
        .config
        .resolve_commercial_auth("custom")
        .unwrap_or_else(|| {
            (
                String::new(),
                "http://127.0.0.1:8000/v1".to_string(),
                "Qwen/Qwen3-ASR-1.7B".to_string(),
            )
        });
    let short_url = custom_url
        .trim_start_matches("http://")
        .trim_start_matches("https://");
    draw_row(
        dl,
        te,
        hits,
        "Self-hosted server",
        &format!("{short_url}  ·  {custom_model}"),
        "Configure",
        snap.config.backend == "custom",
        Some(HitTarget::OpenProviderConfig("custom".to_string())),
    );

    // Commercial & Cloud STT Providers quick-configure strip (shows green dot when ready)
    {
        dl.draw_text(
            te,
            col_x,
            y + 2.0,
            14.0,
            pal.fg,
            FontWeight::Semibold,
            None,
            1,
            "Commercial & Cloud Providers",
        );
        dl.draw_text(
            te,
            col_x,
            y + 21.0,
            12.0,
            pal.fg_dim,
            FontWeight::Regular,
            None,
            1,
            "Manage provider credentials and endpoints",
        );
        y += 42.0;

        let quick_providers = [
            ("openai", "OpenAI"),
            ("groq", "Groq"),
            ("deepgram", "Deepgram"),
            ("elevenlabs", "ElevenLabs"),
            ("mistral", "Mistral"),
            ("gemini", "Gemini"),
            ("huggingface", "Hugging Face"),
            ("cartesia", "Cartesia"),
            ("soniox", "Soniox"),
            ("assemblyai", "AssemblyAI"),
        ];
        let mut px = col_x;
        let mut py = y;
        let pill_h = 28.0;
        for (pid, label) in quick_providers {
            let ready = snap.config.has_api_key(pid);
            let is_cur = snap.config.backend == pid;
            let weight = if is_cur {
                FontWeight::Bold
            } else {
                FontWeight::Semibold
            };
            let lw = te.measure(label, 12.0, weight);
            let pw = (lw + 30.0).max(74.0);
            if px + pw > col_x + col_w && px > col_x {
                px = col_x;
                py += pill_h + 6.0;
            }
            let target = HitTarget::OpenProviderConfig(pid.to_string());
            let is_pressed = snap.pressed_target == Some(&target);
            let prect = Rect::new(px, py + if is_pressed { 1.0 } else { 0.0 }, pw, pill_h);
            let hov = prect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
            let ht = if hov { snap.hover_t } else { 0.0 };
            let fill = if is_cur {
                pal.segment_active.lerp(pal.bg_active, ht)
            } else {
                pal.bg_hover.with_alpha(0).lerp(
                    if is_pressed { pal.bg_active } else { pal.bg_hover },
                    ht,
                )
            };
            dl.fill_rect(prect, 6.0, fill);
            let dot_col = if is_cur {
                pal.accent
            } else if ready {
                pal.success
            } else {
                pal.fg_dim.with_alpha(110)
            };
            dl.fill_circle(prect.x + 11.0, prect.y + prect.h * 0.5, 3.0, dot_col);
            let fg = if is_cur || hov || ready {
                pal.fg
            } else {
                pal.fg_muted
            };
            dl.draw_text_vcentered(
                te,
                prect.x + 20.0,
                prect.y + prect.h * 0.5,
                12.0,
                fg,
                weight,
                None,
                label,
            );
            hits.push(HitZone {
                rect: Rect::new(px, py, pw, pill_h),
                target,
            });
            px += pw + 6.0;
        }
        y = py + pill_h + 14.0;
        dl.fill_rect(Rect::new(col_x, y - 4.0, col_w, 1.0), 0.0, pal.seam);
        y += 6.0;
    }

    // Microphone selector + Refresh SVG button
    {
        dl.draw_text(
            te,
            col_x,
            y + 2.0,
            14.0,
            pal.fg,
            FontWeight::Semibold,
            None,
            1,
            "Microphone",
        );
        dl.draw_text(
            te,
            col_x,
            y + 21.0,
            12.0,
            pal.fg_dim,
            FontWeight::Regular,
            None,
            1,
            "Choose an input device",
        );

        let ref_pressed = snap.pressed_target == Some(&HitTarget::RefreshMicrophones);
        let ref_rect = Rect::new(
            col_x + col_w - 30.0,
            y + 4.0 + if ref_pressed { 1.0 } else { 0.0 },
            30.0,
            30.0,
        );
        let ref_hov = ref_rect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
        let ref_ht = if ref_hov { snap.hover_t } else { 0.0 };
        if ref_hov {
            *tooltip = Some((ref_rect, "Refresh microphones"));
        }
        dl.fill_rect(
            ref_rect,
            8.0,
            pal.bg_hover.with_alpha(0).lerp(
                if ref_pressed { pal.bg_active } else { pal.bg_hover },
                ref_ht,
            ),
        );
        dl.draw_animated_icon(
            te,
            IconId::Refresh,
            ref_rect,
            14.0,
            if ref_hov { pal.fg } else { pal.fg_muted },
            snap.icon_time(ref_hov),
        );
        hits.push(HitZone {
            rect: Rect::new(col_x + col_w - 30.0, y + 4.0, 30.0, 30.0),
            target: HitTarget::RefreshMicrophones,
        });

        let mic_label = snap.active_mic_name;
        let btn_w = (te.measure(mic_label, 12.5, FontWeight::Semibold) + 28.0).clamp(116.0, 260.0);
        let bx = col_x + col_w - 36.0 - btn_w;
        let mic_pressed = snap.pressed_target == Some(&HitTarget::CycleMicrophone);
        let brect = Rect::new(
            bx,
            y + 4.0 + if mic_pressed { 1.0 } else { 0.0 },
            btn_w,
            30.0,
        );
        let hov = brect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
        let ht = if hov { snap.hover_t } else { 0.0 };
        dl.fill_rect(
            brect,
            8.0,
            pal.bg_hover.with_alpha(0).lerp(
                if mic_pressed { pal.bg_active } else { pal.bg_hover },
                ht,
            ),
        );
        dl.draw_text_centered(te, brect, 12.5, pal.fg, FontWeight::Semibold, mic_label);
        hits.push(HitZone {
            rect: Rect::new(bx, y + 4.0, btn_w, 30.0),
            target: HitTarget::CycleMicrophone,
        });

        y += 48.0;
        dl.fill_rect(Rect::new(col_x, y - 4.0, col_w, 1.0), 0.0, pal.seam);
        y += 6.0;
    }

    // Appearance row with Sun/Moon Electron SVG icon (horizontally + vertically centered group)
    {
        dl.draw_text(
            te,
            col_x,
            y + 2.0,
            14.0,
            pal.fg,
            FontWeight::Semibold,
            None,
            1,
            "Appearance",
        );
        dl.draw_text(
            te,
            col_x,
            y + 21.0,
            12.0,
            pal.fg_dim,
            FontWeight::Regular,
            None,
            1,
            "The window closes to the system tray",
        );
        let is_dark = current_theme() == Theme::Dark;
        let lbl = if is_dark { "Dark" } else { "Light" };
        let lbl_w = te.measure(lbl, 12.5, FontWeight::Semibold);
        let group_w = 14.0 + 6.0 + lbl_w;
        let btn_w = (group_w + 26.0).max(94.0);
        let bx = col_x + col_w - btn_w;
        let th_pressed = snap.pressed_target == Some(&HitTarget::ToggleTheme);
        let brect = Rect::new(
            bx,
            y + 4.0 + if th_pressed { 1.0 } else { 0.0 },
            btn_w,
            30.0,
        );
        let hov = brect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
        let ht = if hov { snap.hover_t } else { 0.0 };
        dl.fill_rect(
            brect,
            8.0,
            pal.bg_hover.with_alpha(0).lerp(
                if th_pressed { pal.bg_active } else { pal.bg_hover },
                ht,
            ),
        );
        let gx = brect.x + (brect.w - group_w) * 0.5;
        let cy = brect.y + brect.h * 0.5;
        dl.draw_icon(
            te,
            if is_dark { IconId::Moon } else { IconId::Sun },
            gx,
            cy - 7.0,
            14.0,
            if hov { pal.fg } else { pal.fg_muted },
        );
        dl.draw_text_vcentered(
            te,
            gx + 20.0,
            cy,
            12.5,
            pal.fg,
            FontWeight::Semibold,
            None,
            lbl,
        );
        hits.push(HitZone {
            rect: Rect::new(bx, y + 4.0, btn_w, 30.0),
            target: HitTarget::ToggleTheme,
        });
    }

    dl.clip_instances_from(first_instance, viewport);
    let bottom = viewport.y + viewport.h;
    for hit in &mut hits[first_hit..] {
        let top = hit.rect.y.max(viewport.y);
        let end = (hit.rect.y + hit.rect.h).min(bottom);
        hit.rect.y = top;
        hit.rect.h = (end - top).max(0.0);
    }
    hits.retain(|hit| hit.rect.h > 0.0);
    if !viewport.contains(snap.mouse_pos.0, snap.mouse_pos.1) {
        *tooltip = previous_tooltip;
    }
    let total = (y + 46.0 + offset - viewport.y).ceil() as usize;
    let visible = viewport.h.floor() as usize;
    draw_scrollbar(
        dl,
        hits,
        pal,
        ScrollList::Settings,
        area.x + area.w - 12.0,
        viewport.y,
        viewport.h,
        snap.settings_scroll,
        visible,
        total,
        snap.mouse_pos,
        snap.dragging_scrollbar,
    );
}

#[allow(clippy::too_many_arguments)]
fn draw_modal_input_field(
    dl: &mut DrawList,
    te: &mut TextEngine,
    hits: &mut Vec<HitZone>,
    pal: &Palette,
    snap: &UiSnapshot<'_>,
    x: f32,
    y: f32,
    w: f32,
    label: &str,
    placeholder: &str,
    value: &str,
    masked: bool,
    focused: bool,
    field: ProviderField,
) -> f32 {
    dl.draw_text(
        te,
        x,
        y,
        12.0,
        if focused { pal.fg } else { pal.fg_muted },
        FontWeight::Semibold,
        None,
        1,
        label,
    );
    let frect = Rect::new(x, y + 18.0, w, 34.0);
    let hov = frect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
    let border = if focused {
        pal.ring_focus
    } else if hov {
        pal.ring
    } else {
        pal.seam
    };
    dl.fill_rect_bordered(
        frect,
        6.0,
        pal.bg_panel,
        1.0,
        border,
    );

    let display = if value.is_empty() {
        String::new()
    } else if masked {
        if value.len() <= 8 {
            "*".repeat(value.len())
        } else {
            format!(
                "{}{}{}",
                &value[..4],
                "*".repeat(value.len().saturating_sub(8).min(24)),
                &value[value.len() - 4..]
            )
        }
    } else {
        value.to_string()
    };

    let tx = frect.x + 11.0;
    let cy = frect.y + frect.h * 0.5;
    let max_tw = frect.w - 22.0;

    if display.is_empty() {
        if focused && snap.caret_visible {
            dl.fill_rect(Rect::new(tx, cy - 8.0, 1.5, 16.0), 0.75, pal.fg);
        }
        dl.draw_text_vcentered(
            te,
            tx + if focused { 4.0 } else { 0.0 },
            cy,
            13.0,
            pal.fg_dim,
            FontWeight::Regular,
            Some(max_tw),
            placeholder,
        );
    } else {
        let drawn_w = dl.draw_text_vcentered(
            te,
            tx,
            cy,
            13.0,
            pal.fg,
            FontWeight::Regular,
            Some(max_tw),
            &display,
        );
        if focused && snap.caret_visible {
            let cx = (tx + drawn_w + 1.5).min(tx + max_tw);
            dl.fill_rect(Rect::new(cx, cy - 8.0, 1.5, 16.0), 0.75, pal.fg);
        }
    }

    hits.push(HitZone {
        rect: frect,
        target: HitTarget::FocusProviderField(field),
    });

    y + 62.0
}

#[allow(clippy::too_many_arguments)]
fn render_provider_config_modal(
    dl: &mut DrawList,
    te: &mut TextEngine,
    hits: &mut Vec<HitZone>,
    pal: &Palette,
    snap: &UiSnapshot<'_>,
    provider_id: &str,
    provider_name: &str,
    is_self_hosted: bool,
    base_url: &str,
    model_id: &str,
    api_key: &str,
    active_field: ProviderField,
    reveal_key: bool,
) {
    hits.clear();
    dl.fill_rect(Rect::new(0.0, 0.0, snap.width, snap.height), 0.0, pal.backdrop);
    hits.push(HitZone {
        rect: Rect::new(0.0, 0.0, snap.width, snap.height),
        target: HitTarget::CloseModal,
    });

    let mw = 540.0_f32.min(snap.width - 48.0);
    let mh = if is_self_hosted { 336.0 } else { 278.0 };
    let mx = (snap.width - mw) * 0.5;
    let my = (snap.height - mh) * 0.5;
    let mrect = Rect::new(mx, my, mw, mh);

    dl.fill_rect(mrect, 12.0, pal.bg_popover);
    // Catch clicks inside modal so they don't close it
    hits.push(HitZone {
        rect: mrect,
        target: HitTarget::FocusProviderField(active_field),
    });

    dl.fill_circle(mx + 22.0, my + 26.0, 4.5, pal.accent);
    let heading = if is_self_hosted {
        "Self-Hosted Speech Server".to_string()
    } else {
        format!("Configure {provider_name}")
    };
    dl.draw_text_vcentered(
        te,
        mx + 34.0,
        my + 26.0,
        15.5,
        pal.fg,
        FontWeight::Bold,
        None,
        &heading,
    );

    let spec = find_commercial_provider(provider_id);
    let env_hint = spec
        .and_then(|p| p.env_vars.first().copied())
        .unwrap_or("API_KEY");
    let default_url = spec
        .map(|p| p.default_base_url)
        .unwrap_or("http://127.0.0.1:8000/v1");
    let subtitle = if is_self_hosted {
        "OpenAI-compatible /v1/audio/transcriptions server (vLLM, Speaches, Sherpa-ONNX, faster-whisper)."
            .to_string()
    } else {
        format!("Tab switches fields · Ctrl+V pastes · env fallback: ${env_hint}")
    };
    dl.draw_text(
        te,
        mx + 22.0,
        my + 44.0,
        12.0,
        pal.fg_dim,
        FontWeight::Regular,
        Some(mw - 44.0),
        1,
        &subtitle,
    );

    let mut fy = my + 72.0;
    let fw = mw - 44.0;

    if is_self_hosted {
        fy = draw_modal_input_field(
            dl,
            te,
            hits,
            pal,
            snap,
            mx + 22.0,
            fy,
            fw,
            "Endpoint Base URL",
            default_url,
            base_url,
            false,
            active_field == ProviderField::BaseUrl,
            ProviderField::BaseUrl,
        );
        fy = draw_modal_input_field(
            dl,
            te,
            hits,
            pal,
            snap,
            mx + 22.0,
            fy,
            fw,
            "Model ID",
            "Qwen/Qwen3-ASR-1.7B",
            model_id,
            false,
            active_field == ProviderField::ModelId,
            ProviderField::ModelId,
        );
        let _ = draw_modal_input_field(
            dl,
            te,
            hits,
            pal,
            snap,
            mx + 22.0,
            fy,
            fw,
            "API Key (optional for localhost)",
            "Leave empty for local servers, or paste Bearer token…",
            api_key,
            !reveal_key,
            active_field == ProviderField::ApiKey,
            ProviderField::ApiKey,
        );
    } else {
        let key_label = if provider_id == "huggingface" {
            "Hugging Face Token (HF_TOKEN)"
        } else {
            "API Key"
        };
        fy = draw_modal_input_field(
            dl,
            te,
            hits,
            pal,
            snap,
            mx + 22.0,
            fy,
            fw,
            key_label,
            "Paste key (Ctrl+V)…",
            api_key,
            !reveal_key,
            active_field == ProviderField::ApiKey,
            ProviderField::ApiKey,
        );
        let _ = draw_modal_input_field(
            dl,
            te,
            hits,
            pal,
            snap,
            mx + 22.0,
            fy,
            fw,
            "Base URL Override (optional)",
            default_url,
            base_url,
            false,
            active_field == ProviderField::BaseUrl,
            ProviderField::BaseUrl,
        );
    }

    let btn_y = my + mh - 48.0;
    let save_w = draw_modal_button(
        dl,
        te,
        hits,
        pal,
        mx + 22.0,
        btn_y,
        32.0,
        "Save (Enter)",
        ModalBtnTone::Primary,
        HitTarget::SaveProviderConfig,
        snap.mouse_pos,
        snap.hover_t,
        snap.pressed_target,
    );
    let cancel_w = draw_modal_button(
        dl,
        te,
        hits,
        pal,
        mx + 22.0 + save_w + 10.0,
        btn_y,
        32.0,
        "Cancel (Esc)",
        ModalBtnTone::Secondary,
        HitTarget::CloseModal,
        snap.mouse_pos,
        snap.hover_t,
        snap.pressed_target,
    );
    let _ = draw_modal_button(
        dl,
        te,
        hits,
        pal,
        mx + 22.0 + save_w + 10.0 + cancel_w + 10.0,
        btn_y,
        32.0,
        if reveal_key { "Hide Key" } else { "Show Key" },
        ModalBtnTone::Secondary,
        HitTarget::ToggleRevealKey,
        snap.mouse_pos,
        snap.hover_t,
        snap.pressed_target,
    );
}

#[allow(clippy::too_many_arguments)]
fn render_confirm_download_modal(
    dl: &mut DrawList,
    te: &mut TextEngine,
    hits: &mut Vec<HitZone>,
    pal: &Palette,
    snap: &UiSnapshot<'_>,
    target: &InstallTarget,
) {
    hits.clear();
    let backdrop = Rect::new(0.0, 0.0, snap.width, snap.height);
    dl.fill_rect(backdrop, 0.0, pal.backdrop);
    hits.push(HitZone {
        rect: backdrop,
        target: HitTarget::CloseModal,
    });

    let model = target.model_id().and_then(find_local_whisper_spec);
    let model_ready = target
        .model_id()
        .is_some_and(|id| snap.inventory.contains(id));
    let (title, button, description, h) = match target {
        InstallTarget::Engine(_) => (
            "Install speech engine",
            "Install engine",
            "Downloads the executable and required libraries. You can choose a compatible speech model afterward.",
            184.0,
        ),
        InstallTarget::Model(_) => (
            if model_ready { "Set up model" } else { "Download model" },
            if model_ready { "Set up" } else { "Download" },
            "Once setup is complete, this model transcribes on your device.",
            249.0,
        ),
    };
    let w = 474.0_f32.min(snap.width - 48.0);
    let panel = Rect::new((snap.width - w) * 0.5, (snap.height - h) * 0.5, w, h);
    draw_floating_surface(dl, pal, panel, 12.0);
    hits.push(HitZone {
        rect: panel,
        target: HitTarget::KeepModalOpen,
    });

    let x = panel.x + 22.0;
    let icon = Rect::new(x, panel.y + 20.0, 39.0, 39.0);
    dl.fill_rect(icon, 8.0, pal.bg_element);
    draw_centered_icon(dl, te, IconId::Download, icon, 18.0, pal.fg);
    dl.draw_text_vcentered(
        te,
        x + 51.0,
        panel.y + 28.0,
        18.0,
        pal.fg,
        FontWeight::Bold,
        None,
        title,
    );
    dl.draw_text_vcentered(
        te,
        x + 51.0,
        panel.y + 50.0,
        13.0,
        pal.fg_muted,
        FontWeight::Regular,
        Some(w - 95.0),
        target.name(),
    );

    if let Some(model) = model {
        let details = Rect::new(x, panel.y + 78.0, w - 44.0, 72.0);
        dl.fill_rect(
            Rect::new(x + 12.0, details.y + 36.0, details.w - 24.0, 1.0),
            0.0,
            pal.seam,
        );
        let model_size = if model_ready {
            "Already downloaded".into()
        } else if model.size_mb >= 1000 {
            format!("About {:.1} GB", f64::from(model.size_mb) / 1000.0)
        } else {
            format!("About {} MB", model.size_mb)
        };
        let engine_status = if snap.inventory.has_engine(model.engine) {
            "installed"
        } else {
            "added with setup"
        };
        let engine_detail = format!("{} · {engine_status}", model.engine.label());
        for (index, (item, detail)) in [
            ("Model files", model_size.as_str()),
            ("Speech engine", engine_detail.as_str()),
        ]
        .into_iter()
        .enumerate()
        {
            let cy = details.y + 18.0 + index as f32 * 36.0;
            dl.draw_text_vcentered(
                te,
                x + 12.0,
                cy,
                12.5,
                pal.fg,
                FontWeight::Semibold,
                None,
                item,
            );
            let detail_w = te.measure(detail, 12.5, FontWeight::Regular);
            dl.draw_text_vcentered(
                te,
                x + details.w - 12.0 - detail_w,
                cy,
                12.5,
                pal.fg_muted,
                FontWeight::Regular,
                None,
                detail,
            );
        }
    }
    dl.draw_text(
        te,
        x,
        panel.y + if model.is_some() { 164.0 } else { 80.0 },
        12.5,
        pal.fg_muted,
        FontWeight::Regular,
        Some(w - 44.0),
        2,
        description,
    );

    let action_w = (te.measure(button, 12.5, FontWeight::Bold) + 28.0).max(84.0);
    let y = panel.y + h - 51.0;
    let action_x = panel.x + w - 22.0 - action_w;
    draw_modal_button(
        dl,
        te,
        hits,
        pal,
        action_x - 92.0,
        y,
        33.0,
        "Cancel",
        ModalBtnTone::Secondary,
        HitTarget::CloseModal,
        snap.mouse_pos,
        snap.hover_t,
        snap.pressed_target,
    );
    draw_modal_button(
        dl,
        te,
        hits,
        pal,
        action_x,
        y,
        33.0,
        button,
        ModalBtnTone::Primary,
        HitTarget::ConfirmInstall,
        snap.mouse_pos,
        snap.hover_t,
        snap.pressed_target,
    );
}

fn render_confirm_delete_modal(
    dl: &mut DrawList,
    te: &mut TextEngine,
    hits: &mut Vec<HitZone>,
    pal: &Palette,
    snap: &UiSnapshot<'_>,
    preview: &str,
) {
    hits.clear();
    dl.fill_rect(Rect::new(0.0, 0.0, snap.width, snap.height), 0.0, pal.backdrop);
    hits.push(HitZone {
        rect: Rect::new(0.0, 0.0, snap.width, snap.height),
        target: HitTarget::CloseModal,
    });

    let mw = 460.0_f32.min(snap.width - 48.0);
    let mh = 172.0;
    let mx = (snap.width - mw) * 0.5;
    let my = (snap.height - mh) * 0.5;
    let mrect = Rect::new(mx, my, mw, mh);

    dl.fill_rect(mrect, 12.0, pal.bg_popover);
    dl.fill_circle(mx + 22.0, my + 26.0, 4.5, pal.error);
    dl.draw_text_vcentered(
        te,
        mx + 34.0,
        my + 26.0,
        15.5,
        pal.fg,
        FontWeight::Bold,
        None,
        "Delete this clip?",
    );
    dl.draw_text(
        te,
        mx + 22.0,
        my + 52.0,
        13.0,
        pal.fg_dim,
        FontWeight::Regular,
        Some(mw - 44.0),
        2,
        preview,
    );

    let btn_y = my + mh - 48.0;
    let del_w = draw_modal_button(
        dl,
        te,
        hits,
        pal,
        mx + 22.0,
        btn_y,
        32.0,
        "Delete Clip (Enter)",
        ModalBtnTone::Danger,
        HitTarget::ConfirmDeleteClip,
        snap.mouse_pos,
        snap.hover_t,
        snap.pressed_target,
    );
    let _ = draw_modal_button(
        dl,
        te,
        hits,
        pal,
        mx + 22.0 + del_w + 10.0,
        btn_y,
        32.0,
        "Cancel (Esc)",
        ModalBtnTone::Secondary,
        HitTarget::CloseModal,
        snap.mouse_pos,
        snap.hover_t,
        snap.pressed_target,
    );
}

#[allow(clippy::too_many_arguments)]
fn render_transcript_modal(
    dl: &mut DrawList,
    te: &mut TextEngine,
    hits: &mut Vec<HitZone>,
    pal: &Palette,
    snap: &UiSnapshot<'_>,
    entry: &HistoryEntry,
    scroll: usize,
    copied: bool,
) {
    hits.clear();
    let backdrop = Rect::new(0.0, 0.0, snap.width, snap.height);
    dl.fill_rect(backdrop, 0.0, pal.backdrop);
    hits.push(HitZone {
        rect: backdrop,
        target: HitTarget::CloseModal,
    });
    let w = 640.0_f32.min(snap.width - 48.0);
    let h = (snap.height - 64.0).min(540.0);
    let panel = Rect::new((snap.width - w) * 0.5, (snap.height - h) * 0.5, w, h);
    dl.fill_rect(panel, 12.0, pal.bg_popover);
    hits.push(HitZone {
        rect: panel,
        target: HitTarget::KeepModalOpen,
    });

    let x = panel.x + 22.0;
    dl.draw_text_vcentered(
        te,
        x,
        panel.y + 30.0,
        18.0,
        pal.fg,
        FontWeight::Bold,
        None,
        "Transcript",
    );
    let date = entry
        .created_at
        .with_timezone(&chrono::Local)
        .format("%b %e, %Y · %H:%M");
    let meta = format!("{date}  ·  {:.1}s  ·  {}", entry.duration_secs, entry.model);
    dl.draw_text_vcentered(
        te,
        x,
        panel.y + 54.0,
        12.0,
        pal.fg_muted,
        FontWeight::Regular,
        Some(w - 44.0),
        &meta,
    );

    let body = Rect::new(x, panel.y + 75.0, w - 44.0, h - 139.0);
    dl.fill_rect(body, 8.0, pal.bg_panel);
    let text_w = body.w - 38.0;
    let font_size = 15.0;
    let visible = ((body.h - 24.0) / (font_size * 1.45)).floor() as usize;
    let total = te.measure_wrapped_lines(&entry.text, font_size, text_w, usize::MAX);
    let start = scroll.min(total.saturating_sub(visible));
    dl.draw_text_lines(
        te,
        body.x + 13.0,
        body.y + 11.0,
        font_size,
        pal.fg_prose,
        FontWeight::Regular,
        Some(text_w),
        start,
        visible,
        &entry.text,
    );
    if total > visible {
        draw_scrollbar(
            dl,
            hits,
            pal,
            ScrollList::Transcript,
            body.x + body.w - 12.0,
            body.y + 10.0,
            body.h - 20.0,
            start,
            visible,
            total,
            snap.mouse_pos,
            snap.dragging_scrollbar,
        );
    }

    let y = panel.y + panel.h - 47.0;
    let mut x = x;
    for (label, tone, target) in [
        (
            if copied { "Copied" } else { "Copy transcript" },
            ModalBtnTone::Primary,
            HitTarget::CopyHistory(entry.id.clone()),
        ),
        ("Close", ModalBtnTone::Secondary, HitTarget::CloseModal),
    ] {
        x += draw_modal_button(
            dl,
            te,
            hits,
            pal,
            x,
            y,
            32.0,
            label,
            tone,
            target,
            snap.mouse_pos,
            snap.hover_t,
            snap.pressed_target,
        ) + 8.0;
    }
}

/// Render the borderless, always-on-top Floating Speech Overlay Pill (`420x74`)
/// on top of any active input in the system.
#[allow(clippy::too_many_arguments)]
pub fn render_overlay_pill(
    dl: &mut DrawList,
    te: &mut TextEngine,
    width: f32,
    height: f32,
    config: &DictateConfig,
    from_hotkey: bool,
    phase: &DictationPhase,
    elapsed_secs: u64,
    audio_levels: [f32; 15],
    live_preview: &str,
    status: Option<&str>,
) {
    let pal = palette();
    let is_rec = matches!(phase, DictationPhase::Recording);
    let is_trans = matches!(phase, DictationPhase::Transcribing);
    let ring = match phase {
        DictationPhase::Recording => pal.accent,
        DictationPhase::Transcribing => pal.warning,
        DictationPhase::SetupRequired => pal.warning,
        DictationPhase::Error(_) => pal.error,
        _ => pal.ring,
    };

    let pill = Rect::new(0.0, 0.0, width, height);
    let radius = height * 0.5;
    dl.fill_rect_bordered(pill, radius, pal.bg_popover, 1.5, ring);

    let voice_energy: f32 = if is_rec {
        audio_levels.iter().copied().sum::<f32>() / audio_levels.len() as f32
    } else {
        0.0
    };

    // Pulsing recording orb with Electron Mic SVG icon (expands dynamically with speech)
    let orb_cx = pill.x + 34.0;
    let orb_cy = pill.y + pill.h * 0.5;
    let orb_col = if is_rec {
        pal.accent
    } else if is_trans {
        pal.warning
    } else {
        pal.fg_dim
    };
    if is_rec {
        let pulse_r = 16.0 + voice_energy * 8.0;
        let pulse_a = (22.0 + voice_energy * 64.0).min(96.0) as u8;
        dl.fill_circle(orb_cx, orb_cy, pulse_r, orb_col.with_alpha(pulse_a));
    }
    dl.fill_circle(orb_cx, orb_cy, 14.5, orb_col.with_alpha(48));
    dl.draw_icon(te, IconId::Mic, orb_cx - 8.0, orb_cy - 8.0, 16.0, orb_col);

    let (title, sub) = match phase {
        DictationPhase::Recording => {
            let m = elapsed_secs / 60;
            let s = elapsed_secs % 60;
            let hint = if !from_hotkey {
                "Click here to stop"
            } else {
                match config.hotkey_mode {
                    HotkeyMode::HoldToTalk => "Release the shortcut to stop",
                    HotkeyMode::SmartHoldOrTap => "Hold to talk, or tap again to stop",
                    HotkeyMode::RealTimeStreaming => "Live mode · press shortcut to stop",
                }
            };
            let sub = if live_preview.trim().is_empty() {
                hint.to_string()
            } else {
                live_preview.to_string()
            };
            (format!("Listening · {m}:{s:02}"), sub)
        }
        DictationPhase::Transcribing => ("Transcribing…".to_string(), config.active_summary()),
        DictationPhase::SetupRequired => (
            "Setup required".into(),
            format!("Open {} to finish model setup.", crate::APP_NAME),
        ),
        DictationPhase::Error(err) => ("Error".to_string(), err.clone()),
        DictationPhase::Idle => {
            let title = if live_preview.is_empty() {
                "No speech captured"
            } else {
                "Transcript ready"
            };
            (title.into(), status.unwrap_or_default().to_string())
        }
    };

    dl.draw_text(
        te,
        pill.x + 58.0,
        pill.y + 16.0,
        14.0,
        pal.fg,
        FontWeight::Bold,
        Some(pill.w - if is_rec || is_trans { 188.0 } else { 76.0 }),
        1,
        &title,
    );
    dl.draw_text(
        te,
        pill.x + 58.0,
        pill.y + 38.0,
        11.5,
        pal.fg_muted,
        FontWeight::Regular,
        Some(pill.w - if is_rec || is_trans { 188.0 } else { 76.0 }),
        1,
        &sub,
    );

    if !is_rec && !is_trans {
        return;
    }
    // 15-bar live speech-reactive equalizer (or smooth travelling wave while transcribing)
    let eq_right = pill.x + pill.w - 24.0;
    let eq_start = eq_right - (15.0 * 6.5);
    let eq_cy = pill.y + pill.h * 0.5;
    let now_sec = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| (d.as_secs_f64() % 60.0) as f32)
        .unwrap_or(0.0);
    for (i, &lvl) in audio_levels.iter().enumerate() {
        let (bar_lvl, col) = if is_rec {
            let v = lvl.clamp(0.0, 1.0);
            (v, pal.accent_soft.lerp(pal.accent, (0.35 + v * 0.85).min(1.0)))
        } else {
            let wave = 0.25 + 0.55 * ((now_sec * 6.0 - i as f32 * 0.45).sin() * 0.5 + 0.5);
            (wave, pal.warning)
        };
        let bh = (5.0 + bar_lvl * 30.0).min(36.0);
        let bx = eq_start + i as f32 * 6.5;
        dl.fill_rect(Rect::new(bx, eq_cy - bh * 0.5, 3.5, bh), 1.75, col);
    }
}
