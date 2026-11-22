//! Agente Code Desktop-inspired native `wgpu` UI layout.
//!
//! - Custom frameless window chrome with draggable title bar and minimal `─ □ ×` controls
//! - 2-pane layered layout: `#171717` left sidebar (`Dictate` / `Models` / `Settings`) + `#1c1c1c` canvas
//! - 28px-radius Composer Sheet with 1px subtle ring, live waveform, and single 34px round mic control
//! - Scannable Models & Settings pages (labels hard left, values/actions hard right)
//! - Borderless always-on-top Floating Speech Overlay Pill for system-wide `Ctrl+Shift+Space` dictation.

use crate::catalog::{
    all_stt_choices, find_commercial_provider, LocalEngineKind, UnifiedSttChoice,
    LOCAL_WHISPER_CATALOG,
};
use crate::config::{AppPaths, DictateConfig, HistoryEntry, HotkeyMode};
use crate::engine::{check_local_whisper_model, installed_local_models, is_local_engine_ready};
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
    Installing { label: String, progress: u8 },
    Error(String),
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
    ProviderConfig {
        provider_id: String,
        provider_name: String,
        is_self_hosted: bool,
        base_url: String,
        model_id: String,
        api_key: String,
        active_field: ProviderField,
        reveal_key: bool,
    },
    ConfirmDownload {
        model_id: String,
        label: String,
        size_mb: u32,
        engine_name: String,
    },
    ConfirmDeleteClip {
        index: usize,
        preview: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScrollList {
    Models,
    History,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HitTarget {
    DragWindow,
    MinimizeWindow,
    MaximizeWindow,
    CloseWindow,
    SwitchView(AppView),
    ToggleRecord,
    ToggleModelDropdown,
    CloseModelDropdown,
    SelectDropdownLocal(String),
    SelectDropdownCloud {
        backend: String,
        model_id: String,
    },
    InstallLocalEngine,
    CycleLanguage,
    CycleHotkeyMode,
    CycleMicrophone,
    RefreshMicrophones,
    ToggleVad,
    CycleDelivery,
    ToggleTheme,
    OpenProviderConfig(String),
    FocusProviderField(ProviderField),
    ToggleRevealKey,
    FocusModelSearch,
    ClearModelSearch,
    SetModelTab(ModelFilterTab),
    PickModel(usize),
    ActivateLocalModel(String),
    SelectHistory(usize),
    CopyHistory(usize),
    TypeHistory(usize),
    RequestDeleteHistory(usize),
    Scrollbar {
        list: ScrollList,
        track_y: i32,
        track_h: i32,
        total: usize,
        visible: usize,
    },
    SaveProviderConfig,
    ConfirmDownloadModel,
    ConfirmDeleteClip,
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
                | Self::CloseModelDropdown
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
    pub view: AppView,
    pub model_tab: ModelFilterTab,
    pub model_filter: &'a str,
    pub model_search_focused: bool,
    pub model_dropdown_open: bool,
    pub caret_visible: bool,
    pub model_selected: usize,
    pub model_scroll: usize,
    pub config: &'a DictateConfig,
    pub paths: &'a AppPaths,
    pub phase: &'a DictationPhase,
    pub elapsed_secs: u64,
    pub history_bars: &'a [f32],
    pub vad_speaking: bool,
    pub live_preview: &'a str,
    pub history: &'a [HistoryEntry],
    pub selected_history: usize,
    pub history_scroll: usize,
    pub status_toast: Option<&'a str>,
    pub modal: &'a Modal,
    pub active_mic_name: &'a str,
    pub mouse_pos: (f32, f32),
    pub hover_t: f32,
    pub pressed_target: Option<&'a HitTarget>,
    pub dragging_scrollbar: Option<ScrollList>,
}

pub fn filtered_stt_choices(
    paths: &AppPaths,
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
                        check_local_whisper_model(paths, &c.model_id).is_some()
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
        let is_installed = c.is_local && check_local_whisper_model(paths, &c.model_id).is_some();
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
    let thumb_h = (track_h * (visible as f32 / total as f32)).clamp(28.0, track_h);
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
    let w = (text_w + 24.0).max(52.0);
    let is_pressed = pressed.is_some_and(|p| p == &target);
    let dy = if is_pressed { 1.0 } else { 0.0 };
    let rect = Rect::new(x, y + dy, w, height);
    let hovered = rect.contains(mouse.0, mouse.1);
    let ht = if hovered { hover_t } else { 0.0 };

    let base_fill = if active {
        pal.accent_soft
    } else {
        pal.bg_hover.with_alpha(0)
    };
    let hover_fill = if active {
        pal.accent
    } else if is_pressed {
        pal.bg_active
    } else {
        pal.bg_hover
    };
    let fill = base_fill.lerp(hover_fill, ht);

    let base_fg = if active { pal.accent } else { pal.fg_muted };
    let hover_fg = if active {
        Rgba::rgb(255, 255, 255)
    } else {
        pal.fg
    };
    let fg = base_fg.lerp(hover_fg, ht);

    if fill.0[3] > 0 {
        let ring = if hovered && !active {
            pal.ring.with_alpha((f32::from(pal.ring.0[3]) * ht) as u8)
        } else {
            Rgba::transparent()
        };
        dl.fill_rect_bordered(rect, height * 0.5, fill, if hovered { 1.0 } else { 0.0 }, ring);
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
    dl.fill_rect_bordered(rect, 7.0, pal.bg_element, 1.0, pal.ring);
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
    let weight = match tone {
        ModalBtnTone::Secondary => FontWeight::Semibold,
        ModalBtnTone::Primary | ModalBtnTone::Danger => FontWeight::Bold,
    };
    let text_w = te.measure(label, 12.5, weight);
    let w = (text_w + 28.0).max(84.0);
    let is_pressed = pressed.is_some_and(|p| p == &target);
    let dy = if is_pressed { 1.0 } else { 0.0 };
    let rect = Rect::new(x, y + dy, w, height);
    let hovered = rect.contains(mouse.0, mouse.1);
    let ht = if hovered { hover_t } else { 0.0 };

    let (base_fill, hover_fill, base_fg, hover_fg, border) = match tone {
        ModalBtnTone::Primary => (
            pal.accent_soft,
            pal.accent,
            pal.accent,
            Rgba::rgb(255, 255, 255),
            pal.accent,
        ),
        ModalBtnTone::Danger => (
            pal.error.with_alpha(38),
            pal.error,
            pal.error,
            Rgba::rgb(255, 255, 255),
            pal.error,
        ),
        ModalBtnTone::Secondary => (
            pal.bg_element,
            if is_pressed { pal.bg_active } else { pal.bg_hover },
            pal.fg_muted,
            pal.fg,
            pal.ring,
        ),
    };
    let fill = base_fill.lerp(hover_fill, ht);
    let fg = base_fg.lerp(hover_fg, ht);
    dl.fill_rect_bordered(rect, 8.0, fill, 1.0, border);
    dl.draw_text_centered(te, rect, 12.5, fg, weight, label);
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

    // ── 1. Layered Surfaces: Sidebar + Main Canvas ───────────────────────────
    dl.fill_rect(Rect::new(0.0, 0.0, sidebar_w, h), 0.0, pal.bg_sidebar);
    dl.fill_rect(Rect::new(sidebar_w, 0.0, 1.0, h), 0.0, pal.seam);

    // Draggable custom title bar across the top
    hits.push(HitZone {
        rect: Rect::new(0.0, 0.0, w - 126.0, titlebar_h),
        target: HitTarget::DragWindow,
    });

    // Window controls (Electron SVG icons: minus, maximize-2, x) on top-right
    let win_btns = [
        (IconId::Minus, HitTarget::MinimizeWindow, false, "Minimize"),
        (IconId::Maximize, HitTarget::MaximizeWindow, false, "Maximize / Restore"),
        (IconId::Close, HitTarget::CloseWindow, true, "Hide to System Tray"),
    ];
    for (i, (icon, target, is_close, tip)) in win_btns.into_iter().enumerate() {
        let bx = w - 126.0 + i as f32 * 42.0;
        let rect = Rect::new(bx, 0.0, 42.0, titlebar_h);
        let hov = rect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
        if hov {
            dl.fill_rect(
                rect,
                0.0,
                if is_close {
                    Rgba::rgb(196, 43, 28)
                } else {
                    pal.bg_hover
                },
            );
            tooltip = Some((rect, tip));
        }
        draw_centered_icon(
            dl,
            te,
            icon,
            rect,
            14.0,
            if hov && is_close {
                Rgba::rgb(255, 255, 255)
            } else if hov {
                pal.fg
            } else {
                pal.fg_dim
            },
        );
        hits.push(HitZone { rect, target });
    }

    // ── 2. Sidebar Navigation ────────────────────────────────────────────────
    let brand_tile = Rect::new(16.0, 12.0, 26.0, 26.0);
    dl.fill_rect_bordered(
        brand_tile,
        7.0,
        Rgba::rgb(17, 17, 17),
        1.0,
        Rgba::rgba(255, 255, 255, 36),
    );
    draw_centered_icon(dl, te, IconId::Brand, brand_tile, 20.0, Rgba::rgb(255, 255, 255));
    dl.draw_text_vcentered(te, 50.0, 25.0, 15.0, pal.fg, FontWeight::Bold, None, "Enio");

    let nav_items = [
        (AppView::Dictate, IconId::Mic, "Dictate"),
        (AppView::Models, IconId::Cpu, "Models"),
        (AppView::Settings, IconId::Settings, "Settings"),
    ];
    for (i, (v, icon, label)) in nav_items.into_iter().enumerate() {
        let ny = 60.0 + i as f32 * 36.0;
        let nrect = Rect::new(12.0, ny, sidebar_w - 24.0, 32.0);
        let active = snap.view == v;
        let hov = nrect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
        if active || hov {
            dl.fill_rect(
                nrect,
                8.0,
                if active { pal.bg_element } else { pal.bg_hover },
            );
        }
        if active {
            dl.fill_rect(Rect::new(16.0, ny + 8.0, 3.0, 16.0), 1.5, pal.accent);
        }
        dl.draw_icon(
            te,
            icon,
            26.0,
            ny + 8.5,
            15.0,
            if active {
                pal.accent
            } else if hov {
                pal.fg
            } else {
                pal.fg_muted
            },
        );
        dl.draw_text_vcentered(
            te,
            48.0,
            ny + 16.0,
            13.5,
            if active || hov { pal.fg } else { pal.fg_muted },
            if active {
                FontWeight::Bold
            } else {
                FontWeight::Semibold
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
    let installed_ids = installed_local_models(snap.paths);
    if !installed_ids.is_empty() {
        let sec_y = 186.0;
        dl.draw_text(
            te,
            22.0,
            sec_y,
            11.0,
            pal.fg_dim,
            FontWeight::Bold,
            None,
            1,
            "INSTALLED MODELS",
        );
        for (idx, mid) in installed_ids.iter().take(6).enumerate() {
            let my = sec_y + 22.0 + idx as f32 * 28.0;
            if my + 26.0 > h - 90.0 {
                break;
            }
            let mrect = Rect::new(12.0, my, sidebar_w - 24.0, 26.0);
            let is_cur = snap.config.backend == "local" && &snap.config.local_model == mid;
            let hov = mrect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
            if is_cur || hov {
                dl.fill_rect(
                    mrect,
                    6.0,
                    if is_cur { pal.bg_element } else { pal.bg_hover },
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
                12.0,
                if is_cur || hov { pal.fg } else { pal.fg_muted },
                if is_cur {
                    FontWeight::Bold
                } else {
                    FontWeight::Semibold
                },
                Some(mrect.w - 28.0),
                mid,
            );
            hits.push(HitZone {
                rect: mrect,
                target: HitTarget::ActivateLocalModel(mid.clone()),
            });
        }
    }

    // Sidebar bottom status footer (scannable, quiet)
    let runtime_ready = if snap.config.backend == "local" {
        is_local_engine_ready(snap.paths, snap.config)
    } else {
        snap.config.has_api_key(&snap.config.backend)
    };
    let foot_y = h - 76.0;
    let status_rect = Rect::new(12.0, foot_y, sidebar_w - 24.0, 62.0);
    dl.fill_rect_bordered(status_rect, 10.0, pal.bg_panel, 1.0, pal.seam);
    dl.fill_circle(
        status_rect.x + 14.0,
        status_rect.y + 19.0,
        4.0,
        if runtime_ready { pal.success } else { pal.warning },
    );
    dl.draw_text_vcentered(
        te,
        status_rect.x + 24.0,
        status_rect.y + 19.0,
        12.5,
        pal.fg,
        FontWeight::Semibold,
        Some(status_rect.w - 32.0),
        &snap.config.active_summary(),
    );
    dl.draw_text(
        te,
        status_rect.x + 12.0,
        status_rect.y + 35.0,
        11.5,
        pal.fg_dim,
        FontWeight::Regular,
        Some(status_rect.w - 24.0),
        1,
        "Ctrl+Shift+Space anywhere",
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
        h - titlebar_h,
    );

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

    // Subtle 1px outer border around the frameless window
    dl.fill_rect_bordered(
        Rect::new(0.0, 0.0, w, h),
        0.0,
        Rgba::transparent(),
        1.0,
        pal.seam,
    );

    // Paint floating icon tooltip on top of main UI when no modal is blocking
    if matches!(snap.modal, Modal::None) {
        if let Some((anchor, tip_text)) = tooltip {
            draw_tooltip(dl, te, &pal, w, h, anchor, tip_text);
        }
    }

    // ── 4. Modals (Provider Config, Download Confirm, Delete Clip Confirm) ───
    match snap.modal {
        Modal::None => {}
        Modal::ProviderConfig {
            provider_id,
            provider_name,
            is_self_hosted,
            base_url,
            model_id,
            api_key,
            active_field,
            reveal_key,
        } => {
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
        }
        Modal::ConfirmDownload {
            label,
            size_mb,
            engine_name,
            ..
        } => {
            render_confirm_download_modal(
                dl,
                te,
                &mut hits,
                &pal,
                snap,
                label,
                *size_mb,
                engine_name,
            );
        }
        Modal::ConfirmDeleteClip { preview, .. } => {
            render_confirm_delete_modal(dl, te, &mut hits, &pal, snap, preview);
        }
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
    let is_rec = matches!(snap.phase, DictationPhase::Recording);

    // ── 28px-Radius Agente Composer Pill ─────────────────────────────────────
    let pill_y = area.y + 18.0;
    let pill_h = 132.0;
    let pill = Rect::new(col_x, pill_y, col_w, pill_h);
    dl.fill_rect_bordered(
        pill,
        28.0,
        pal.bg_panel,
        1.0,
        if is_rec { pal.accent } else { pal.ring },
    );

    // Top section of composer sheet: waveform while recording, calm prompt when idle
    if is_rec {
        let mins = snap.elapsed_secs / 60;
        let secs = snap.elapsed_secs % 60;
        dl.fill_circle(pill.x + 26.0, pill.y + 30.0, 4.5, pal.accent);
        let vad_note = if snap.vad_speaking {
            "  ·  voice active"
        } else {
            ""
        };
        dl.draw_text_vcentered(
            te,
            pill.x + 38.0,
            pill.y + 30.0,
            14.5,
            pal.fg,
            FontWeight::Bold,
            None,
            &format!("Listening  ·  {mins}:{secs:02}{vad_note}"),
        );

        let wave_x = pill.x + 22.0;
        let wave_w = pill.w - 44.0;
        let wave_cy = pill.y + 62.0;
        let num_bars = HISTORY_BARS.min((wave_w / 7.5).floor() as usize).max(18);
        let step = wave_w / num_bars as f32;
        let bar_w = (step * 0.5).clamp(2.5, 4.0);
        let slice_start = snap.history_bars.len().saturating_sub(num_bars);
        for (i, &val) in snap.history_bars[slice_start..].iter().enumerate() {
            let v = val.clamp(0.0, 1.0);
            let bh = (4.0 + v * 26.0).min(30.0);
            let bx = wave_x + i as f32 * step;
            let col = pal.accent_soft.lerp(pal.accent, (v * 1.4).clamp(0.0, 1.0));
            dl.fill_rect(
                Rect::new(bx, wave_cy - bh * 0.5, bar_w, bh),
                bar_w * 0.5,
                col,
            );
        }
    } else {
        let (headline, headline_color) = match snap.phase {
            DictationPhase::Transcribing => ("Transcribing speech…".to_string(), pal.warning),
            DictationPhase::Installing { label, progress } => {
                (format!("{label} ({progress}%)"), pal.accent)
            }
            DictationPhase::Error(err) => (err.clone(), pal.error),
            _ => {
                if !snap.live_preview.trim().is_empty() {
                    (snap.live_preview.to_string(), pal.fg_prose)
                } else {
                    (
                        "Press Ctrl+Shift+Space in any app, or click the mic to dictate…"
                            .to_string(),
                        pal.fg_dim,
                    )
                }
            }
        };
        dl.draw_text(
            te,
            pill.x + 22.0,
            pill.y + 24.0,
            15.5,
            headline_color,
            FontWeight::Regular,
            Some(pill.w - 44.0),
            2,
            &headline,
        );
    }

    // Bottom bar inside composer sheet: quiet 32px chips left, single 34px round mic right
    let bar_y = pill.y + pill_h - 44.0;
    let mut cx = pill.x + 14.0;

    let model_chip_x = cx;
    let short_model = if snap.config.backend == "local" {
        format!(
            "{} {}",
            snap.config.local_model,
            if snap.model_dropdown_open { "▴" } else { "▾" }
        )
    } else {
        format!(
            "{} {}",
            snap.config.commercial_model,
            if snap.model_dropdown_open { "▴" } else { "▾" }
        )
    };
    cx += draw_chip(
        dl,
        te,
        hits,
        pal,
        cx,
        bar_y,
        30.0,
        &short_model,
        snap.model_dropdown_open,
        HitTarget::ToggleModelDropdown,
        snap.mouse_pos,
        snap.hover_t,
        snap.pressed_target,
    ) + 4.0;

    cx += draw_chip(
        dl,
        te,
        hits,
        pal,
        cx,
        bar_y,
        30.0,
        &snap.config.language,
        false,
        HitTarget::CycleLanguage,
        snap.mouse_pos,
        snap.hover_t,
        snap.pressed_target,
    ) + 4.0;

    let delivery_short = match snap.config.delivery_mode {
        crate::config::DeliveryMode::PasteAndClipboard => "Auto-Paste",
        crate::config::DeliveryMode::TypeAndClipboard => "Auto-Type",
        crate::config::DeliveryMode::ClipboardOnly => "Clipboard",
    };
    cx += draw_chip(
        dl,
        te,
        hits,
        pal,
        cx,
        bar_y,
        30.0,
        delivery_short,
        false,
        HitTarget::CycleDelivery,
        snap.mouse_pos,
        snap.hover_t,
        snap.pressed_target,
    ) + 4.0;

    if !runtime_ready {
        let action_lbl = if snap.config.backend == "local" {
            "Install Engine"
        } else {
            "Set Key"
        };
        let action_target = if snap.config.backend == "local" {
            HitTarget::InstallLocalEngine
        } else {
            HitTarget::OpenProviderConfig(snap.config.backend.clone())
        };
        let _ = draw_chip(
            dl,
            te,
            hits,
            pal,
            cx,
            bar_y,
            30.0,
            action_lbl,
            true,
            action_target,
            snap.mouse_pos,
            snap.hover_t,
            snap.pressed_target,
        );
    }

    // 34px round Mic / Stop button on the hard right (with smooth hover lift & press depth)
    let btn_r = 17.0;
    let btn_cx = pill.x + pill.w - 30.0;
    let mic_pressed = snap.pressed_target == Some(&HitTarget::ToggleRecord);
    let btn_cy = bar_y + 15.0 + if mic_pressed { 1.0 } else { 0.0 };
    let btn_rect = Rect::new(btn_cx - btn_r, btn_cy - btn_r, btn_r * 2.0, btn_r * 2.0);
    let btn_hov = btn_rect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
    let ht = if btn_hov { snap.hover_t } else { 0.0 };
    let btn_fill = if is_rec {
        pal.accent
    } else if mic_pressed {
        pal.bg_active
    } else {
        pal.bg_element.lerp(pal.bg_active, ht)
    };
    if btn_hov && !is_rec {
        dl.fill_circle(btn_cx, btn_cy, btn_r + 2.0, pal.accent_soft.with_alpha((32.0 * ht) as u8));
    }
    dl.fill_circle(btn_cx, btn_cy, btn_r, btn_fill);
    if is_rec {
        draw_centered_icon(dl, te, IconId::Square, btn_rect, 14.0, Rgba::rgb(255, 255, 255));
    } else {
        draw_centered_icon(
            dl,
            te,
            IconId::Mic,
            btn_rect,
            16.0,
            if btn_hov { pal.accent } else { pal.fg },
        );
    }
    if btn_hov && !snap.model_dropdown_open {
        *tooltip = Some((
            btn_rect,
            if is_rec {
                "Stop & transcribe (Space)"
            } else {
                "Start dictation (Space)"
            },
        ));
    }
    hits.push(HitZone {
        rect: btn_rect,
        target: HitTarget::ToggleRecord,
    });

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
        &format!("Recent ({})", snap.history.len()),
    );
    if let Some(toast) = snap.status_toast {
        let tw = te.measure(toast, 12.5, FontWeight::Regular);
        dl.draw_text(
            te,
            col_x + col_w - tw - 4.0,
            list_header_y,
            12.5,
            pal.fg_dim,
            FontWeight::Regular,
            None,
            1,
            toast,
        );
    }

    let mut cur_y = list_header_y + 26.0;
    let bottom = area.y + area.h - 18.0;

    if snap.history.is_empty() {
        dl.draw_text(
            te,
            col_x + 4.0,
            cur_y + 6.0,
            13.5,
            pal.fg_dim,
            FontWeight::Regular,
            Some(col_w - 8.0),
            2,
            "Your dictations appear here. Use Ctrl+Shift+Space from any window to speak directly into editors, browsers, or chat inputs.",
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

            if is_sel || is_hov {
                dl.fill_rect(
                    rrect,
                    10.0,
                    if is_sel { pal.bg_panel } else { pal.bg_panel.with_alpha(150) },
                );
                dl.fill_rect(
                    Rect::new(rrect.x + 2.0, rrect.y + 10.0, 2.5, row_h - 20.0),
                    1.25,
                    pal.accent,
                );
            }
            hits.push(HitZone {
                rect: rrect,
                target: HitTarget::SelectHistory(idx),
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
                "{}  ·  {}  ·  {:.1}s  ·  {}ms",
                entry.created_at.format("%H:%M"),
                entry.model,
                entry.duration_secs,
                entry.elapsed_ms
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

            // Inline quiet SVG icon actions on hover or selection (mathematically centered + tooltips)
            if is_hov || is_sel {
                let mut ax = rrect.x + rrect.w - 8.0;
                for (icon, target, is_del, tip) in [
                    (IconId::Trash, HitTarget::RequestDeleteHistory(idx), true, "Delete clip"),
                    (IconId::Globe, HitTarget::TypeHistory(idx), false, "Type into active window"),
                    (IconId::Copy, HitTarget::CopyHistory(idx), false, "Copy to clipboard"),
                ] {
                    let bw = 26.0;
                    ax -= bw;
                    let arect = Rect::new(ax, meta_y - 5.0, bw, 24.0);
                    let ahov = arect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
                    if ahov {
                        dl.fill_rect(arect, 6.0, pal.bg_hover);
                        if !snap.model_dropdown_open {
                            *tooltip = Some((arect, tip));
                        }
                    }
                    draw_centered_icon(
                        dl,
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
    if snap.model_dropdown_open {
        render_model_dropdown(dl, te, hits, pal, snap, model_chip_x, bar_y + 34.0);
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

    // Click-outside catcher behind the dropdown panel
    hits.push(HitZone {
        rect: Rect::new(0.0, 0.0, snap.width, snap.height),
        target: HitTarget::CloseModelDropdown,
    });

    let installed = installed_local_models(snap.paths);
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
    dl.fill_rect_bordered(drect, 12.0, pal.bg_panel, 1.0, pal.ring);

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
        area.y + 12.0,
        18.0,
        pal.fg,
        FontWeight::Bold,
        None,
        1,
        "Speech Models",
    );
    dl.draw_text(
        te,
        col_x,
        area.y + 38.0,
        13.0,
        pal.fg_dim,
        FontWeight::Regular,
        None,
        1,
        "On-device Non-Whisper (Qwen3-ASR, Parakeet, SenseVoice, Moonshine), Whisper GGML, Hugging Face, and 15 Commercial STT APIs.",
    );

    // Interactive Search bar with focus ring, hover lift, blinking caret, and Clear (×) button
    let search_rect = Rect::new(col_x, area.y + 68.0, col_w, 36.0);
    let search_hov = search_rect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
    let search_focused = snap.model_search_focused;
    let search_fill = if search_focused {
        pal.bg_element
    } else if search_hov {
        pal.bg_panel.lerp(pal.bg_element, snap.hover_t)
    } else {
        pal.bg_panel
    };
    let search_border = if search_focused {
        pal.accent
    } else if search_hov {
        pal.fg_dim
    } else {
        pal.ring
    };
    dl.fill_rect_bordered(
        search_rect,
        9.0,
        search_fill,
        if search_focused { 1.5 } else { 1.0 },
        search_border,
    );
    dl.draw_icon(
        te,
        IconId::Search,
        search_rect.x + 12.0,
        search_rect.y + 10.5,
        15.0,
        if search_focused {
            pal.accent
        } else if search_hov {
            pal.fg_muted
        } else {
            pal.fg_dim
        },
    );

    let text_x = search_rect.x + 35.0;
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
            dl.fill_rect(Rect::new(cx, text_cy - 8.5, 1.5, 17.0), 0.75, pal.accent);
        }
    } else {
        if search_focused && snap.caret_visible {
            dl.fill_rect(Rect::new(text_x, text_cy - 8.5, 1.5, 17.0), 0.75, pal.accent);
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
        draw_centered_icon(
            dl,
            te,
            IconId::Close,
            clear_rect,
            14.0,
            if clear_hov { pal.fg } else { pal.fg_muted },
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
    let ty = area.y + 114.0;
    for tab in tabs {
        tx += draw_chip(
            dl,
            te,
            hits,
            pal,
            tx,
            ty,
            28.0,
            tab.label(),
            snap.model_tab == tab,
            HitTarget::SetModelTab(tab),
            snap.mouse_pos,
            snap.hover_t,
            snap.pressed_target,
        ) + 6.0;
    }

    // Scrollable model list + interactive scrollbar + scannable position indicator
    let choices = filtered_stt_choices(snap.paths, snap.config, snap.model_tab, snap.model_filter);
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

        let is_active = if choice.is_local {
            snap.config.backend == "local" && snap.config.local_model == choice.model_id
        } else {
            snap.config.backend == choice.backend
                && snap.config.commercial_model == choice.model_id
        };

        if is_sel || is_hov {
            dl.fill_rect(
                rrect,
                8.0,
                if is_sel { pal.bg_element } else { pal.bg_panel },
            );
        }

        if is_active {
            dl.fill_circle(rrect.x + 12.0, rrect.y + 20.0, 4.0, pal.accent);
        }

        let has_cfg_btn = !choice.is_local;
        let text_right_pad = if has_cfg_btn { 165.0 } else { 135.0 };

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
            &choice.label,
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

        let (badge, bcol) = if is_active {
            ("Active", pal.accent)
        } else if choice.is_local {
            if check_local_whisper_model(snap.paths, &choice.model_id).is_some() {
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

        hits.push(HitZone {
            rect: rrect,
            target: HitTarget::PickModel(idx),
        });

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
                dl.fill_rect_bordered(cfg_rect, 6.0, pal.bg_hover, 1.0, pal.ring);
                *tooltip = Some((
                    cfg_rect,
                    if choice.backend == "custom" {
                        "Configure Self-Hosted URL, model & key"
                    } else {
                        "Configure API key & endpoint"
                    },
                ));
            }
            draw_centered_icon(
                dl,
                te,
                IconId::Sliders,
                cfg_rect,
                14.0,
                if cfg_hov { pal.fg } else { pal.fg_dim },
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
    let mut y = area.y + 12.0;

    dl.draw_text(
        te,
        col_x,
        y,
        18.0,
        pal.fg,
        FontWeight::Bold,
        None,
        1,
        "Settings",
    );
    y += 38.0;

    let mut draw_row = |dl: &mut DrawList,
                        te: &mut TextEngine,
                        hits: &mut Vec<HitZone>,
                        label: &str,
                        desc: &str,
                        val: &str,
                        highlight: bool,
                        target: HitTarget| {
        dl.draw_text(
            te,
            col_x,
            y + 2.0,
            14.0,
            pal.fg,
            FontWeight::Semibold,
            None,
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
            None,
            1,
            desc,
        );

        let bweight = if highlight {
            FontWeight::Bold
        } else {
            FontWeight::Semibold
        };
        let btn_w = (te.measure(val, 12.5, bweight) + 28.0).max(94.0);
        let bx = col_x + col_w - btn_w;
        let is_pressed = snap.pressed_target == Some(&target);
        let dy = if is_pressed { 1.0 } else { 0.0 };
        let brect = Rect::new(bx, y + 4.0 + dy, btn_w, 30.0);
        let hov = brect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
        let ht = if hov { snap.hover_t } else { 0.0 };
        let base_fill = if highlight {
            pal.accent_soft
        } else {
            pal.bg_panel
        };
        let hover_fill = if highlight {
            pal.accent
        } else if is_pressed {
            pal.bg_active
        } else {
            pal.bg_hover
        };
        let fill = base_fill.lerp(hover_fill, ht);
        let fg = if highlight {
            pal.accent.lerp(Rgba::rgb(255, 255, 255), ht)
        } else {
            pal.fg
        };
        dl.fill_rect_bordered(
            brect,
            8.0,
            fill,
            1.0,
            if highlight { pal.accent } else { pal.ring },
        );
        dl.draw_text_centered(te, brect, 12.5, fg, bweight, val);
        hits.push(HitZone {
            rect: Rect::new(bx, y + 4.0, btn_w, 30.0),
            target,
        });

        y += 48.0;
        dl.fill_rect(Rect::new(col_x, y - 4.0, col_w, 1.0), 0.0, pal.seam);
        y += 6.0;
    };

    draw_row(
        dl,
        te,
        hits,
        "Global Hotkey Behavior (Ctrl+Shift+Space)",
        "Choose Release-to-Dictate, Smart Hold/Tap, or Real-Time Live Streaming",
        snap.config.hotkey_mode.label().trim_start_matches("Hotkey: "),
        false,
        HitTarget::CycleHotkeyMode,
    );

    draw_row(
        dl,
        te,
        hits,
        "Text Delivery",
        "How transcribed speech is inserted into the active application",
        snap.config.delivery_mode.label(),
        false,
        HitTarget::CycleDelivery,
    );

    draw_row(
        dl,
        te,
        hits,
        "Spoken Language",
        "Auto-detect or pin a specific ISO-639-1 language code",
        &snap.config.language,
        snap.config.language != "auto",
        HitTarget::CycleLanguage,
    );

    draw_row(
        dl,
        te,
        hits,
        "Voice Activity Auto-Stop",
        "Stop recording automatically after 600 ms of silence",
        if snap.config.vad_auto_stop { "On" } else { "Off" },
        snap.config.vad_auto_stop,
        HitTarget::ToggleVad,
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
        "Self-Hosted Server (vLLM / Speaches / Sherpa-ONNX)",
        &format!("{short_url}  ·  {custom_model}"),
        "Configure",
        snap.config.backend == "custom",
        HitTarget::OpenProviderConfig("custom".to_string()),
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
            "Click any provider to configure its API key, base URL override, or default model",
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
                pal.accent_soft.lerp(pal.accent, ht)
            } else {
                pal.bg_panel.lerp(
                    if is_pressed { pal.bg_active } else { pal.bg_hover },
                    ht,
                )
            };
            let border = if is_cur { pal.accent } else { pal.ring };
            dl.fill_rect_bordered(prect, 8.0, fill, 1.0, border);
            let dot_col = if is_cur && hov {
                Rgba::rgb(255, 255, 255)
            } else if ready {
                pal.success
            } else {
                pal.fg_dim.with_alpha(110)
            };
            dl.fill_circle(prect.x + 11.0, prect.y + prect.h * 0.5, 3.0, dot_col);
            let fg = if is_cur {
                pal.accent.lerp(Rgba::rgb(255, 255, 255), ht)
            } else if hov || ready {
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
            "Click device name to cycle detected microphones, or refresh",
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
        dl.fill_rect_bordered(
            ref_rect,
            8.0,
            pal.bg_panel.lerp(
                if ref_pressed { pal.bg_active } else { pal.bg_hover },
                ref_ht,
            ),
            1.0,
            pal.ring,
        );
        draw_centered_icon(
            dl,
            te,
            IconId::Refresh,
            ref_rect,
            14.0,
            if ref_hov { pal.fg } else { pal.fg_muted },
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
        dl.fill_rect_bordered(
            brect,
            8.0,
            pal.bg_panel.lerp(
                if mic_pressed { pal.bg_active } else { pal.bg_hover },
                ht,
            ),
            1.0,
            pal.ring,
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
            "Closing the window hides Enio to the Activity Bar (System Tray)",
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
        dl.fill_rect_bordered(
            brect,
            8.0,
            pal.bg_panel.lerp(
                if th_pressed { pal.bg_active } else { pal.bg_hover },
                ht,
            ),
            1.0,
            pal.ring,
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
        pal.accent
    } else if hov {
        pal.fg_dim
    } else {
        pal.ring
    };
    dl.fill_rect_bordered(
        frect,
        8.0,
        pal.bg_element,
        if focused { 1.5 } else { 1.0 },
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
            dl.fill_rect(Rect::new(tx, cy - 8.0, 1.5, 16.0), 0.75, pal.accent);
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
            dl.fill_rect(Rect::new(cx, cy - 8.0, 1.5, 16.0), 0.75, pal.accent);
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

    dl.fill_rect_bordered(mrect, 18.0, pal.bg_panel, 1.0, pal.ring);
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
    label: &str,
    size_mb: u32,
    engine_name: &str,
) {
    hits.clear();
    dl.fill_rect(Rect::new(0.0, 0.0, snap.width, snap.height), 0.0, pal.backdrop);
    hits.push(HitZone {
        rect: Rect::new(0.0, 0.0, snap.width, snap.height),
        target: HitTarget::CloseModal,
    });

    let mw = 480.0_f32.min(snap.width - 48.0);
    let mh = 176.0;
    let mx = (snap.width - mw) * 0.5;
    let my = (snap.height - mh) * 0.5;
    let mrect = Rect::new(mx, my, mw, mh);

    dl.fill_rect_bordered(mrect, 18.0, pal.bg_panel, 1.0, pal.ring);
    dl.fill_circle(mx + 22.0, my + 26.0, 4.5, pal.accent);
    dl.draw_text_vcentered(
        te,
        mx + 34.0,
        my + 26.0,
        15.5,
        pal.fg,
        FontWeight::Bold,
        Some(mw - 56.0),
        "Download Speech Model?",
    );
    dl.draw_text(
        te,
        mx + 22.0,
        my + 50.0,
        13.5,
        pal.fg_prose,
        FontWeight::Semibold,
        Some(mw - 44.0),
        1,
        label,
    );
    dl.draw_text(
        te,
        mx + 22.0,
        my + 74.0,
        12.5,
        pal.fg_dim,
        FontWeight::Regular,
        Some(mw - 44.0),
        2,
        &format!(
            "Downloads ~{size_mb} MB model weights (plus {engine_name} runtime if not yet installed) to your local Enio folder."
        ),
    );

    let btn_y = my + mh - 48.0;
    let dl_w = draw_modal_button(
        dl,
        te,
        hits,
        pal,
        mx + 22.0,
        btn_y,
        32.0,
        &format!("Download (~{size_mb} MB)"),
        ModalBtnTone::Primary,
        HitTarget::ConfirmDownloadModel,
        snap.mouse_pos,
        snap.hover_t,
        snap.pressed_target,
    );
    let _ = draw_modal_button(
        dl,
        te,
        hits,
        pal,
        mx + 22.0 + dl_w + 10.0,
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

    dl.fill_rect_bordered(mrect, 18.0, pal.bg_panel, 1.0, pal.ring);
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

/// Render the borderless, always-on-top Floating Speech Overlay Pill (`420x74`)
/// on top of any active input in the system.
#[allow(clippy::too_many_arguments)]
pub fn render_overlay_pill(
    dl: &mut DrawList,
    te: &mut TextEngine,
    width: f32,
    height: f32,
    config: &DictateConfig,
    phase: &DictationPhase,
    elapsed_secs: u64,
    audio_levels: [f32; 15],
    live_preview: &str,
) {
    let pal = palette();
    let is_rec = matches!(phase, DictationPhase::Recording);
    let is_trans = matches!(phase, DictationPhase::Transcribing);
    let ring = match phase {
        DictationPhase::Recording => pal.accent,
        DictationPhase::Transcribing => pal.warning,
        DictationPhase::Error(_) => pal.error,
        _ => pal.ring,
    };

    let pill = Rect::new(0.0, 0.0, width, height);
    let radius = height * 0.5;
    dl.fill_rect_bordered(pill, radius, pal.bg_panel, 1.5, ring);

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
            let hint = if config.delivery_mode == crate::config::DeliveryMode::ClipboardOnly {
                "Clipboard Only · switch to Auto-Paste to insert"
            } else {
                match config.hotkey_mode {
                    HotkeyMode::HoldToTalk => "Release Ctrl+Shift+Space to insert",
                    HotkeyMode::SmartHoldOrTap => "Release keys (or tap again) to insert",
                    HotkeyMode::RealTimeStreaming => "Live streaming · pauses auto-insert",
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
        DictationPhase::Installing { label, progress } => {
            (format!("Installing ({progress}%)"), label.clone())
        }
        DictationPhase::Error(err) => ("Error".to_string(), err.clone()),
        DictationPhase::Idle => ("Ready".to_string(), config.active_summary()),
    };

    dl.draw_text(
        te,
        pill.x + 58.0,
        pill.y + 16.0,
        14.0,
        pal.fg,
        FontWeight::Bold,
        Some(pill.w - 188.0),
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
        Some(pill.w - 188.0),
        1,
        &sub,
    );

    // 15-bar live speech-reactive equalizer (or smooth travelling wave while transcribing)
    let eq_right = pill.x + pill.w - 24.0;
    let eq_start = eq_right - (15.0 * 6.5);
    let eq_cy = pill.y + pill.h * 0.5;
    let now_sec = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f32())
        .unwrap_or(0.0);
    for (i, &lvl) in audio_levels.iter().enumerate() {
        let (bar_lvl, col) = if is_rec {
            let v = lvl.clamp(0.0, 1.0);
            (v, pal.accent_soft.lerp(pal.accent, (0.35 + v * 0.85).min(1.0)))
        } else if is_trans {
            let wave = 0.25 + 0.55 * ((now_sec * 6.0 - i as f32 * 0.45).sin() * 0.5 + 0.5);
            (wave, pal.warning)
        } else {
            (0.12, pal.bg_active)
        };
        let bh = (5.0 + bar_lvl * 30.0).min(36.0);
        let bx = eq_start + i as f32 * 6.5;
        dl.fill_rect(Rect::new(bx, eq_cy - bh * 0.5, 3.5, bh), 1.75, col);
    }
}
