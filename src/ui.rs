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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Modal {
    None,
    ApiKeyInput {
        provider_id: String,
        provider_name: String,
        buffer: String,
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
    InstallLocalEngine,
    CycleLanguage,
    CycleHotkeyMode,
    CycleMicrophone,
    RefreshMicrophones,
    ToggleVad,
    CycleDelivery,
    ToggleTheme,
    OpenApiKey(String),
    FocusModelSearch,
    ClearModelSearch,
    SetModelTab(ModelFilterTab),
    PickModel(usize),
    ActivateLocalModel(String),
    SelectHistory(usize),
    CopyHistory(usize),
    TypeHistory(usize),
    DeleteHistory(usize),
    Scrollbar {
        list: ScrollList,
        track_y: i32,
        track_h: i32,
        total: usize,
        visible: usize,
    },
    SaveApiKey,
    CloseModal,
}

impl HitTarget {
    pub fn is_text_input(&self) -> bool {
        matches!(self, Self::FocusModelSearch)
    }

    pub fn is_pointer(&self) -> bool {
        !matches!(self, Self::DragWindow | Self::FocusModelSearch)
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

pub fn render_main_window(
    dl: &mut DrawList,
    te: &mut TextEngine,
    snap: &UiSnapshot<'_>,
) -> Vec<HitZone> {
    let pal = palette();
    let mut hits = Vec::with_capacity(96);
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
        (IconId::Minus, HitTarget::MinimizeWindow, false),
        (IconId::Maximize, HitTarget::MaximizeWindow, false),
        (IconId::Close, HitTarget::CloseWindow, true),
    ];
    for (i, (icon, target, is_close)) in win_btns.into_iter().enumerate() {
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
        }
        dl.draw_icon(
            te,
            icon,
            bx + 14.0,
            12.0,
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
    dl.fill_rect_bordered(
        Rect::new(16.0, 12.0, 26.0, 26.0),
        7.0,
        Rgba::rgb(17, 17, 17),
        1.0,
        Rgba::rgba(255, 255, 255, 36),
    );
    dl.draw_icon(te, IconId::Brand, 18.0, 14.0, 22.0, Rgba::rgb(255, 255, 255));
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
        AppView::Dictate => render_dictate_view(dl, te, &mut hits, &pal, content, snap, runtime_ready),
        AppView::Models => render_models_view(dl, te, &mut hits, &pal, content, snap),
        AppView::Settings => render_settings_view(dl, te, &mut hits, &pal, content, snap),
    }

    // Subtle 1px outer border around the frameless window
    dl.fill_rect_bordered(
        Rect::new(0.0, 0.0, w, h),
        0.0,
        Rgba::transparent(),
        1.0,
        pal.seam,
    );

    // ── 4. API Key / HF Token Modal ──────────────────────────────────────────
    if let Modal::ApiKeyInput {
        provider_id,
        provider_name,
        buffer,
    } = snap.modal
    {
        render_api_key_modal(
            dl,
            te,
            &mut hits,
            &pal,
            snap,
            provider_id,
            provider_name,
            buffer,
        );
    }

    hits
}

fn render_dictate_view(
    dl: &mut DrawList,
    te: &mut TextEngine,
    hits: &mut Vec<HitZone>,
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

    let short_model = if snap.config.backend == "local" {
        format!("{} ▾", snap.config.local_model)
    } else {
        format!("{} ▾", snap.config.commercial_model)
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
        false,
        HitTarget::SwitchView(AppView::Models),
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
            HitTarget::OpenApiKey(snap.config.backend.clone())
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
        dl.draw_icon(
            te,
            IconId::Square,
            btn_cx - 7.5,
            btn_cy - 7.5,
            15.0,
            Rgba::rgb(255, 255, 255),
        );
    } else {
        dl.draw_icon(
            te,
            IconId::Mic,
            btn_cx - 8.5,
            btn_cy - 8.5,
            17.0,
            if btn_hov { pal.accent } else { pal.fg },
        );
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
        return;
    }

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

        // Inline quiet SVG icon actions on hover or selection
        if is_hov || is_sel {
            let mut ax = rrect.x + rrect.w - 8.0;
            for (icon, target, is_del) in [
                (IconId::Trash, HitTarget::DeleteHistory(idx), true),
                (IconId::Globe, HitTarget::TypeHistory(idx), false),
                (IconId::Copy, HitTarget::CopyHistory(idx), false),
            ] {
                let bw = 24.0;
                ax -= bw;
                let arect = Rect::new(ax, meta_y - 4.0, bw, 22.0);
                let ahov = arect.contains(snap.mouse_pos.0, snap.mouse_pos.1);
                if ahov {
                    dl.fill_rect(arect, 6.0, pal.bg_hover);
                }
                dl.draw_icon(
                    te,
                    icon,
                    ax + 5.0,
                    meta_y - 1.0,
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

fn render_models_view(
    dl: &mut DrawList,
    te: &mut TextEngine,
    hits: &mut Vec<HitZone>,
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
        }
        dl.draw_icon(
            te,
            IconId::Close,
            clear_rect.x + 5.0,
            clear_rect.y + 5.0,
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
            Some(row_w - 135.0),
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
            Some(row_w - 135.0),
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
            ("Set Key", pal.warning)
        };
        let bweight = if is_active {
            FontWeight::Bold
        } else {
            FontWeight::Semibold
        };
        let bw = te.measure(badge, 12.0, bweight);
        dl.draw_text_vcentered(
            te,
            rrect.x + rrect.w - bw - 14.0,
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

fn render_settings_view(
    dl: &mut DrawList,
    te: &mut TextEngine,
    hits: &mut Vec<HitZone>,
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

    let hf_set = !snap.config.resolve_hf_token().is_empty();
    draw_row(
        dl,
        te,
        hits,
        "Hugging Face Token (HF_TOKEN)",
        "Authenticates gated Hugging Face downloads & HF Serverless Inference",
        if hf_set { "Configured" } else { "Set HF Token" },
        !hf_set,
        HitTarget::OpenApiKey("huggingface".to_string()),
    );

    let active_cloud_pid = if snap.config.backend == "local" {
        "openai"
    } else {
        snap.config.backend.as_str()
    };
    let active_cloud_name = find_commercial_provider(active_cloud_pid)
        .map(|p| p.name)
        .unwrap_or(active_cloud_pid);
    let cloud_set = snap.config.has_api_key(active_cloud_pid);
    draw_row(
        dl,
        te,
        hits,
        &format!("Commercial Provider Key ({active_cloud_name})"),
        "Configure API key for OpenAI, Groq, Deepgram, ElevenLabs, Mistral, Cartesia, Soniox…",
        if cloud_set { "Configured" } else { "Set API Key" },
        false,
        HitTarget::OpenApiKey(active_cloud_pid.to_string()),
    );

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
        dl.draw_icon(
            te,
            IconId::Refresh,
            ref_rect.x + 8.0,
            ref_rect.y + 8.0,
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
fn render_api_key_modal(
    dl: &mut DrawList,
    te: &mut TextEngine,
    hits: &mut Vec<HitZone>,
    pal: &Palette,
    snap: &UiSnapshot<'_>,
    provider_id: &str,
    provider_name: &str,
    buffer: &str,
) {
    hits.clear();
    dl.fill_rect(Rect::new(0.0, 0.0, snap.width, snap.height), 0.0, pal.backdrop);

    let mw = 500.0_f32.min(snap.width - 48.0);
    let mh = 200.0;
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
        None,
        &format!("{provider_name} Key"),
    );

    let env_hint = find_commercial_provider(provider_id)
        .and_then(|p| p.env_vars.first().copied())
        .unwrap_or("API_KEY");
    dl.draw_text(
        te,
        mx + 22.0,
        my + 46.0,
        12.5,
        pal.fg_dim,
        FontWeight::Regular,
        None,
        1,
        &format!("Press Ctrl+V to paste, or set ${env_hint} in your environment."),
    );

    let field = Rect::new(mx + 22.0, my + 78.0, mw - 44.0, 38.0);
    dl.fill_rect_bordered(field, 8.0, pal.bg_element, 1.0, pal.accent);
    let masked = if buffer.is_empty() {
        "Paste key (Ctrl+V)…▏".to_string()
    } else if buffer.len() <= 8 {
        format!("{}▏", "*".repeat(buffer.len()))
    } else {
        format!(
            "{}{}{}▏",
            &buffer[..4],
            "*".repeat(buffer.len().saturating_sub(8).min(24)),
            &buffer[buffer.len() - 4..]
        )
    };
    dl.draw_text_vcentered(
        te,
        field.x + 12.0,
        field.y + field.h * 0.5,
        13.5,
        if buffer.is_empty() { pal.fg_dim } else { pal.fg },
        FontWeight::Regular,
        Some(field.w - 24.0),
        &masked,
    );

    let save_w = draw_chip(
        dl,
        te,
        hits,
        pal,
        mx + 22.0,
        my + mh - 48.0,
        32.0,
        "Save Key (Enter)",
        true,
        HitTarget::SaveApiKey,
        snap.mouse_pos,
        snap.hover_t,
        snap.pressed_target,
    );
    let _ = draw_chip(
        dl,
        te,
        hits,
        pal,
        mx + 22.0 + save_w + 10.0,
        my + mh - 48.0,
        32.0,
        "Cancel (Esc)",
        false,
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
