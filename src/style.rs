//! Exact Agente Code Desktop (`app.css`) monochrome-first design tokens for native `wgpu`.
//!
//! Dark theme: `#1c1c1c` canvas, `#212121` chrome sidebar, `#232323` cards,
//! `#252527` composer sheet, `#272727` floating popovers/menus, `#ffffff` primary CTA
//! with `#000000` ink, `#ffffff14` hairline seams, and `#e8793a` brand orange reserved
//! strictly for live status/recording moments.

use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Theme {
    Dark = 0,
    Light = 1,
}

static CURRENT_THEME: AtomicU8 = AtomicU8::new(0);

pub fn set_theme(theme: Theme) {
    CURRENT_THEME.store(theme as u8, Ordering::Relaxed);
}

pub fn current_theme() -> Theme {
    match CURRENT_THEME.load(Ordering::Relaxed) {
        1 => Theme::Light,
        _ => Theme::Dark,
    }
}

pub fn toggle_theme() -> Theme {
    let next = match current_theme() {
        Theme::Dark => Theme::Light,
        Theme::Light => Theme::Dark,
    };
    set_theme(next);
    next
}

pub fn window_border_color(focused: bool) -> Rgba {
    match (current_theme(), focused) {
        (Theme::Dark, true) => Rgba::rgb(77, 77, 77),
        (Theme::Dark, false) => Rgba::rgb(55, 55, 55),
        (Theme::Light, true) => Rgba::rgb(142, 142, 142),
        (Theme::Light, false) => Rgba::rgb(186, 186, 186),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rgba(pub [u8; 4]);

impl Rgba {
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self([r, g, b, 255])
    }

    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self([r, g, b, a])
    }

    pub const fn transparent() -> Self {
        Self([0, 0, 0, 0])
    }

    pub fn with_alpha(self, a: u8) -> Self {
        Self([self.0[0], self.0[1], self.0[2], a])
    }

    pub fn lerp(self, other: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let inv = 1.0 - t;
        Self([
            (f32::from(self.0[0]) * inv + f32::from(other.0[0]) * t).round() as u8,
            (f32::from(self.0[1]) * inv + f32::from(other.0[1]) * t).round() as u8,
            (f32::from(self.0[2]) * inv + f32::from(other.0[2]) * t).round() as u8,
            (f32::from(self.0[3]) * inv + f32::from(other.0[3]) * t).round() as u8,
        ])
    }

    pub fn to_f32(self) -> [f32; 4] {
        [
            f32::from(self.0[0]) / 255.0,
            f32::from(self.0[1]) / 255.0,
            f32::from(self.0[2]) / 255.0,
            f32::from(self.0[3]) / 255.0,
        ]
    }
}

pub struct Palette {
    pub bg_sidebar: Rgba,
    pub bg: Rgba,
    pub bg_panel: Rgba,
    pub bg_composer: Rgba,
    pub bg_popover: Rgba,
    pub bg_element: Rgba,
    pub bg_hover: Rgba,
    pub bg_active: Rgba,
    pub segment_active: Rgba,
    pub seam: Rgba,
    pub ring: Rgba,
    pub popover_border: Rgba,
    pub popover_shadow: Rgba,
    pub ring_focus: Rgba,
    pub backdrop: Rgba,
    pub fg: Rgba,
    pub fg_prose: Rgba,
    pub fg_muted: Rgba,
    pub fg_dim: Rgba,
    pub primary: Rgba,
    pub primary_fg: Rgba,
    pub primary_hover: Rgba,
    pub accent: Rgba,
    pub accent_soft: Rgba,
    pub success: Rgba,
    pub warning: Rgba,
    pub error: Rgba,
}

pub fn palette() -> Palette {
    match current_theme() {
        Theme::Dark => Palette {
            bg_sidebar: Rgba::rgb(33, 33, 33),       // #212121 (--sidebar)
            bg: Rgba::rgb(28, 28, 28),               // #1c1c1c (--background)
            bg_panel: Rgba::rgb(35, 35, 35),         // #232323 (--card)
            bg_composer: Rgba::rgb(37, 37, 39),      // #252527 (--composer-surface)
            bg_popover: Rgba::rgb(39, 39, 39),       // #272727 (--popover)
            bg_element: Rgba::rgb(43, 43, 43),       // #2b2b2b (--muted / --user-bubble)
            bg_hover: Rgba::rgb(47, 47, 47),         // #2f2f2f (--surface-hover)
            bg_active: Rgba::rgb(54, 54, 54),        // #363636 (pressed / active fill)
            segment_active: Rgba::rgba(255, 255, 255, 34), // rgba(255,255,255,0.14) (--segment-active-bg)
            seam: Rgba::rgba(255, 255, 255, 20),     // #ffffff14 (--border)
            ring: Rgba::rgba(255, 255, 255, 26),     // rgba(255,255,255,0.10) (--composer-ring)
            popover_border: Rgba::rgba(255, 255, 255, 33),
            popover_shadow: Rgba::rgba(0, 0, 0, 125),
            ring_focus: Rgba::rgba(255, 255, 255, 62), // rgba(255,255,255,0.24) (--composer-ring-focus)
            backdrop: Rgba::rgba(0, 0, 0, 154),      // rgba(0,0,0,0.6) (--overlay)
            fg: Rgba::rgb(250, 250, 250),            // #fafafa (--foreground)
            fg_prose: Rgba::rgb(219, 219, 219),      // ~86% fg on #1c1c1c (--prose-ink)
            fg_muted: Rgba::rgb(179, 179, 179),      // #b3b3b3 (--muted-foreground)
            fg_dim: Rgba::rgb(132, 132, 132),        // #848484 (subtle caption/placeholder)
            primary: Rgba::rgb(255, 255, 255),       // #ffffff (--primary)
            primary_fg: Rgba::rgb(0, 0, 0),          // #000000 (--primary-foreground)
            primary_hover: Rgba::rgb(232, 232, 232), // #e8e8e8
            accent: Rgba::rgb(232, 121, 58),         // #e8793a (--brand-accent)
            accent_soft: Rgba::rgba(232, 121, 58, 34),
            success: Rgba::rgb(108, 197, 138),       // #6cc58a (--success)
            warning: Rgba::rgb(229, 180, 84),        // #e5b454 (--warning)
            error: Rgba::rgb(240, 123, 114),         // #f07b72 (--destructive)
        },
        Theme::Light => Palette {
            bg_sidebar: Rgba::rgb(255, 255, 255),    // #ffffff (--sidebar)
            bg: Rgba::rgb(250, 250, 250),            // #fafafa (--card / light canvas)
            bg_panel: Rgba::rgb(255, 255, 255),      // #ffffff (--popover)
            bg_composer: Rgba::rgb(255, 255, 255),   // #ffffff (--composer-surface)
            bg_popover: Rgba::rgb(255, 255, 255),    // #ffffff (--popover)
            bg_element: Rgba::rgb(243, 243, 244),    // #f3f3f4 (--user-bubble)
            bg_hover: Rgba::rgb(240, 240, 240),      // #f0f0f0
            bg_active: Rgba::rgb(232, 232, 232),     // #e8e8e8 (--accent-hover)
            segment_active: Rgba::rgb(255, 255, 255), // #ffffff (--segment-active-bg)
            seam: Rgba::rgba(0, 0, 0, 20),           // rgba(0,0,0,0.08) (--border)
            ring: Rgba::rgba(0, 0, 0, 24),           // rgba(0,0,0,0.09) (--composer-ring)
            popover_border: Rgba::rgba(0, 0, 0, 28),
            popover_shadow: Rgba::rgba(21, 24, 29, 49),
            ring_focus: Rgba::rgba(0, 0, 0, 68),     // rgba(0,0,0,0.26) (--composer-ring-focus)
            backdrop: Rgba::rgba(0, 0, 0, 90),       // rgba(0,0,0,0.35) (--overlay)
            fg: Rgba::rgb(0, 0, 0),                  // #000000 (--foreground)
            fg_prose: Rgba::rgb(23, 23, 23),         // #171717
            fg_muted: Rgba::rgb(82, 82, 82),         // #525252 (--muted-foreground)
            fg_dim: Rgba::rgb(115, 115, 115),        // #737373
            primary: Rgba::rgb(0, 0, 0),             // #000000 (--primary)
            primary_fg: Rgba::rgb(255, 255, 255),    // #ffffff (--primary-foreground)
            primary_hover: Rgba::rgb(23, 23, 23),    // #171717 (--primary-hover)
            accent: Rgba::rgb(231, 109, 31),         // #e76d1f (--brand-accent)
            accent_soft: Rgba::rgba(231, 109, 31, 26),
            success: Rgba::rgb(22, 163, 74),         // #16a34a (--success)
            warning: Rgba::rgb(202, 138, 4),         // #ca8a04 (--warning)
            error: Rgba::rgb(220, 38, 38),           // #dc2626 (--error)
        },
    }
}
