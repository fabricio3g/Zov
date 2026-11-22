//! Unified Agente Code Desktop color tokens for the native `wgpu` interface.
//!
//! Dark theme: `#181818` sidebar, `#1c1c1c` graphite canvas, `#232323` composer/cards,
//! `rgba(255,255,255,0.08)` subtle ring, `#e8793a` single brand orange accent.

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
    pub bg_element: Rgba,
    pub bg_hover: Rgba,
    pub bg_active: Rgba,
    pub seam: Rgba,
    pub ring: Rgba,
    pub backdrop: Rgba,
    pub fg: Rgba,
    pub fg_prose: Rgba,
    pub fg_muted: Rgba,
    pub fg_dim: Rgba,
    pub accent: Rgba,
    pub accent_soft: Rgba,
    pub success: Rgba,
    pub warning: Rgba,
    pub error: Rgba,
}

pub fn palette() -> Palette {
    match current_theme() {
        Theme::Dark => Palette {
            bg_sidebar: Rgba::rgb(23, 23, 23),     // #171717 (sits a step behind canvas)
            bg: Rgba::rgb(28, 28, 28),             // #1c1c1c
            bg_panel: Rgba::rgb(35, 35, 35),       // #232323
            bg_element: Rgba::rgb(44, 44, 44),     // #2c2c2c
            bg_hover: Rgba::rgb(52, 52, 52),       // #343434
            bg_active: Rgba::rgb(58, 58, 58),      // #3a3a3a
            seam: Rgba::rgba(255, 255, 255, 16),   // 6% divider seam
            ring: Rgba::rgba(255, 255, 255, 22),   // 8.5% composer ring
            backdrop: Rgba::rgba(10, 10, 10, 185), // modal backdrop
            fg: Rgba::rgb(250, 250, 250),          // #fafafa
            fg_prose: Rgba::rgb(230, 230, 230),    // #e6e6e6
            fg_muted: Rgba::rgb(174, 174, 174),    // #aeaeae
            fg_dim: Rgba::rgb(126, 126, 126),      // #7e7e7e
            accent: Rgba::rgb(232, 121, 58),       // #e8793a
            accent_soft: Rgba::rgba(232, 121, 58, 36),
            success: Rgba::rgb(108, 197, 138),     // #6cc58a
            warning: Rgba::rgb(229, 180, 84),      // #e5b454
            error: Rgba::rgb(240, 123, 114),       // #f07b72
        },
        Theme::Light => Palette {
            bg_sidebar: Rgba::rgb(241, 240, 236),  // #f1f0ec
            bg: Rgba::rgb(248, 247, 244),          // #f8f7f4
            bg_panel: Rgba::rgb(255, 255, 255),    // #ffffff
            bg_element: Rgba::rgb(236, 235, 230),  // #ecebe6
            bg_hover: Rgba::rgb(227, 226, 220),    // #e3e2dc
            bg_active: Rgba::rgb(220, 219, 212),   // #dcdbd4
            seam: Rgba::rgba(0, 0, 0, 18),
            ring: Rgba::rgba(0, 0, 0, 26),
            backdrop: Rgba::rgba(20, 20, 20, 120),
            fg: Rgba::rgb(23, 23, 23),             // #171717
            fg_prose: Rgba::rgb(38, 38, 38),       // #262626
            fg_muted: Rgba::rgb(92, 92, 92),       // #5c5c5c
            fg_dim: Rgba::rgb(138, 138, 138),      // #8a8a8a
            accent: Rgba::rgb(214, 98, 34),        // #d66222
            accent_soft: Rgba::rgba(214, 98, 34, 28),
            success: Rgba::rgb(28, 140, 74),       // #1c8c4a
            warning: Rgba::rgb(184, 126, 16),      // #b87e10
            error: Rgba::rgb(204, 52, 44),         // #cc342c
        },
    }
}
