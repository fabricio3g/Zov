use crate::gpu_ui::IconId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layer {
    Whole,
    Base,
    A,
    B,
    C,
}

impl Layer {
    pub fn includes(self, svg_line: &str) -> bool {
        let layer = if svg_line.contains("class=\"a\"") {
            Self::A
        } else if svg_line.contains("class=\"b\"") {
            Self::B
        } else if svg_line.contains("class=\"c\"") {
            Self::C
        } else {
            Self::Base
        };
        self == layer
    }
}

#[derive(Clone, Copy)]
pub struct Pose {
    pub offset: [f32; 2],
    pub scale: [f32; 2],
    pub rotation: f32,
    pub opacity: f32,
}

impl Pose {
    pub const STILL: Self = Self {
        offset: [0.0; 2],
        scale: [1.0; 2],
        rotation: 0.0,
        opacity: 1.0,
    };

    pub const fn translate(x: f32, y: f32) -> Self {
        Self {
            offset: [x, y],
            ..Self::STILL
        }
    }

    const fn scale(x: f32, y: f32) -> Self {
        Self {
            scale: [x, y],
            ..Self::STILL
        }
    }

    pub const fn rotate(degrees: f32) -> Self {
        Self {
            rotation: degrees,
            ..Self::STILL
        }
    }

    fn blend(self, to: Self, amount: f32) -> Self {
        let mix = |a, b| a + (b - a) * amount;
        Self {
            offset: std::array::from_fn(|axis| mix(self.offset[axis], to.offset[axis])),
            scale: std::array::from_fn(|axis| mix(self.scale[axis], to.scale[axis])),
            rotation: mix(self.rotation, to.rotation),
            opacity: mix(self.opacity, to.opacity),
        }
    }
}

pub const CENTER: [f32; 2] = [8.0, 8.0];
pub const HOVER_MS: u64 = 320;

/// The AgenteCode SVGs carry CSS animation rules. Reproduce their per-shape keyframes
/// here because resvg rasterizes the SVG geometry without evaluating animations.
pub struct Track {
    pub layer: Layer,
    pub pivot: [f32; 2],
    duration: f32,
    delay: f32,
    smooth: bool,
    frames: [(f32, Pose); 2],
}

impl Track {
    const fn new(layer: Layer, duration: f32, keyframe: (f32, Pose)) -> Self {
        Self {
            layer,
            pivot: CENTER,
            duration,
            delay: 0.0,
            smooth: false,
            frames: [keyframe, (1.0, Pose::STILL)],
        }
    }

    pub fn is_active(&self, elapsed_ms: f32) -> bool {
        elapsed_ms < self.duration + self.delay
    }

    pub fn sample(&self, elapsed_ms: f32) -> Pose {
        let raw = (elapsed_ms - self.delay) / self.duration;
        if !(0.0..1.0).contains(&raw) {
            return Pose::STILL;
        }
        let mut previous = (0.0, Pose::STILL);
        for (time, pose) in self.frames.into_iter().chain([(1.0, Pose::STILL)]) {
            if raw < time {
                let t = (raw - previous.0) / (time - previous.0);
                let eased = if self.smooth {
                    t * t * (3.0 - 2.0 * t)
                } else {
                    1.0 - (1.0 - t).powi(2)
                };
                return previous.1.blend(pose, eased);
            }
            previous = (time, pose);
        }
        Pose::STILL
    }
}

pub fn tracks(icon: IconId) -> &'static [Track] {
    use Layer::{Whole, A, B, C};
    match icon {
        IconId::Download => const { &[Track::new(A, 260.0, (0.5, Pose::translate(0.0, 1.5)))] },
        IconId::Minus => const { &[Track::new(Whole, 220.0, (0.5, Pose::scale(0.7, 1.0)))] },
        IconId::Maximize => {
            const {
                &[
                    Track::new(A, 240.0, (0.5, Pose::translate(1.0, -1.0))),
                    Track::new(B, 240.0, (0.5, Pose::translate(-1.0, 1.0))),
                ]
            }
        }
        IconId::Close => const { &[Track::new(Whole, 240.0, (0.0, Pose::rotate(-90.0)))] },
        IconId::Sun => {
            const {
                &[Track::new(
                    A,
                    280.0,
                    (
                        0.0,
                        Pose {
                            opacity: 0.4,
                            ..Pose::rotate(-45.0)
                        },
                    ),
                )]
            }
        }
        IconId::Moon => {
            const {
                &[Track {
                    pivot: [12.0, 4.0],
                    smooth: true,
                    ..Track::new(
                        A,
                        280.0,
                        (
                            0.5,
                            Pose {
                                rotation: 45.0,
                                opacity: 0.5,
                                ..Pose::scale(0.85, 0.85)
                            },
                        ),
                    )
                }]
            }
        }
        IconId::Mic => {
            const {
                &[Track {
                    pivot: [8.0, 6.0],
                    smooth: true,
                    ..Track::new(A, 260.0, (0.5, Pose::scale(1.1, 1.1)))
                }]
            }
        }
        IconId::Square => const { &[Track::new(Whole, 200.0, (0.5, Pose::scale(0.9, 0.9)))] },
        IconId::Settings => const { &[Track::new(A, 280.0, (0.0, Pose::rotate(-60.0)))] },
        IconId::Cpu => {
            const {
                &[Track::new(
                    A,
                    240.0,
                    (
                        0.5,
                        Pose {
                            opacity: 0.5,
                            ..Pose::scale(1.15, 1.15)
                        },
                    ),
                )]
            }
        }
        IconId::Search => {
            const {
                &[Track {
                    layer: Whole,
                    pivot: CENTER,
                    duration: 300.0,
                    delay: 0.0,
                    smooth: true,
                    frames: [
                        (0.33, Pose::translate(-1.0, -1.0)),
                        (0.66, Pose::translate(1.0, -1.0)),
                    ],
                }]
            }
        }
        IconId::Refresh => const { &[Track::new(Whole, 300.0, (0.0, Pose::rotate(-180.0)))] },
        IconId::Copy => {
            const {
                &[Track::new(
                    A,
                    260.0,
                    (
                        0.0,
                        Pose {
                            opacity: 0.4,
                            ..Pose::translate(-2.0, -2.0)
                        },
                    ),
                )]
            }
        }
        IconId::Trash => {
            const {
                &[Track {
                    pivot: [13.5, 4.5],
                    ..Track::new(
                        A,
                        280.0,
                        (
                            0.45,
                            Pose {
                                rotation: 14.0,
                                ..Pose::translate(0.0, -1.0)
                            },
                        ),
                    )
                }]
            }
        }
        IconId::Play => {
            const {
                &[Track::new(
                    Whole,
                    240.0,
                    (
                        0.4,
                        Pose {
                            offset: [1.5, 0.0],
                            ..Pose::scale(1.08, 1.08)
                        },
                    ),
                )]
            }
        }
        IconId::Pause => {
            const {
                &[
                    Track::new(A, 220.0, (0.5, Pose::scale(1.0, 0.85))),
                    Track {
                        delay: 60.0,
                        ..Track::new(B, 220.0, (0.5, Pose::scale(1.0, 0.85)))
                    },
                ]
            }
        }
        IconId::Sliders => {
            const {
                &[
                    Track {
                        smooth: true,
                        ..Track::new(A, 260.0, (0.5, Pose::translate(0.0, -1.5)))
                    },
                    Track {
                        delay: 20.0,
                        smooth: true,
                        ..Track::new(B, 260.0, (0.5, Pose::translate(0.0, 1.5)))
                    },
                    Track {
                        delay: 40.0,
                        smooth: true,
                        ..Track::new(C, 260.0, (0.5, Pose::translate(0.0, -1.5)))
                    },
                ]
            }
        }
        _ => &[],
    }
}
