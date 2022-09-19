use lyon::math::Point;

use crate::gpu::{GpuStoreHandle, GpuStore};
use crate::tiling::{TilePosition, PatternData};
use crate::{Color};
use crate::custom_pattern::*;

pub type CheckerboardPatternBuilder = CustomPatternBuilder<CheckerboardPattern>;
pub type CheckerboardPatternRenderer = CustomPatternRenderer<CheckerboardPattern>;

pub fn add_checkerboard(gpu_store: &mut GpuStore, color0: Color, color1: Color, offset: Point, scale: f32) -> CheckerboardPattern {
    let is_opaque = color0.is_opaque() && color1.is_opaque();
    let color0 = color0.to_f32();
    let color1 = color1.to_f32();
    let handle = gpu_store.push(&[
        color0[0], color0[1], color0[2], color0[3],
        color1[0], color1[1], color1[2], color1[3],
        offset.x, offset.y,
        scale,
    ]);

    CheckerboardPattern { handle, is_opaque }
}

pub struct CheckerboardPattern {
    handle: GpuStoreHandle,
    is_opaque: bool,
}

impl CheckerboardPattern {
    pub fn new() -> Self {
        CheckerboardPattern {
            handle: GpuStoreHandle::INVALID,
            is_opaque: false,
        }
    }

    pub fn new_renderer(
        device: &wgpu::Device,
        helper: &mut CustomPatterns,
    ) -> CustomPatternRenderer<Self> {
        let descriptor = CustomPatternDescriptor {
            name: "checkerboard",
            source: include_str!("../../shaders/pattern/checkerboard.wgsl"),
            varyings:  &[
                Varying { name: "uv", kind: "vec2<f32>", interpolate: "perspective" },
                Varying { name: "color0", kind: "vec4<f32>", interpolate: "flat" },
                Varying { name: "color1", kind: "vec4<f32>", interpolate: "flat" },
            ],
            extra_bind_groups: &[],
        };

        CustomPatternRenderer::new(device, helper, &descriptor, ())
    }
}

impl CustomPattern for CheckerboardPattern {
    type RenderData = ();

    fn is_opaque(&self) -> bool { self.is_opaque }

    fn new_tile(&mut self, _: TilePosition) -> PatternData {
            self.handle.to_u32()
    }
}
