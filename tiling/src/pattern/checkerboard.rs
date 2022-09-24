use lyon::math::Point;

use crate::gpu::{GpuStoreHandle, GpuStore};
use crate::tiling::{TilePosition, PatternData};
use crate::{Color};
use crate::custom_pattern::*;

pub type CheckerboardPatternBuilder = CustomPatternBuilder<CheckerboardPattern>;

pub fn add_checkerboard(gpu_store: &mut GpuStore, pattern: &Checkerboard) -> CheckerboardPattern {
    let is_opaque = pattern.color0.is_opaque() && pattern.color1.is_opaque();
    let color0 = pattern.color0.to_f32();
    let color1 = pattern.color1.to_f32();
    let handle = gpu_store.push(&[
        color0[0], color0[1], color0[2], color0[3],
        color1[0], color1[1], color1[2], color1[3],
        pattern.offset.x, pattern.offset.y,
        pattern.scale,
    ]);

    CheckerboardPattern { handle, is_opaque }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Checkerboard {
    pub color0: Color,
    pub color1: Color,
    pub offset: Point,
    pub scale: f32,
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

    pub fn create_pipelines(
        device: &wgpu::Device,
        helper: &mut CustomPatterns,
    ) -> TilePipelines {
        let descriptor = CustomPatternDescriptor {
            name: "checkerboard",
            source: include_str!("../../shaders/pattern/checkerboard.wgsl"),
            varyings:  &[
                Varying { name: "uv", kind: "vec2<f32>", interpolated: true },
                Varying { name: "color0", kind: "vec4<f32>", interpolated: false },
                Varying { name: "color1", kind: "vec4<f32>", interpolated: false },
            ],
            extra_bind_groups: &[],
        };

        helper.create_tile_render_pipelines(device, &descriptor)
    }
}

impl CustomPattern for CheckerboardPattern {
    type RenderData = ();

    fn is_opaque(&self) -> bool { self.is_opaque }

    fn new_tile(&mut self, _: TilePosition) -> PatternData {
            self.handle.to_u32()
    }
}
