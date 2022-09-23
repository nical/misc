use crate::tiling::{TilePosition, PatternData};
use crate::gpu::{GpuStore, GpuStoreHandle};
use crate::{Color, Point};
use crate::custom_pattern::*;

pub type SimpleGradientBuilder = CustomPatternBuilder<SimpleGradient>;

pub fn add_gradient(store: &mut GpuStore, p0: Point, color0: Color, p1: Point, color1: Color) -> SimpleGradient {
    let can_stretch_horizontally = p1.x == p1.y || color0 == color1;
    let is_opaque = color0.is_opaque() && color1.is_opaque();
    let color0 = color0.to_f32();
    let color1 = color1.to_f32();

    let handle = store.push(&[
        p0.x, p0.y, p1.x, p1.y,
        color0[0], color0[1], color0[2], color0[3],
        color1[0], color1[1], color1[2], color1[3],
    ]);

    SimpleGradient {
        handle,
        is_opaque,
        can_stretch_horizontally,
    }
}
pub struct SimpleGradient {
    handle: GpuStoreHandle,
    is_opaque: bool,
    can_stretch_horizontally: bool,
}

impl SimpleGradient {
    pub fn new() -> Self {
        SimpleGradient {
            handle: GpuStoreHandle::INVALID,
            is_opaque: false,
            can_stretch_horizontally: false,
        }
    }

    pub fn create_pipelines(
        device: &wgpu::Device,
        helper: &mut CustomPatterns,
    ) -> TilePipelines {
        let descriptor = CustomPatternDescriptor {
            name: "simple gradient",
            source: include_str!("../../shaders/pattern/simple_gradient.wgsl"),
            varyings: &[
                Varying { name: "position", kind: "vec2<f32>", interpolated: true },
                Varying { name: "color0", kind: "vec4<f32>", interpolated: false },
                Varying { name: "color1", kind: "vec4<f32>", interpolated: false },
                Varying { name: "dir_offset", kind: "vec3<f32>", interpolated: false },
            ],
            extra_bind_groups: &[],
        };

        helper.create_tile_render_pipelines(device, &descriptor)
    }
}

impl CustomPattern for SimpleGradient {
    type RenderData = ();

    fn is_opaque(&self) -> bool { self.is_opaque }

    fn can_stretch_horizontally(&self) -> bool { self.can_stretch_horizontally }

    fn new_tile(&mut self, _: TilePosition) -> PatternData {
        self.handle.to_u32()
    }
}
