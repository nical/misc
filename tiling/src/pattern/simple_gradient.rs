use crate::tiling::{TilePosition, PatternData};
use crate::gpu::{GpuStore, GpuStoreHandle};
use crate::{Color, Point};
use crate::custom_pattern::*;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Gradient {
    pub from: Point,
    pub to: Point,
    pub color0: Color,
    pub color1: Color,
}

impl Gradient {
    pub fn write_gpu_data(&self, store: &mut GpuStore) -> SimpleGradient {
        let can_stretch_horizontally = self.from.x == self.to.y || self.color0 == self.color1;
        let is_opaque = self.color0.is_opaque() && self.color1.is_opaque();
        let color0 = self.color0.to_f32();
        let color1 = self.color1.to_f32();
    
        let handle = store.push(&[
            self.from.x, self.from.y, self.to.x, self.to.y,
            color0[0], color0[1], color0[2], color0[3],
            color1[0], color1[1], color1[2], color1[3],
        ]);
    
        SimpleGradient {
            handle,
            is_opaque,
            can_stretch_horizontally,
        }    
    }
}

pub type SimpleGradientBuilder = CustomPatternBuilder<SimpleGradient>;

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
