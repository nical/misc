use crate::tiling::{TilePosition, PatternData};
use crate::gpu::{GpuStore, GpuStoreHandle};
use crate::{Color, Point};
use crate::custom_pattern::*;

pub type SimpleGradientBuilder = CustomPatternBuilder<SimpleGradient>;
pub type SimpleGradientRenderer = CustomPatternRenderer<SimpleGradient>;

pub fn add_gradient(store: &mut GpuStore, p0: Point, color0: Color, p1: Point, color1: Color) -> SimpleGradient {
    let is_mergeable = p1.x == p1.y || color0 == color1;
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
        is_mergeable,
    }
}
pub struct SimpleGradient {
    handle: GpuStoreHandle,
    is_opaque: bool,
    is_mergeable: bool,
}

impl SimpleGradient {
    pub fn new() -> Self {
        SimpleGradient {
            handle: GpuStoreHandle::INVALID,
            is_opaque: false,
            is_mergeable: false,
        }
    }

    pub fn new_renderer(
        device: &wgpu::Device,
        helper: &mut CustomPatterns,
    ) -> CustomPatternRenderer<Self> {
        let descriptor = CustomPatternDescriptor {
            name: "simple gradient",
            source: include_str!("../../shaders/pattern/simple_gradient.wgsl"),
            varyings: &[
                Varying { name: "position", kind: "vec2<f32>", interpolate: "perspective" },
                Varying { name: "color0", kind: "vec4<f32>", interpolate: "flat" },
                Varying { name: "color1", kind: "vec4<f32>", interpolate: "flat" },
                Varying { name: "dir_offset", kind: "vec3<f32>", interpolate: "flat" },
            ],
            extra_bind_groups: &[],
        };

        CustomPatternRenderer::new(device, helper, &descriptor, ())
    }
}

impl CustomPattern for SimpleGradient {
    type RenderData = ();

    fn is_opaque(&self) -> bool { self.is_opaque }

    fn is_mergeable(&self) -> bool { self.is_mergeable }

    fn new_tile(&mut self, _: TilePosition) -> PatternData {
        self.handle.to_u32()
    }
}
