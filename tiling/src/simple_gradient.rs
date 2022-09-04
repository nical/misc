use crate::gpu_store::{GpuStore, GpuStoreHandle};
use crate::{Color, Point, TilePosition, PatternData};

use crate::custom_pattern::*;

pub type SimpleGradientBuilder = CustomPatternBuilder<SimpleGradient>;
pub type SimpleGradientRenderer = CustomPatternRenderer<SimpleGradient>;

pub fn add_gradient(store: &mut GpuStore, p0: Point, color0: Color, p1: Point, color1: Color) -> SimpleGradient {
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
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Stop {
    pub position: Point,
    pub color: Color,
}
pub struct SimpleGradient {
    handle: GpuStoreHandle,
    is_opaque: bool,
}

impl SimpleGradient {
    pub fn new() -> Self {
        SimpleGradient {
            handle: GpuStoreHandle::INVALID,
            is_opaque: false,
        }
    }

    pub fn new_renderer(
        device: &wgpu::Device,
        helper: &mut CustomPatterns,
    ) -> CustomPatternRenderer<Self> {
        let varyings = &[
            Varying { name: "position", kind: "vec2<f32>", interpolate: "perspective" },
            Varying { name: "color0", kind: "vec4<f32>", interpolate: "flat" },
            Varying { name: "color1", kind: "vec4<f32>", interpolate: "flat" },
            Varying { name: "dir_offset", kind: "vec3<f32>", interpolate: "flat" },
        ];

        let src = include_str!("../shaders/simple_gradient_pattern.wgsl");

        let name = &"simple gradient";
        let pipeline = helper.create_tile_render_pipeline(
            device,
            &[],
            name,
            varyings,
            src,
        );

        CustomPatternRenderer::new(name, device, pipeline, ())
    }
}

impl CustomPattern for SimpleGradient {
    type RenderData = ();

    fn is_opaque(&self) -> bool { self.is_opaque }

    fn new_tile(&mut self, pattern_position: TilePosition) -> PatternData {
        [
           pattern_position.to_u32(),
            self.handle.to_u32(),
        ]
    }

    fn set_render_pass_state<'a, 'b: 'a>(_: &'a Self::RenderData, _: &mut wgpu::RenderPass<'b>) {}
}
