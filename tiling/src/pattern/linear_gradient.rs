use lyon::geom::traits::Transformation;
use lyon::math::Point;

use crate::gpu::{GpuStore};
use crate::gpu::shader::{ShaderPatternId, Shaders, Varying, PatternDescriptor};
use crate::{Color};

use super::BuiltPattern;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LinearGradient {
    pub from: Point,
    pub to: Point,
    pub color0: Color,
    pub color1: Color,
}

impl LinearGradient {
    pub fn transformed<T: Transformation<f32>>(&self, tx: &T) -> Self {
        LinearGradient {
            from: tx.transform_point(self.from),
            to: tx.transform_point(self.to),
            color0: self.color0,
            color1: self.color1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LinearGradientRenderer {
    shader: ShaderPatternId,
}

impl LinearGradientRenderer {
    pub fn register(shaders: &mut Shaders) -> Self {
        let shader = shaders.register_pattern(PatternDescriptor {
            name: "pattern::linear_gradient".into(),
            source: include_str!("../../shaders/pattern/simple_gradient.wgsl").into(),
            varyings: vec![
                Varying::float32x2("position").with_interpolation(true),
                Varying::float32x4("color0").with_interpolation(false),
                Varying::float32x4("color1").with_interpolation(false),
                Varying::float32x3("dir_offset").with_interpolation(false),
            ],
            bindings: None,
        });

        LinearGradientRenderer { shader }
    }

    pub fn add(&self, gpu_store: &mut GpuStore, gradient: LinearGradient) -> BuiltPattern {
        let can_stretch_horizontally = gradient.from.x == gradient.to.y || gradient.color0 == gradient.color1;
        let is_opaque = gradient.color0.is_opaque() && gradient.color1.is_opaque();
        let color0 = gradient.color0.to_f32();
        let color1 = gradient.color1.to_f32();

        let handle = gpu_store.push(&[
            gradient.from.x, gradient.from.y, gradient.to.x, gradient.to.y,
            color0[0], color0[1], color0[2], color0[3],
            color1[0], color1[1], color1[2], color1[3],
        ]);

        BuiltPattern {
            data: handle.to_u32(),
            is_opaque,
            can_stretch_horizontally,
            shader: self.shader,
            favor_prerendering: false,
        }
    }
}
