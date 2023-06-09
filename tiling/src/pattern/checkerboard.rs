use lyon::geom::euclid::vec2;
use lyon::geom::traits::Transformation;
use lyon::math::Point;

use crate::gpu::shader::{PatternDescriptor, ShaderPatternId, Varying};
use crate::gpu::{GpuStore, Shaders};
use crate::{Color};

use super::BuiltPattern;


#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Checkerboard {
    pub color0: Color,
    pub color1: Color,
    pub offset: Point,
    pub scale: f32,
}

#[derive(Clone, Debug)]
pub struct CheckerboardRenderer {
    shader: ShaderPatternId,
}

impl Checkerboard {
    pub fn transformed<T: Transformation<f32>>(&self, tx: &T) -> Self {
        Checkerboard {
            offset: tx.transform_point(self.offset),
            scale: tx.transform_vector(vec2(0.0, self.scale)).y,
            color0: self.color0,
            color1: self.color1,
        }
    }
}


impl CheckerboardRenderer {
    pub fn register(shaders: &mut Shaders) -> Self {
        let shader = shaders.register_pattern(PatternDescriptor {
            name: "pattern::checkerboard".into(),
            source: include_str!("../../shaders/pattern/checkerboard.wgsl").into(),
            varyings: vec![
                Varying::float32x2("uv").with_interpolation(true),
                Varying::float32x4("color0").with_interpolation(false),
                Varying::float32x4("color1").with_interpolation(false),
            ],
            bindings: None,
        });

        CheckerboardRenderer { shader }
    }

    pub fn add(&self, gpu_store: &mut GpuStore, pattern: &Checkerboard) -> BuiltPattern {
        let is_opaque = pattern.color0.is_opaque() && pattern.color1.is_opaque();
        let color0 = pattern.color0.to_f32();
        let color1 = pattern.color1.to_f32();
        let handle = gpu_store.push(&[
            color0[0], color0[1], color0[2], color0[3],
            color1[0], color1[1], color1[2], color1[3],
            pattern.offset.x, pattern.offset.y,
            pattern.scale,
        ]);
    
        BuiltPattern {
            data: handle.to_u32(),
            shader: self.shader,
            is_opaque,
            can_stretch_horizontally: false,
            favor_prerendering: true,
        }
    }
}
