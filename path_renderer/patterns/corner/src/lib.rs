#![allow(exported_private_dependencies)]

use core::shading::{ PatternDescriptor, ShaderPatternId, Shaders, Varying };
use core::gpu::GpuBufferWriter;
use core::pattern::BuiltPattern;
use core::units::LocalRect;
use core::wgpu;

#[derive(Clone, Debug)]
pub struct RoundedCornerRenderer {
    pub rounded_corner_shader: ShaderPatternId,
}

impl RoundedCornerRenderer {
    pub fn register(_device: &wgpu::Device, shaders: &mut Shaders) -> Self {
        let rounded_corner_shader = shaders.register_pattern(PatternDescriptor {
            name: "pattern::uniform_rounded_corner".into(),
            source: ROUNDED_CORNER_SRC.into(),
            varyings: vec![
                Varying::float32x2("local_pos").interpolated(),
                Varying::float32x4("rectangle").flat(),
                Varying::float32x4("radii").flat(),
            ],
            bindings: None,
        });

        RoundedCornerRenderer {
            rounded_corner_shader,
        }
    }

    #[inline]
    pub fn uniform(
        &self,
        f32_buffer: &mut GpuBufferWriter,
        dst_rect: &LocalRect,
        radius: f32,
    ) -> BuiltPattern {
        let handle = f32_buffer.push_slice(&[
            dst_rect.min.x,
            dst_rect.min.y,
            dst_rect.max.x,
            dst_rect.max.y,
            radius,
            radius,
            radius,
            radius,
        ]);

        BuiltPattern::new(self.rounded_corner_shader, handle.to_u32())
            .with_opacity(false)
    }

    #[inline]
    pub fn non_uniform(
        &self,
        f32_buffer: &mut GpuBufferWriter,
        dst_rect: &LocalRect,
        top_left_radius: f32,
        top_right_radius: f32,
        bottom_right_radius: f32,
        bottom_left_radius: f32,
    ) -> BuiltPattern {
        // The shader only works if the radii fit in their respective quadrant.
        let max = dst_rect.width().min(dst_rect.height()) * 0.5;

        let handle = f32_buffer.push_slice(&[
            dst_rect.min.x,
            dst_rect.min.y,
            dst_rect.max.x,
            dst_rect.max.y,
            top_left_radius.min(max),
            top_right_radius.min(max),
            bottom_right_radius.min(max),
            bottom_left_radius.min(max),
        ]);

        BuiltPattern::new(self.rounded_corner_shader, handle.to_u32())
            .with_opacity(false)
    }
}

const ROUNDED_CORNER_SRC: &'static str = "
fn pattern_vertex(pattern_pos: vec2<f32>, pattern_handle: u32) -> Pattern {
    let data = f32_gpu_buffer_fetch_2(pattern_handle);
    let rectangle = data.data0;
    let radii = data.data1;
    return Pattern(
        pattern_pos,
        rectangle,
        radii,
    );
}

fn sdf_rounded_rect(pos: vec2f, rect: vec4f, radii: vec4f) -> f32 {
    let center = (rect.xy + rect.zw) * 0.5;
    let half_box_size = (rect.zw - rect.xy) * 0.5;

    // Position relative to the center of the rect.
    let p = pos - center;
    // TODO: this assumes that the radii are never more than half of the smallest side
    // of the rectangle.
    let r2 = mix(radii.xw, radii.yz, f32(p.x > 0.0));
    let radius = mix(r2.x, r2.y, f32(p.y > 0.0));

    let q = abs(p) - half_box_size + vec2f(radius);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2f(0.0))) - radius;
}

fn sdf_uniform_rounded_rect(pos: vec2f, rect: vec4f, radius: f32) -> f32 {
    let center = (rect.xy + rect.zw) * 0.5;
    let half_box_size = (rect.zw - rect.xy) * 0.5;

    let p = pos - center;
    let q = abs(p) - half_box_size + vec2f(radius);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2f(0.0))) - radius;
}

fn distance_aa(pos: vec2f, sdf: f32) -> f32 {
    let w = fwidth(pos);
    let aa_range = inverseSqrt(0.5 * dot(w, w));

    return clamp(0.5 - sdf * aa_range, 0.0, 1.0);
}

fn pattern_fragment(pattern: Pattern) -> vec4<f32> {
    let sdf = sdf_rounded_rect(pattern.local_pos, pattern.rectangle, pattern.radii);
    let alpha = distance_aa(pattern.local_pos, sdf);

    return vec4f(1.0, 1.0, 1.0, alpha);
}
";
