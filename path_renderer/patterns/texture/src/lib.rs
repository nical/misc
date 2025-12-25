#![allow(exported_private_dependencies)]

use core::geom::Box2D;
use core::render_task::RenderTaskAdress;
use core::shading::{
    BindGroupLayout, BindGroupLayoutId, Binding, PatternDescriptor, ShaderPatternId, Shaders,
    Varying,
};
use core::gpu::GpuBufferWriter;
use core::pattern::BuiltPattern;
use core::BindingsId;
use core::wgpu;

#[derive(Clone, Debug)]
pub struct TextureRenderer {
    load_shader: ShaderPatternId,
    sample_shader: ShaderPatternId,
    bind_group_layout: BindGroupLayoutId,
}

impl TextureRenderer {
    pub fn register(device: &wgpu::Device, shaders: &mut Shaders) -> Self {
        let bind_group_layout = shaders.register_bind_group_layout(BindGroupLayout::new(
            device,
            "color_texture".into(),
            vec![Binding {
                name: "src_color_texture".into(),
                struct_type: "f32".into(),
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
            }],
        ));

        let load_shader = shaders.register_pattern(PatternDescriptor {
            name: "pattern::texture_load".into(),
            source: LOAD_SHADER_SRC.into(),
            varyings: vec![Varying::float32x2("uv").interpolated()],
            bindings: Some(bind_group_layout),
        });

        let sample_shader = shaders.register_pattern(PatternDescriptor {
            name: "pattern::texture_sample".into(),
            source: SAMPLE_SHADER_SRC.into(),
            varyings: vec![
                Varying::float32x3("uv_alpha").interpolated(),
                Varying::float32x4("uv_bounds").interpolated(),
                Varying::float32x4("dbg").interpolated(),
            ],
            bindings: Some(bind_group_layout),
        });

        TextureRenderer {
            load_shader,
            sample_shader,
            bind_group_layout,
        }
    }

    pub fn load_pattern_id(&self) -> ShaderPatternId {
        self.load_shader
    }

    pub fn sample_pattern_id(&self) -> ShaderPatternId {
        self.sample_shader
    }

    pub fn bind_group_layout(&self) -> BindGroupLayoutId {
        self.bind_group_layout
    }

    #[inline]
    pub fn load_direct(&self, is_opaque: bool) -> BuiltPattern {
        BuiltPattern::new(self.load_shader, 0)
            .with_opacity(is_opaque)
    }

    #[inline]
    pub fn sample_rect(
        &self,
        f32_buffer: &mut GpuBufferWriter,
        src_texture: BindingsId,
        src_task: RenderTaskAdress,
        dst_rect: &Box2D<f32>,
        alpha: f32,
        is_opaque: bool,
    ) -> BuiltPattern {
        let handle = f32_buffer.push_slice(&[
            src_task.to_u32() as f32,
            alpha,
            0.0,
            0.0,
            dst_rect.min.x,
            dst_rect.min.y,
            dst_rect.max.x,
            dst_rect.max.y,
        ]);

        BuiltPattern::new(self.sample_shader, handle.to_u32())
            .with_bindings(src_texture)
            .with_opacity(is_opaque)
    }
}

const LOAD_SHADER_SRC: &'static str = "
#import pattern::color

fn pattern_vertex(pattern_pos: vec2<f32>, pattern_handle: u32) -> Pattern {
    return Pattern(pattern_pos);
}

fn pattern_fragment(pattern: Pattern) -> vec4<f32> {
    let uv = vec2<i32>(pattern.uv);
    let color = textureLoad(src_color_texture, uv, 0);
    return unpremultiply_color(color);
}
";

const SAMPLE_SHADER_SRC: &'static str = "
#import gpu_buffer
#import render_task
#import pattern::color

fn pattern_vertex(pattern_pos: vec2<f32>, pattern_handle: u32) -> Pattern {
    let data = f32_gpu_buffer_fetch_2(pattern_handle);

    let opacity = data.data0.y;

    // Source and destination rects in pixels.
    let src_task_address = u32(data.data0.x);
    let src_rect = render_task_fetch_image_source(src_task_address);

    let dst_rect = data.data1;

    let inv_texture_size = vec2<f32>(1.0) / vec2<f32>(textureDimensions(src_color_texture, 0i).xy);

    // dst_uv and uv_bounds are in normalized texture space.

    let src_size = src_rect.zw - src_rect.xy;
    let dst_size = dst_rect.zw - dst_rect.xy;
    let uv = (pattern_pos - dst_rect.xy) / dst_size;
    let dst_uv = (src_rect.xy + uv * src_size) * inv_texture_size;

    // Shrink the sample bounds by half a pixel to prevent the interpolation from
    // sampling outside of the desired rect when the geometry is inflated.
    var uv_bounds = vec4<f32>(
        (src_rect.xy + vec2<f32>(0.5)) * inv_texture_size,
        (src_rect.zw - vec2<f32>(0.5)) * inv_texture_size,
    );

    return Pattern(vec3(dst_uv, opacity), uv_bounds, src_rect);
}

fn pattern_fragment(pattern: Pattern) -> vec4<f32> {
    var uv = pattern.uv_alpha.xy;
    let alpha = pattern.uv_alpha.z;
    // restrict samples.
    uv = max(uv, pattern.uv_bounds.xy);
    uv = min(uv, pattern.uv_bounds.zw);

    let color = textureSampleLevel(src_color_texture, default_sampler, uv, 0.0) * alpha;

    return unpremultiply_color(color);
}
";
