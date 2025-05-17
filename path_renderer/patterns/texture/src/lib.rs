#![allow(exported_private_dependencies)]

use core::geom::Box2D;
use core::shading::{
    BindGroupLayout, BindGroupLayoutId, Binding, PatternDescriptor, ShaderPatternId, Shaders,
    Varying, BlendMode,
};
use core::gpu::GpuStoreWriter;
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
                Varying::float32x2("uv").interpolated(),
                Varying::float32x4("uv_bounds").interpolated(),
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
            .with_blend_mode(if is_opaque { BlendMode::None } else { BlendMode::PremultipliedAlpha })
    }

    #[inline]
    pub fn sample_rect(
        &self,
        gpu_store: &mut GpuStoreWriter,
        src_texture: BindingsId,
        src_rect: &Box2D<f32>,
        dst_rect: &Box2D<f32>,
        is_opaque: bool,
    ) -> BuiltPattern {
        let handle = gpu_store.push_slice(&[
            src_rect.min.x,
            src_rect.min.y,
            src_rect.max.x,
            src_rect.max.y,
            dst_rect.min.x,
            dst_rect.min.y,
            dst_rect.max.x,
            dst_rect.max.y,
        ]);

        BuiltPattern::new(self.sample_shader, handle.to_u32())
            .with_bindings(src_texture)
            .with_opacity(is_opaque)
            .with_blend_mode(if is_opaque { BlendMode::None } else { BlendMode::PremultipliedAlpha })
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
#import gpu_store
#import pattern::color

fn pattern_vertex(pattern_pos: vec2<f32>, pattern_handle: u32) -> Pattern {
    let data = gpu_store_fetch_2(pattern_handle);
    // Source and destination rects in pixels.
    let src_rect = data.data0;
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

    return Pattern(dst_uv, uv_bounds);
}

fn pattern_fragment(pattern: Pattern) -> vec4<f32> {
    var uv = pattern.uv;
    // restrict samples.
    uv = max(uv, pattern.uv_bounds.xy);
    uv = min(uv, pattern.uv_bounds.zw);

    let color = textureSampleLevel(src_color_texture, default_sampler, uv, 0.0);

    return unpremultiply_color(color);
}
";
