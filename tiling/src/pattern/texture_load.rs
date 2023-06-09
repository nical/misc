use crate::gpu::shader::{ShaderPatternId, BindGroupLayoutId, Shaders, Varying, PatternDescriptor, BindGroupLayout, Binding};

#[derive(Clone, Debug)]
pub struct TextureLoadRenderer {
    shader: ShaderPatternId,
    bind_group_layout: BindGroupLayoutId,
}

impl TextureLoadRenderer {
    pub fn register(device: &wgpu::Device, shaders: &mut Shaders) -> Self {
        let bind_group_layout = shaders.register_bind_group_layout(BindGroupLayout::new(
            device,
            "color_texture".into(),
            vec![Binding {
                name: "src_color_texture".into(),
                struct_type: "f32".into(),
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }
            }],
        ));

        let shader = shaders.register_pattern(PatternDescriptor {
            name: "pattern::texture_load".into(),
            source: SHADER_SRC.into(),
            varyings: vec![
                Varying::float32x2("uv").interpolated(),
            ],
            bindings: Some(bind_group_layout),
        });

        TextureLoadRenderer { shader, bind_group_layout }
    }

    pub fn pattern_id(&self) -> ShaderPatternId {
        self.shader
    }

    pub fn bind_group_layout(&self) -> BindGroupLayoutId {
        self.bind_group_layout
    }
}

const SHADER_SRC: &'static str = "
fn pattern_vertex(pattern_pos: vec2<f32>, pattern_handle: u32) -> Pattern {
    return Pattern(pattern_pos);
}

fn pattern_fragment(pattern: Pattern) -> vec4<f32> {
    var uv = vec2<i32>(i32(pattern.uv.x), i32(pattern.uv.y));
    return textureLoad(src_color_texture, uv, 0);
}
";
