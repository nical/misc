use core::gpu::PipelineDefaults;
use core::gpu::shader::Varying;
use core::wgpu;
use core::{
    gpu::shader::{
        GeneratedPipelineId, BaseShaderDescriptor, PipelineDescriptor,
        BaseShaderId, Shaders, VertexAtribute,
    },
    resources::RendererResources,
};

pub struct WpfGpuResources {
    pub base_shader: BaseShaderId,
    pub pipeline: GeneratedPipelineId,
}

impl WpfGpuResources {
    pub fn new(_device: &wgpu::Device, shaders: &mut Shaders) -> Self {
        let wpf_mesh_base = shaders.register_base_shader(BaseShaderDescriptor {
            name: "geometry::wpf_mesh".into(),
            source: WPF_MESH_SRC.into(),
            vertex_attributes: vec![
                VertexAtribute::float32x2("position"),
                VertexAtribute::float32("coverage"),
                VertexAtribute::uint32("pattern"),
            ],
            instance_attributes: Vec::new(),
            varyings: vec![
                Varying::float32("coverage"),
            ],
            bindings: None,
            primitive: PipelineDefaults::primitive_state(),
        });

        let pipeline = shaders.register_pipeline(PipelineDescriptor {
            label: "wpf mesh",
            base: wpf_mesh_base,
            shader_defines: Vec::new(),
        });

        WpfGpuResources {
            base_shader: wpf_mesh_base,
            pipeline,
        }
    }
}

impl RendererResources for WpfGpuResources {
    fn name(&self) -> &'static str {
        "MeshGpuResources"
    }

    fn begin_frame(&mut self) {}

    fn begin_rendering(&mut self, _encoder: &mut wgpu::CommandEncoder) {}

    fn end_frame(&mut self) {}
}

const WPF_MESH_SRC: &'static str = "
#import render_target

fn base_vertex(vertex_index: u32, canvas_position: vec2<f32>, coverage: f32, pattern_data: u32) -> BaseVertex {
    var target_position = canvas_to_target(canvas_position);

    var position = vec4<f32>(
        target_position.x,
        target_position.y,
        0.0, // todo: z_index_to_f32(z_index);
        1.0,
    );

    return BaseVertex(
        position,
        canvas_position,
        pattern_data,
        coverage,
    );
}

fn base_fragment(coverage: f32) -> f32 { return coverage; }
";
