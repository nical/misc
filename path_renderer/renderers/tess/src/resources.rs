use core::gpu::PipelineDefaults;
use core::wgpu;
use core::{
    gpu::shader::{
        GeneratedPipelineId, BaseShaderDescriptor, PipelineDescriptor,
        BaseShaderId, Shaders, VertexAtribute,
    },
    resources::RendererResources,
};

pub struct MeshGpuResources {
    pub simple_mesh_base: BaseShaderId,
    pub pipeline: GeneratedPipelineId,
}

impl MeshGpuResources {
    pub fn new(_device: &wgpu::Device, shaders: &mut Shaders) -> Self {
        let simple_mesh_base = shaders.register_base_shader(BaseShaderDescriptor {
            name: "geometry::simple_mesh".into(),
            source: SIMPLE_MESH_SRC.into(),
            vertex_attributes: vec![
                VertexAtribute::float32x2("position"),
                VertexAtribute::uint32("z_index"),
                VertexAtribute::uint32("pattern"),
            ],
            instance_attributes: Vec::new(),
            varyings: Vec::new(),
            bindings: None,
            primitive: PipelineDefaults::primitive_state()
        });

        let pipeline = shaders.register_pipeline(PipelineDescriptor {
            label: "mesh(opaque)",
            base: simple_mesh_base,
            shader_defines: Vec::new(),
        });

        MeshGpuResources {
            simple_mesh_base,
            pipeline,
        }
    }
}

impl RendererResources for MeshGpuResources {
    fn name(&self) -> &'static str {
        "MeshGpuResources"
    }

    fn begin_frame(&mut self) {}

    fn begin_rendering(&mut self, _encoder: &mut wgpu::CommandEncoder) {}

    fn end_frame(&mut self) {}
}

const SIMPLE_MESH_SRC: &'static str = "
#import render_target
#import z_index

fn base_vertex(vertex_index: u32, canvas_position: vec2<f32>, z_index: u32, pattern_data: u32) -> BaseVertex {
    var target_position = canvas_to_target(canvas_position);

    var position = vec4<f32>(
        target_position.x,
        target_position.y,
        z_index_to_f32(z_index),
        1.0,
    );

    return BaseVertex(
        position,
        canvas_position,
        pattern_data,
    );
}

fn base_fragment() -> f32 { return 1.0; }

";
