use core::{
    gpu::shader::{Shaders, ShaderGeometryId, GeometryDescriptor, VertexAtribute, OutputType, BlendMode, PipelineDescriptor, GeneratedPipelineId, ShaderMaskId},
    resources::{RendererResources}
};
use core::wgpu;

pub struct MeshGpuResources {
    pub simple_mesh_geometry: ShaderGeometryId,
    pub opaque_pipeline: GeneratedPipelineId,
    pub alpha_pipeline: GeneratedPipelineId,
}

impl MeshGpuResources {
    pub fn new(
        _device: &wgpu::Device,
        shaders: &mut Shaders,
    ) -> Self {
        let simple_mesh_geometry = shaders.register_geometry(GeometryDescriptor {
            name: "geometry::simple_mesh".into(),
            source: SIMPLE_MESH_SRC.into(),
            vertex_attributes: vec![
                VertexAtribute::float32x2("position"),
                VertexAtribute::uint32("z_index"),
                VertexAtribute::uint32("pattern"),
            ],
            instance_attributes: Vec::new(),
            bindings: None,
        });

        let opaque_pipeline = shaders.register_pipeline(PipelineDescriptor {
            label: "mesh(opaque)",
            geometry: simple_mesh_geometry,
            mask: ShaderMaskId::NONE,
            user_flags: 0,
            output: OutputType::Color,
            blend: BlendMode::None,
        });
        let alpha_pipeline = shaders.register_pipeline(PipelineDescriptor {
            label: "mesh(alpha)",
            geometry: simple_mesh_geometry,
            mask: ShaderMaskId::NONE,
            user_flags: 0,
            output: OutputType::Color,
            blend: BlendMode::PremultipliedAlpha,
        });

        MeshGpuResources {
            simple_mesh_geometry,
            opaque_pipeline,
            alpha_pipeline,
        }
    }
}

impl RendererResources for MeshGpuResources {
    fn name(&self) -> &'static str { "MeshGpuResources" }

    fn begin_frame(&mut self) {
    }

    fn begin_rendering(&mut self, _encoder: &mut wgpu::CommandEncoder) {
    }

    fn end_frame(&mut self) {
    }
}

const SIMPLE_MESH_SRC: &'static str = "
#import render_target

fn geometry(vertex_index: u32, canvas_position: vec2<f32>, z_index: u32, pattern_data: u32) -> Geometry {
    var target_position = canvas_to_target(canvas_position);

    var position = vec4<f32>(
        target_position.x,
        target_position.y,
        f32(z_index) / 8192.0,
        1.0,
    );

    return Geometry(
        position,
        canvas_position,
        pattern_data,
        // No suport for masks.
        vec2<f32>(0.0),
        0u,
    );
}
";
