use core::gpu::PipelineDefaults;
use core::wgpu;
use core::{
    gpu::shader::{
        BlendMode, GeneratedPipelineId, GeometryDescriptor, PipelineDescriptor,
        ShaderGeometryId, ShaderMaskId, Shaders, VertexAtribute,
    },
    resources::RendererResources,
};

pub struct MeshGpuResources {
    pub simple_mesh_geometry: ShaderGeometryId,
    pub opaque_pipeline: GeneratedPipelineId,
    pub alpha_pipeline: GeneratedPipelineId,
}

impl MeshGpuResources {
    pub fn new(_device: &wgpu::Device, shaders: &mut Shaders) -> Self {
        let simple_mesh_geometry = shaders.register_geometry(GeometryDescriptor {
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

        let opaque_pipeline = shaders.register_pipeline(PipelineDescriptor {
            label: "mesh(opaque)",
            geometry: simple_mesh_geometry,
            mask: ShaderMaskId::NONE,
            user_flags: 0,
            blend: BlendMode::None,
            shader_defines: Vec::new(),
        });
        let alpha_pipeline = shaders.register_pipeline(PipelineDescriptor {
            label: "mesh(alpha)",
            geometry: simple_mesh_geometry,
            mask: ShaderMaskId::NONE,
            user_flags: 0,
            blend: BlendMode::PremultipliedAlpha,
            shader_defines: Vec::new(),
        });

        MeshGpuResources {
            simple_mesh_geometry,
            opaque_pipeline,
            alpha_pipeline,
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

fn geometry_vertex(vertex_index: u32, canvas_position: vec2<f32>, z_index: u32, pattern_data: u32) -> Geometry {
    var target_position = canvas_to_target(canvas_position);

    var position = vec4<f32>(
        target_position.x,
        target_position.y,
        z_index_to_f32(z_index),
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

fn geometry_fragment() -> f32 { return 1.0; }

";
