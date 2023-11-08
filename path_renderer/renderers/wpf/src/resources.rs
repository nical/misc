use core::gpu::shader::Varying;
use core::wgpu;
use core::{
    gpu::shader::{
        BlendMode, GeneratedPipelineId, GeometryDescriptor, PipelineDescriptor,
        ShaderGeometryId, ShaderMaskId, Shaders, VertexAtribute,
    },
    resources::RendererResources,
};

pub struct WpfGpuResources {
    pub geometry: ShaderGeometryId,
    pub pipeline: GeneratedPipelineId,
}

impl WpfGpuResources {
    pub fn new(_device: &wgpu::Device, shaders: &mut Shaders) -> Self {
        let wpf_mesh_geometry = shaders.register_geometry(GeometryDescriptor {
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
        });

        let pipeline = shaders.register_pipeline(PipelineDescriptor {
            label: "wpf mesh",
            geometry: wpf_mesh_geometry,
            mask: ShaderMaskId::NONE,
            user_flags: 0,
            blend: BlendMode::PremultipliedAlpha,
            shader_defines: Vec::new(),
        });

        WpfGpuResources {
            geometry: wpf_mesh_geometry,
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

fn geometry_vertex(vertex_index: u32, canvas_position: vec2<f32>, coverage: f32, pattern_data: u32) -> Geometry {
    var target_position = canvas_to_target(canvas_position);

    var position = vec4<f32>(
        target_position.x,
        target_position.y,
        0.0,
        1.0,
    );

    return Geometry(
        position,
        canvas_position,
        pattern_data,
        // No suport for masks.
        vec2<f32>(0.0),
        0u,
        coverage,
    );
}

fn geometry_fragment(coverage: f32) -> f32 { return coverage; }

";
