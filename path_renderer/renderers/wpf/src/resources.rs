use core::wgpu;
use core::shading::{
    GeometryDescriptor, PipelineDefaults,
    GeometryId, Shaders, VertexAtribute, Varying,
};
use core::context::RendererId;

use crate::WpfMeshRenderer;

pub struct Wpf {
    pub geometry: GeometryId,
}

impl Wpf {
    pub fn new(_device: &wgpu::Device, shaders: &mut Shaders) -> Self {
        let wpf_mesh_base = shaders.register_geometry(GeometryDescriptor {
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
            shader_defines: Vec::new(),
            constants: Vec::new(),
        });

        Wpf {
            geometry: wpf_mesh_base,
        }
    }

    pub fn new_renderer(&self, id: RendererId) -> WpfMeshRenderer {
        WpfMeshRenderer::new(id, self.geometry)
    }
}

const WPF_MESH_SRC: &'static str = "
#import render_target

fn geometry_vertex(vertex_index: u32, canvas_position: vec2<f32>, coverage: f32, pattern_data: u32) -> GeometryVertex {
    var target_position = canvas_to_target(canvas_position);

    var position = vec4<f32>(
        target_position.x,
        target_position.y,
        0.0, // todo: z_index_to_f32(z_index);
        1.0,
    );

    return GeometryVertex(
        position,
        canvas_position,
        pattern_data,
        coverage,
    );
}

fn geometry_fragment(coverage: f32) -> f32 { return coverage; }
";
