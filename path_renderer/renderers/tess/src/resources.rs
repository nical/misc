use core::batching::RendererId;
use core::wgpu;
use core::shading::{
    GeometryDescriptor, PipelineDefaults,
    GeometryId, Shaders, VertexAtribute,
};

use crate::MeshRenderer;

pub struct Tessellation {
    pub geometry: GeometryId,
}

impl Tessellation {
    pub fn new(_device: &wgpu::Device, shaders: &mut Shaders) -> Self {
        let geometry = shaders.register_geometry(GeometryDescriptor {
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
            primitive: PipelineDefaults::primitive_state(),
            shader_defines: Vec::new(),
            constants: Vec::new(),
        });

        Tessellation {
            geometry,
        }
    }

    pub fn new_renderer(&self, renderer_id: RendererId) -> MeshRenderer {
        MeshRenderer::new(renderer_id, self.geometry)
    }
}

const SIMPLE_MESH_SRC: &'static str = "
#import render_target
#import z_index

fn geometry_vertex(vertex_index: u32, canvas_position: vec2<f32>, z_index: u32, pattern_data: u32) -> GeometryVertex {
    var target_position = canvas_to_target(canvas_position);

    var position = vec4<f32>(
        target_position.x,
        target_position.y,
        z_index_to_f32(z_index),
        1.0,
    );

    return GeometryVertex(
        position,
        canvas_position,
        pattern_data,
    );
}

fn geometry_fragment() -> f32 { return 1.0; }

";
