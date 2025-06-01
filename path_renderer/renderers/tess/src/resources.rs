use core::batching::RendererId;
use core::wgpu;
use core::shading::{
    GeometryDescriptor, GeometryId, PipelineDefaults, Shaders, Varying, VertexAtribute
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
                VertexAtribute::uint32("address"),
                VertexAtribute::float32("_padding"),
            ],
            instance_attributes: Vec::new(),
            varyings: vec![
                Varying::float32("opacity").flat(),
            ],
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

    pub fn geometry(&self) -> GeometryId {
        self.geometry
    }
}

const SIMPLE_MESH_SRC: &'static str = "
#import render_task
#import gpu_buffer
#import z_index

struct MeshInfo {
    z: f32,
    pattern: u32,
    opacity: f32,
    render_task: u32,
};

fn decode_mesh_info(address: u32) -> MeshInfo {
    let encoded = u32_gpu_buffer_fetch_1(address);
    return MeshInfo(
        z_index_to_f32(encoded.x),
        encoded.y,
        f32(encoded.z & 0xFFFFu) / 65535.0,
        encoded.w,
    );
}

fn geometry_vertex(vertex_index: u32, canvas_position: vec2<f32>, mesh_address: u32, _padding: f32) -> GeometryVertex {
    var mesh = decode_mesh_info(mesh_address);
    var task = render_task_fetch(mesh.render_task);

    var target_position = render_task_target_position(task, canvas_position);

    var position = vec4<f32>(
        target_position.x,
        target_position.y,
        mesh.z,
        1.0,
    );

    return GeometryVertex(
        position,
        canvas_position,
        mesh.pattern,
        mesh.opacity,
    );
}

fn geometry_fragment(opacity: f32) -> f32 { return opacity; }

";
