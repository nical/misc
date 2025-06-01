use core::wgpu;
use core::shading::{
    GeometryDescriptor, PipelineDefaults,
    GeometryId, Shaders, VertexAtribute, Varying,
};
use core::render_pass::RendererId;

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
                VertexAtribute::uint32("path"),
            ],
            instance_attributes: Vec::new(),
            varyings: vec![
                Varying::float32("coverage"),
                Varying::float32("opacity").flat(),
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
#import render_task
#import gpu_buffer
#import z_index

struct PathInfo {
    z: f32,
    pattern: u32,
    opacity: f32,
    render_task: u32,
};

fn decode_path(address: u32) -> PathInfo {
    let encoded = u32_gpu_buffer_fetch_1(address);
    return PathInfo(
        z_index_to_f32(encoded.x),
        encoded.y,
        f32(encoded.z & 0xFFFFu) / 65535.0,
        encoded.w,
    );
}

fn geometry_vertex(vertex_index: u32, canvas_position: vec2<f32>, coverage: f32, path_address: u32) -> GeometryVertex {
    var path = decode_path(path_address);
    var task = render_task_fetch(path.render_task);

    var target_position = render_task_target_position(task, canvas_position);

    var position = vec4<f32>(
        target_position.x,
        target_position.y,
        path.z,
        1.0,
    );

    return GeometryVertex(
        position,
        canvas_position,
        path.pattern,
        coverage,
        path.opacity,
    );
}

fn geometry_fragment(coverage: f32, opacity: f32) -> f32 { return coverage * opacity; }
";
