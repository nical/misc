use core::gpu::PipelineDefaults;
use core::gpu::shader::{BindGroupLayout, Binding, BindGroupLayoutId};
use core::gpu::storage_buffer::{StorageBuffer, StorageKind};
use core::wgpu;
use core::{
    gpu::shader::{
        GeneratedPipelineId, BaseShaderDescriptor, PipelineDescriptor,
        BaseShaderId, Shaders, VertexAtribute,
    },
    resources::RendererResources,
};

use crate::PathData;

pub struct MsaaStrokeGpuResources {
    pub base_shader: BaseShaderId,
    pub opaque_pipeline: GeneratedPipelineId,
    pub alpha_pipeline: GeneratedPipelineId,
    pub paths: StorageBuffer,
    pub geom_bind_group_layout: BindGroupLayoutId,
    pub geom_bind_group: Option<wgpu::BindGroup>,
}

impl MsaaStrokeGpuResources {
    pub fn new(device: &wgpu::Device, shaders: &mut Shaders) -> Self {

        let bgl = shaders.register_bind_group_layout(BindGroupLayout::new(
            device,
            "msaa stroker geom".into(),
            vec![
                Binding {
                    name: "path_data".into(),
                    struct_type: "array<PathData>".into(),
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<PathData>() as u64),
                    },
                }
            ]
        ));

        let curve_geometry = shaders.register_base_shader(BaseShaderDescriptor {
            name: "geometry::skeletal_stroke_curves".into(),
            source: SHADER_SRC.into(),
            vertex_attributes: Vec::new(),
            instance_attributes: vec![
                VertexAtribute::float32x2("seg_from"),
                VertexAtribute::float32x2("seg_ctrl1"),
                VertexAtribute::float32x2("seg_ctrl2"),
                VertexAtribute::float32x2("seg_to"),
                VertexAtribute::float32x2("prev_ctrl"),
                VertexAtribute::uint32("path_index"),
                VertexAtribute::uint32("segment_counts"),
            ],
            varyings: Vec::new(),
            bindings: Some(bgl),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                .. PipelineDefaults::primitive_state()
            },
        });

        let opaque_pipeline = shaders.register_pipeline(PipelineDescriptor {
            label: "skel_stroke(opaque)",
            base: curve_geometry,
            shader_defines: Vec::new(),
        });
        let alpha_pipeline = shaders.register_pipeline(PipelineDescriptor {
            label: "skel_stroke(alpha)",
            base: curve_geometry,
            shader_defines: Vec::new(),
        });

        MsaaStrokeGpuResources {
            base_shader: curve_geometry,
            opaque_pipeline,
            alpha_pipeline,
            paths: StorageBuffer::new::<PathData>(device, "stroke path data", 4096 * 16, StorageKind::Buffer),
            geom_bind_group_layout: bgl,
            geom_bind_group: None,
        }
    }
}

impl RendererResources for MsaaStrokeGpuResources {
    fn name(&self) -> &'static str {
        "SkeletalStrokeGpuResources"
    }

    fn begin_frame(&mut self) {
        self.paths.begin_frame();    
    }

    fn begin_rendering(&mut self, _encoder: &mut wgpu::CommandEncoder) {}

    fn end_frame(&mut self) {}
}

const SHADER_SRC: &'static str = "
#import render_target
#import z_index

struct PathData {
    transform: vec4f,
    tx: f32,
    ty: f32,
    width: f32,
    pad0: f32,
    pattern: u32,
    z_index: u32,
    pad1: u32,
    pad2: u32,
};

fn base_vertex(
    vertex_index: u32,
    seg_from: vec2<f32>,
    seg_ctrl1: vec2<f32>,
    seg_ctrl2: vec2<f32>,
    seg_to: vec2<f32>,
    prev_ctrl: vec2<f32>,
    path_index: u32,
    segment_counts: u32
) -> BaseVertex {
    var join_segments = segment_counts & 0xffffu;
    var curve_segments = segment_counts >> 16u;
    var side = f32(i32(vertex_index % 2u) * 2i - 1i);
    let segment_index = vertex_index / 2u;
    var is_curve = vertex_index >= join_segments;

    var path = path_data[path_index];
    var z_index = path.z_index;

    // +--+--+--+--+--+--+--+--+--+--+
    // | /| /| /| /| /| /| /| /| /| /|
    // |/ |/ |/ |/ |/ |/ |/ |/ |/ |/ |
    // +--+--+--+--+--+--+--+--+--+--+
    //       ________________________
    // ______  curve segments
    // join segments

    var curve_segment_index = min(max(segment_index, join_segments) - join_segments, curve_segments);
    var t: f32 = f32(curve_segment_index) / f32(curve_segments);

    var t2 = t * t;
    var t3 = t2 * t;
    var one_t = 1.0 - t;
    var one_t2 = one_t * one_t;
    var one_t3 = one_t2 * one_t;

    var p = seg_from * one_t3
        + seg_ctrl1 * (3.0 * one_t2 * t)
        + seg_ctrl2 * (3.0 * one_t * t2)
        + seg_to * t3;

    var deriv: vec2f;
    if (is_curve) {
        deriv = seg_from * (6.0 * t - 3.0 * t2 - 3.0)
            + seg_ctrl1 * (9.0 * t2 - 12.0 * t + 3.0)
            + seg_ctrl2 * (6.0 * t - 9.0 * t2)
            + seg_to * (3.0 * t2);
    } else {
        deriv = seg_from - prev_ctrl;
    }

    var n = normalize(vec2f(-deriv.y, deriv.x));

    var local_position = p + n * 0.5 * path.width * side;
    var transform = mat2x2<f32>(path.transform.xy, path.transform.zw);
    var translation = vec2f(path.tx, path.ty);

    var canvas_position = local_position * transform + translation;

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
        path.pattern,
        // No suport for masks.
    );
}

fn base_fragment() -> f32 { return 1.0; }

";
