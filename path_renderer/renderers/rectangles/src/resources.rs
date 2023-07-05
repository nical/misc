use core::{
    gpu::shader::{Shaders, ShaderGeometryId, GeometryDescriptor, VertexAtribute, OutputType, BlendMode, PipelineDescriptor, GeneratedPipelineId, ShaderMaskId, Varying},
    resources::{RendererResources}, units::LocalRect
};
use core::wgpu;
use core::bitflags::bitflags;
use core::bytemuck;

bitflags!{
    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct InstanceFlags: u16 {
        const AaTop    = 1;
        const AaRight  = 2;
        const AaBottom = 4;
        const AaLeft   = 8;
        const Aa       = 1|2|4|8;
        const AaCenter = 16;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Instance {
    pub local_rect: LocalRect,
    pub z_index: u32,
    pub pattern: u32,
    pub mask: u32,
    pub transform: u16,
    pub flags: InstanceFlags,
}

unsafe impl bytemuck::Zeroable for InstanceFlags {}
unsafe impl bytemuck::Pod for InstanceFlags {}
unsafe impl bytemuck::Zeroable for Instance {}
unsafe impl bytemuck::Pod for Instance {}

#[derive(Clone)]
pub(crate) struct Pipelines {
    pub opaque_pipeline: GeneratedPipelineId,
    pub alpha_pipeline: GeneratedPipelineId,
    pub opaque_pipeline_no_aa: GeneratedPipelineId,
    pub alpha_pipeline_no_aa: GeneratedPipelineId,
}

impl Pipelines {
    pub fn get(&self, opaque: bool, edge_aa: bool) -> GeneratedPipelineId {
        match (opaque, edge_aa) {
            (true, true) => self.opaque_pipeline,
            (false, true) => self.alpha_pipeline,
            (true, false) => self.opaque_pipeline_no_aa,
            (false, false) => self.alpha_pipeline_no_aa,
        }
    }
}

pub struct RectangleGpuResources {
    pub rect_geometry: ShaderGeometryId,
    pub(crate) pipelines: Pipelines
}

impl RectangleGpuResources {
    pub fn new(
        _device: &wgpu::Device,
        shaders: &mut Shaders,
    ) -> Self {
        let rect_geometry = shaders.register_geometry(GeometryDescriptor {
            name: "geometry::rectangle".into(),
            source: RECTANGLE_SRC.into(),
            vertex_attributes: Vec::new(),
            instance_attributes: vec![
                VertexAtribute::float32x4("local_rect"),
                VertexAtribute::uint32("z_index"),
                VertexAtribute::uint32("pattern"),
                VertexAtribute::uint32("mask"),
                VertexAtribute::uint32("flags"),
            ],
            varyings: vec![
                Varying::float32x4("aa_distances"),
            ],
            bindings: None,
        });

        let opaque_pipeline = shaders.register_pipeline(PipelineDescriptor {
            label: "rectangle(opaque)",
            geometry: rect_geometry,
            mask: ShaderMaskId::NONE,
            user_flags: 0,
            output: OutputType::Color,
            blend: BlendMode::None,
            shader_defines: vec!["EDGE_AA"],
        });
        let alpha_pipeline = shaders.register_pipeline(PipelineDescriptor {
            label: "rectangle(alpha)",
            geometry: rect_geometry,
            mask: ShaderMaskId::NONE,
            user_flags: 0,
            output: OutputType::Color,
            blend: BlendMode::PremultipliedAlpha,
            shader_defines: vec!["ALPHA_PASS", "EDGE_AA"],
        });

        let opaque_pipeline_no_aa = shaders.register_pipeline(PipelineDescriptor {
            label: "rectangle(opaque, no-aa)",
            geometry: rect_geometry,
            mask: ShaderMaskId::NONE,
            user_flags: 0,
            output: OutputType::Color,
            blend: BlendMode::None,
            shader_defines: Vec::new(),
        });
        let alpha_pipeline_no_aa = shaders.register_pipeline(PipelineDescriptor {
            label: "rectangle(alpha, no-aa)",
            geometry: rect_geometry,
            mask: ShaderMaskId::NONE,
            user_flags: 0,
            output: OutputType::Color,
            blend: BlendMode::PremultipliedAlpha,
            shader_defines: vec!["ALPHA_PASS"],
        });

        RectangleGpuResources {
            rect_geometry,
            pipelines: Pipelines {
                opaque_pipeline,
                alpha_pipeline,
                opaque_pipeline_no_aa,
                alpha_pipeline_no_aa,
            }
        }
    }
}

impl RendererResources for RectangleGpuResources {
    fn name(&self) -> &'static str { "RectangleGpuResources" }

    fn begin_frame(&mut self) {
    }

    fn begin_rendering(&mut self, _encoder: &mut wgpu::CommandEncoder) {
    }

    fn end_frame(&mut self) {
    }
}

const RECTANGLE_SRC: &'static str = "
#import render_target
#import rect

const FLAG_AA_TOP: u32      = 1u;
const FLAG_AA_RIGHT: u32    = 2u;
const FLAG_AA_BOTTOM: u32   = 4u;
const FLAG_AA_LEFT: u32     = 8u;
const FLAG_AA_CENTER: u32   = 16u;

fn edge_aa_offset(mask: u32, flags: u32, offset: f32) -> f32 {
    if ((flags & mask) == 0u) {
        return 0.0;
    }

    var sign = 1.0;
    if ((flags & FLAG_AA_CENTER) != 0u) {
        sign = -1.0;
    }

    return sign * offset;
}

fn geometry_vertex(vertex_index: u32, rect: vec4<f32>, z_index: u32, pattern: u32, mask: u32, transform_flags: u32) -> Geometry {

    let transform_id = transform_flags & 0xFFFFu;
    let flags = (transform_flags >> 16u) & 0xFFFFu;

    var local_rect = rect;
    var uv = rect_get_uv(vertex_index);

    // TODO: fetch from the gpu store.
    let transform = mat3x2<f32>(
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 0.0),
    );

    var aa_distances = vec4<f32>(42.0);

    #if EDGE_AA {
        let local_size = local_rect.zw - local_rect.xy;

        // Basis vectors of the transformation
        let basis_x = transform[0];
        let basis_y = transform[1];
        // Squared lengths of the two basis vectors.
        let scale2 = vec2<f32>(dot(basis_x, basis_x), dot(basis_y, basis_y));
        let inv_scale = inverseSqrt(scale2);
        let scale = scale2 * inv_scale;

        // Offsets in local space to apply the extrusion on the local rect.
        let offsets = vec4<f32>(
            -edge_aa_offset(FLAG_AA_LEFT, flags, inv_scale.x),
            -edge_aa_offset(FLAG_AA_TOP, flags, inv_scale.y),
            edge_aa_offset(FLAG_AA_RIGHT, flags, inv_scale.x),
            edge_aa_offset(FLAG_AA_BOTTOM, flags, inv_scale.y),
        ) * 0.5;

        // Extrusion.
        local_rect += offsets;

        #if ALPHA_PASS {
            // Compute the signed distance between the current vertex and each
            // edge of the rectangle. Start in local space and convert to device
            // space by applying the scale.
            // TODO: this would not work with clipping. Instead of using the offsets
            // in the computation, it should use the extruded and clipped rect.
            let uvuv = vec4<f32>(uv, vec2<f32>(1.0) - uv);
            let local_aa_distances = uvuv * vec4<f32>(local_size, local_size)
                + (uvuv * 2.0 - vec4<f32>(1.0)) * abs(offsets);
    
            aa_distances = local_aa_distances * scale.xyxy;
        }
    }

    let local_position = mix(local_rect.xy, local_rect.zw, uv);

    let canvas_position = (transform * vec3(local_position, 1.0)).xy;
    var target_position = canvas_to_target(canvas_position);

    var position = vec4<f32>(
        target_position.x,
        target_position.y,
        f32(z_index) / 8192.0,
        1.0,
    );

    return Geometry(
        position,
        local_position,
        pattern,
        local_position,
        mask,
        aa_distances,
    );
}

fn geometry_fragment(aa_distances: vec4<f32>) -> f32 {
    var aa = 1.0;
    #if EDGE_AA {
        #if ALPHA_PASS {
            let dist = min(aa_distances.xy, aa_distances.zw);
            aa = clamp(min(dist.x, dist.y) + 0.5, 0.0, 1.0);
        }
    }

    return aa;
}

";
