use core::batching::RendererId;
use core::bitflags::bitflags;
use core::bytemuck;
use core::shading::PipelineDefaults;
use core::shading::GeometryId;
use core::wgpu;
use core::units::LocalRect;
use core::shading::{GeometryDescriptor, Shaders, Varying, VertexAtribute};

use crate::RectangleRenderer;

bitflags! {
    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct InstanceFlags: u32 {
        const AaTop    = 1 << 20;
        const AaRight  = 2 << 20;
        const AaBottom = 4 << 20;
        const AaLeft   = 8 << 20;
        const Aa       = (1|2|4|8) << 20;
        const AaCenter = 16 << 20;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Instance {
    pub local_rect: LocalRect,
    pub z_index: u32,
    pub pattern: u32,
    pub mask: u32,
    pub flags_transform: u32,
}

unsafe impl bytemuck::Zeroable for InstanceFlags {}
unsafe impl bytemuck::Pod for InstanceFlags {}
unsafe impl bytemuck::Zeroable for Instance {}
unsafe impl bytemuck::Pod for Instance {}

#[derive(Clone)]
pub(crate) struct Geometryes {
    pub opaque: GeometryId,
    pub alpha: GeometryId,
    pub opaque_no_aa: GeometryId,
    pub alpha_no_aa: GeometryId,
}

impl Geometryes {
    pub fn get(&self, opaque: bool, edge_aa: bool) -> GeometryId {
        match (opaque, edge_aa) {
            (true, true) => self.opaque,
            (false, true) => self.alpha,
            (true, false) => self.opaque_no_aa,
            (false, false) => self.alpha_no_aa,
        }
    }
}

pub struct Rectangles {
    pub(crate) geometryes: Geometryes,
}

impl Rectangles {
    pub fn new(_device: &wgpu::Device, shaders: &mut Shaders) -> Self {
        let geometry_descriptor = GeometryDescriptor {
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
            varyings: vec![Varying::float32x4("aa_distances")],
            bindings: None,
            primitive: PipelineDefaults::primitive_state(),
            shader_defines: Vec::new(),
            constants: Vec::new(),
        };

        let alpha = shaders.register_geometry(GeometryDescriptor {
            shader_defines: vec!["ALPHA_PASS", "EDGE_AA"],
            ..geometry_descriptor.clone()
        });

        let opaque = shaders.register_geometry(GeometryDescriptor {
            shader_defines: vec!["EDGE_AA"],
            ..geometry_descriptor.clone()
        });

        let alpha_no_aa = shaders.register_geometry(GeometryDescriptor {
            shader_defines: vec!["ALPHA_PASS"],
            ..geometry_descriptor.clone()
        });

        let opaque_no_aa = shaders.register_geometry(GeometryDescriptor {
            ..geometry_descriptor.clone()
        });

        Rectangles {
            geometryes: Geometryes {
                opaque,
                alpha,
                opaque_no_aa,
                alpha_no_aa,
            },
        }
    }

    pub fn new_renderer(&self, id: RendererId) -> RectangleRenderer {
        RectangleRenderer::new(id, self.geometryes.clone())
    }
}

const RECTANGLE_SRC: &'static str = "
#import render_target
#import rect
#import gpu_store
#import z_index

const FLAG_AA_TOP: u32      = 1048576u; // 1u << 20u;
const FLAG_AA_RIGHT: u32    = 2097152u; // 2u << 20u;
const FLAG_AA_BOTTOM: u32   = 4194304u; // 4u << 20u;
const FLAG_AA_LEFT: u32     = 8388608u; // 8u << 20u;
const FLAG_AA_CENTER: u32   = 16777216u; // 16u << 20u;

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

fn fetch_transform(address: u32) -> mat3x2<f32> {
    if address == GPU_STORE_HANDLE_NONE {
        return mat3x2<f32>(
            vec2<f32>(1.0, 0.0),
            vec2<f32>(0.0, 1.0),
            vec2<f32>(0.0, 0.0),
        );
    }

    let d = gpu_store_fetch_2(address);

    return mat3x2<f32>(
        d.data0.xy,
        d.data0.zw,
        d.data1.xy,
    );
}

fn geometry_vertex(vertex_index: u32, rect: vec4<f32>, z_index: u32, pattern: u32, mask: u32, transform_flags: u32) -> GeometryVertex {

    let transform_id = transform_flags & 0xFFFFFu;
    let flags = transform_flags & 0xFFF00000u;

    var local_rect = rect;
    var uv = rect_get_uv(vertex_index);

    let transform = fetch_transform(transform_id);

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
        z_index_to_f32(z_index),
        1.0,
    );

    return GeometryVertex(
        position,
        local_position,
        pattern,
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
