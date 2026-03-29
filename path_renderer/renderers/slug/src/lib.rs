#![allow(exported_private_dependencies, dead_code)]

pub extern crate core;

mod renderer;

pub use renderer::SlugRenderer;

use core::batching::RendererId;
use core::shading::{
    BindGroupLayout, Binding, BindGroupLayoutId, GeometryDescriptor, GeometryId,
    PipelineDefaults, Shaders, Varying, VertexAtribute,
};
use core::wgpu;

pub struct Slug {
    pub geometry: GeometryId,
    pub bind_group_layout: BindGroupLayoutId,
}

impl Slug {
    pub fn new(device: &wgpu::Device, shaders: &mut Shaders) -> Self {
        // Register a bind group layout for the data textures (curve + band).
        // This becomes @group(1) in the generated shader.
        let bgl = shaders.register_bind_group_layout(BindGroupLayout::new(
            device,
            "slug_data".into(),
            vec![
                Binding {
                    name: "curve_texture".into(),
                    struct_type: "".into(),
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    visibility: wgpu::ShaderStages::FRAGMENT,
                },
                Binding {
                    name: "band_texture".into(),
                    struct_type: "".into(),
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    visibility: wgpu::ShaderStages::FRAGMENT,
                },
            ],
        ));

        // Register the coverage computation as a shader library so that the
        // geometry shader can `#import slug_coverage`.
        shaders.register_library(
            "slug_coverage",
            include_str!("../shaders/slug_coverage.wgsl").into(),
        );

        let geometry = shaders.register_geometry(GeometryDescriptor {
            name: "geometry::slug".into(),
            source: include_str!("../shaders/slug.wgsl").into(),
            vertex_attributes: Vec::new(),
            instance_attributes: vec![
                VertexAtribute::float32x4("local_rect"),
                VertexAtribute::float32x4("band_params"),
                VertexAtribute::uint32("band_loc"),
                VertexAtribute::uint32("band_max"),
                VertexAtribute::uint32("z_index"),
                VertexAtribute::uint32("pattern"),
                VertexAtribute::uint32("render_task"),
                VertexAtribute::uint32("flags_transform"),
            ],
            varyings: vec![
                Varying::float32x2("texcoord"),
                Varying::float32x4("band_params").flat(),
                Varying::sint32x2("glyph_loc").flat(),
                Varying::sint32x2("band_max_val").flat(),
            ],
            bindings: Some(bgl),
            primitive: PipelineDefaults::primitive_state(),
            shader_defines: Vec::new(),
            constants: Vec::new(),
        });

        Slug { geometry, bind_group_layout: bgl }
    }

    pub fn new_renderer(&self, renderer_id: RendererId) -> SlugRenderer {
        SlugRenderer::new(renderer_id, self.geometry, self.bind_group_layout)
    }
}
