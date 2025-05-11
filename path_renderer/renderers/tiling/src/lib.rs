#![allow(exported_private_dependencies)]

pub extern crate core;
pub extern crate lyon;

mod renderer;
pub mod tiler;
pub mod occlusion;
mod flatten;
mod simd4;

use core::batching::RendererId;
use core::wgpu;
use core::units::{LocalSpace, SurfaceSpace};
use core::gpu::{GpuStoreDescriptor, GpuStoreResources, PipelineDefaults};
use core::gpu::shader::{Shaders, BaseShaderDescriptor, BaseShaderId, BindGroupLayout, BindGroupLayoutId, Binding, Varying, VertexAtribute};

pub use renderer::*;

pub use lyon::lyon_tessellation::FillRule;
use tiler::EncodedPathInfo;

pub type Transform = lyon::geom::euclid::Transform2D<f32, LocalSpace, SurfaceSpace>;

#[derive(Copy, Clone, Debug)]
pub enum AaMode {
    /// High quality and fast, has conflation artifacts.
    AreaCoverage,
    /// Low quality, prevents conflation artifacts.
    Ssaa4,
    /// Medium quality, prevents conflation artifacts.
    Ssaa8,
}

#[derive(Clone, Debug)]
pub struct TilingOptions {
    pub antialiasing: AaMode,
}

impl Default for TilingOptions {
    fn default() -> Self {
        TilingOptions {
            antialiasing: AaMode::AreaCoverage
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Occlusion {
    /// Discard occluded content early on the CPU.
    pub cpu: bool,

    /// Use the depth buffer to discard occluded content.
    pub gpu: bool,
}

impl Default for Occlusion {
    fn default() -> Self {
        Occlusion {
            gpu: false,
            cpu: true,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RendererOptions {
    /// The flattening tolerance threshold.
    pub tolerance: f32,
    /// Whether to allow occlusion culling on CPU and/or GPU.
    pub occlusion: Occlusion,
    /// Don't produce draw calls with blending disabled.
    ///
    /// This puts the masked and interrior parts of the each path in the
    /// same draw call. It can avoid a large number of batches when both
    /// the CPU and GPU occlusion methods are disabled.
    pub no_opaque_batches: bool,
}

impl Default for RendererOptions {
    fn default() -> Self {
        RendererOptions {
            occlusion: Occlusion::default(),
            tolerance: 0.25,
            no_opaque_batches: false,
        }
    }
}

pub struct FillOptions<'l> {
    pub fill_rule: FillRule,
    pub inverted: bool,
    pub z_index: u32,
    pub tolerance: f32,
    pub transform: Option<&'l Transform>,
    pub opacity: f32,
}

impl<'l> FillOptions<'l> {
    pub fn new() -> FillOptions<'static> {
        FillOptions {
            fill_rule: FillRule::EvenOdd,
            inverted: false,
            z_index: 0,
            tolerance: 0.25,
            transform: None,
            opacity: 1.0,
        }
    }

    pub fn transformed<'a>(transform: &'a Transform) -> FillOptions<'a> {
        FillOptions {
            fill_rule: FillRule::EvenOdd,
            inverted: false,
            z_index: 0,
            tolerance: 0.25,
            transform: Some(transform),
            opacity: 1.0,
        }
    }

    pub fn with_transform<'a>(self, transform: Option<&'a Transform>) -> FillOptions<'a>
    where
        'l: 'a,
    {
        FillOptions {
            fill_rule: self.fill_rule,
            inverted: false,
            z_index: self.z_index,
            tolerance: self.tolerance,
            transform,
            opacity: self.opacity,
        }
    }

    pub fn with_fill_rule(mut self, fill_rule: FillRule) -> Self {
        self.fill_rule = fill_rule;
        self
    }

    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn with_inverted(mut self, inverted: bool) -> Self {
        self.inverted = inverted;
        self
    }

    pub fn with_z_index(mut self, z_index: u32) -> Self {
        self.z_index = z_index;
        self
    }

    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TilePosition(u32);

impl TilePosition {
    const MASK: u32 = 0x3FF;
    pub const ZERO: Self = TilePosition(0);
    pub const INVALID: Self = TilePosition(std::u32::MAX);

    pub fn extended(x: u32, y: u32, extend: u32) -> Self {
        debug_assert!(x <= Self::MASK);
        debug_assert!(y <= Self::MASK);
        debug_assert!(extend <= Self::MASK);

        TilePosition(extend << 20 | x << 10 | y)
    }

    pub fn new(x: u32, y: u32) -> Self {
        debug_assert!(x <= Self::MASK);
        debug_assert!(y <= Self::MASK);

        TilePosition(x << 10 | y)
    }

    pub fn extend(&mut self) {
        self.0 += 1 << 20;
    }

    pub fn with_flag(mut self) -> Self {
        self.add_flag();
        self
    }
    pub fn to_u32(&self) -> u32 {
        self.0
    }
    pub fn x(&self) -> u32 {
        (self.0 >> 10) & Self::MASK
    }
    pub fn y(&self) -> u32 {
        (self.0) & Self::MASK
    }
    pub fn extension(&self) -> u32 {
        (self.0 >> 20) & Self::MASK
    }

    // TODO: we have two unused bits and we use one of them to store
    // whether a tile in an indirection buffer is opaque. That's not
    // great.
    pub fn flag(&self) -> bool {
        self.0 & 1 << 31 != 0
    }
    pub fn add_flag(&mut self) {
        self.0 |= 1 << 31
    }
}

pub struct Tiling {
    base_shader: BaseShaderId,
    bind_group_layout: BindGroupLayoutId,
}

impl Tiling {
    pub fn new(device: &wgpu::Device, shaders: &mut Shaders, descriptor: &TilingOptions) -> Self {

        let bind_group_layout = shaders.register_bind_group_layout(BindGroupLayout::new(
            device,
            "tiling2::layout".into(),
            vec![
                Binding {
                    name: "path_texture".into(),
                    struct_type: "u32".into(),
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                },
                Binding {
                    name: "edge_texture".into(),
                    struct_type: "f32".into(),
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                }
            ],
        ));

        let mut shader_defines = Vec::new();
        match descriptor.antialiasing {
            AaMode::Ssaa4 => {
                shader_defines.push("TILING_SSAA4");
            }
            AaMode::Ssaa8 => {
                shader_defines.push("TILING_SSAA8");
            }
            AaMode::AreaCoverage => {}
        }

        let base_shader = shaders.register_base_shader(BaseShaderDescriptor {
            name: "geometry::tile2".into(),
            source: include_str!("../shaders/tile.wgsl").into(),
            vertex_attributes: Vec::new(),
            instance_attributes: vec![VertexAtribute::uint32x4("instance")],
            varyings: vec![
                Varying::float32x2("uv"),
                Varying::uint32x2("edges"),
                Varying::sint32("backdrop"),
                Varying::uint32("fill_rule"),
                Varying::float32("opacity"),
            ],
            bindings: Some(bind_group_layout),
            primitive: PipelineDefaults::primitive_state(),
            shader_defines,
        });

        Tiling {
            base_shader,
            bind_group_layout,
        }
    }

    pub fn new_renderer(
        &self,
        device: &wgpu::Device,
        shaders: &Shaders,
        renderer_id: RendererId,
        options: &RendererOptions,
    ) -> TileRenderer {
        let path_desc = GpuStoreDescriptor::Texture {
            format: wgpu::TextureFormat::Rgba32Uint,
            width: 2048,
            label: Some("tiling::paths"),
            alignment: std::mem::size_of::<EncodedPathInfo>() as u32,
        };
        let mut path_store = GpuStoreResources::new(&path_desc);
        path_store.allocate(4096 * 8, device);

        let edge_desc = GpuStoreDescriptor::rgba8_unorm_texture("tiling::edges");
        let mut edge_store = GpuStoreResources::new(&edge_desc);
        edge_store.allocate(4096 * 256, device);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tiling2::geom"),
            layout: &shaders.get_bind_group_layout(self.bind_group_layout).handle,
            entries: &[
                path_store.as_bind_group_entry(0).unwrap(),
                edge_store.as_bind_group_entry(1).unwrap(),
            ],
        });

        TileRenderer {
            renderer_id,
            tolerance: options.tolerance,
            occlusion: options.occlusion,
            no_opaque_batches: options.no_opaque_batches,
            back_to_front: options.occlusion.cpu || options.occlusion.gpu,
            tiler: crate::tiler::Tiler::new(),
            batches: core::batching::BatchList::new(renderer_id),
            base_shader: self.base_shader,
            mask_instances: None,
            opaque_instances: None,

            resources: TileGpuResources {
                bind_group,
                bind_group_layout: self.bind_group_layout,
                paths_epoch: path_store.epoch(),
                edges_epoch: edge_store.epoch(),
                paths: path_store,
                edges: edge_store,
            },

            path_transfer_ops: Vec::new(),
            edge_transfer_ops: Vec::new(),
            parallel: false,
        }
    }
}

#[test]
fn tile_position() {
    let mut p0 = TilePosition::new(1, 2);
    assert_eq!(p0.x(), 1);
    assert_eq!(p0.y(), 2);
    assert_eq!(p0.extension(), 0);

    p0.extend();

    assert_eq!(p0.x(), 1);
    assert_eq!(p0.y(), 2);
    assert_eq!(p0.extension(), 1);

    p0.extend();

    assert_eq!(p0.x(), 1);
    assert_eq!(p0.y(), 2);
    assert_eq!(p0.extension(), 2);
}

pub struct TileGpuResources {
    pub(crate) edges: GpuStoreResources,
    pub(crate) paths: GpuStoreResources,
    pub(crate) bind_group: wgpu::BindGroup,
    pub(crate) bind_group_layout: BindGroupLayoutId,
    pub(crate) edges_epoch: u32,
    pub(crate) paths_epoch: u32,
}
