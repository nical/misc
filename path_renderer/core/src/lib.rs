pub extern crate wgpu;

pub mod render_pass;
pub mod gpu;
pub mod batching;
pub mod cache;
pub mod path;
pub mod pattern;
pub mod render_task;
pub mod resources;
pub mod shape;
pub mod stroke;
pub mod transform;
pub mod graph;
pub mod instance;
pub mod worker;
pub mod shading;
pub mod utils;
pub mod color;

use shading::{Shaders, PrepareRenderPipelines, RenderPipelines};
pub use crate::shading::{SurfaceDrawConfig, SurfaceKind, StencilMode, DepthMode};

use render_pass::{BuiltRenderPass, RenderPassContext};
pub use render_pass::{RenderPassConfig};

use gpu::{GpuBuffer, GpuStreams, StagingBufferPool, UploadStats};
pub use lyon::path::math::{point, vector, Point, Vector};

use std::sync::{Arc, Mutex};
pub use bitflags;
pub use bytemuck;
pub use lyon::geom;
use pattern::BuiltPattern;
use resources::GpuResources;
use transform::Transforms;

pub mod units {
    use lyon::geom::euclid::{self, Box2D, Point2D, Size2D, Vector2D};

    #[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
    pub struct LocalSpace;

    #[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
    pub struct SurfaceSpace;

    pub type Rect = euclid::default::Box2D<f32>;
    pub type Point = euclid::default::Point2D<f32>;
    pub type Vector = euclid::default::Vector2D<f32>;
    pub type Size = euclid::default::Size2D<f32>;

    pub type SurfaceRect = Box2D<f32, SurfaceSpace>;
    pub type SurfacePoint = Point2D<f32, SurfaceSpace>;
    pub type SurfaceVector = Vector2D<f32, SurfaceSpace>;
    pub type SurfaceSize = Size2D<f32, SurfaceSpace>;
    pub type SurfaceIntPoint = Point2D<i32, SurfaceSpace>;
    pub type SurfaceIntSize = Size2D<i32, SurfaceSpace>;
    pub type SurfaceIntRect = Box2D<i32, SurfaceSpace>;

    pub type LocalRect = Box2D<f32, LocalSpace>;
    pub type LocalPoint = Point2D<f32, LocalSpace>;
    pub type LocalVector = Vector2D<f32, LocalSpace>;
    pub type LocalSize = Size2D<f32, LocalSpace>;

    pub type LocalToSurfaceTransform = euclid::Transform2D<f32, LocalSpace, SurfaceSpace>;
    pub type LocalTransform = euclid::Transform2D<f32, LocalSpace, LocalSpace>;

    pub use euclid::point2 as point;
    pub use euclid::vec2 as vector;
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ColorF {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const RED: Self = Color {
        r: 255,
        g: 0,
        b: 0,
        a: 255,
    };
    pub const GREEN: Self = Color {
        r: 0,
        g: 255,
        b: 0,
        a: 255,
    };
    pub const BLUE: Self = Color {
        r: 0,
        g: 0,
        b: 255,
        a: 255,
    };
    pub const BLACK: Self = Color {
        r: 0,
        g: 0,
        b: 0,
        a: 255,
    };
    pub const WHITE: Self = Color {
        r: 255,
        g: 255,
        b: 255,
        a: 255,
    };

    pub fn is_opaque(self) -> bool {
        self.a == 255
    }

    pub fn to_u32(self) -> u32 {
        (self.r as u32) << 24 | (self.g as u32) << 16 | (self.b as u32) << 8 | self.a as u32
    }

    pub fn to_f32(self) -> [f32; 4] {
        [
            self.r as f32 / 255.0,
            self.g as f32 / 255.0,
            self.b as f32 / 255.0,
            self.a as f32 / 255.0,
        ]
    }

    pub fn to_colorf(self) -> ColorF {
        ColorF {
            r: self.r as f32 / 255.0,
            g: self.g as f32 / 255.0,
            b: self.b as f32 / 255.0,
            a: self.a as f32 / 255.0,
        }
    }

    pub fn to_wgpu(self) -> wgpu::Color {
        wgpu::Color {
            r: self.r as f64 / 255.0,
            g: self.g as f64 / 255.0,
            b: self.b as f64 / 255.0,
            a: self.a as f64 / 255.0,
        }
    }

    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Color { r, g, b, a }
    }

    pub fn linear_to_srgb(r: u8, g: u8, b: u8, a: u8) -> Self {
        fn f(linear: f32) -> f32 {
            if linear <= 0.0031308 {
                linear * 12.92
            } else {
                1.055 * linear.powf(1.0 / 2.4) - 0.055
            }
        }
        let r = (f(r as f32 / 255.0) * 255.0) as u8;
        let g = (f(g as f32 / 255.0) * 255.0) as u8;
        let b = (f(b as f32 / 255.0) * 255.0) as u8;

        Color { r, g, b, a }
    }

    pub fn srgb_to_linear(r: u8, g: u8, b: u8, a: u8) -> Self {
        fn f(srgb: f32) -> f32 {
            if srgb <= 0.04045 {
                srgb * 12.92
            } else {
                ((srgb + 0.055) / 1.055).powf(2.4)
            }
        }
        let r = (f(r as f32 / 255.0) * 255.0) as u8;
        let g = (f(g as f32 / 255.0) * 255.0) as u8;
        let b = (f(b as f32 / 255.0) * 255.0) as u8;

        Color { r, g, b, a }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BindingsId {
    namespace: BindingsNamespace,
    index: u32,
}

impl BindingsId {
    pub const NONE: Self = BindingsId {
        namespace: BindingsNamespace::None,
        index: 0,
    };

    #[inline]
    pub fn is_none(&self) -> bool {
        self.namespace == BindingsNamespace::None
    }

    #[inline]
    pub fn is_some(&self) -> bool {
        self.namespace != BindingsNamespace::None
    }

    // TODO: the index used to be stored as a u16 but the render stack needed
    // more bits. Decide whether to make it u32 for everyone or pack more
    // aggressively.
    #[inline]
    pub(crate) const fn new(namespace: BindingsNamespace, index: u16) -> Self {
        BindingsId { namespace, index: index as u32 }
    }

    #[inline]
    pub const fn renderer(idx: u16) -> Self {
        BindingsId::new(BindingsNamespace::Renderer, idx)
    }

    #[inline]
    pub const fn external(idx: u16) -> Self {
        BindingsId::new(BindingsNamespace::External, idx)
    }

    #[inline]
    pub(crate) const fn graph(idx: u16) -> Self {
        BindingsId::new(BindingsNamespace::RenderGraph, idx)
    }

    #[inline]
    pub fn namespace(&self) -> BindingsNamespace {
        self.namespace
    }

    #[inline]
    pub fn index(&self) -> usize {
        self.index as usize
    }

    #[inline]
    pub fn to_u32(self) -> u32 {
        (self.namespace as u32) << 16 | self.index as u32
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BindingsNamespace {
    /// The bindings are generated by the render graph.
    RenderGraph,
    /// The bindings are generated by a render stack.
    RenderStack,
    /// The bindings are created externally and registered to the render graph.
    External,
    /// The bindings are opaque to the core systems. Renderers can use them internally.
    Renderer,
    /// The bindings are unused.
    None,
}

pub trait BindingResolver {
    fn resolve_input(&self, id: BindingsId) -> Option<&wgpu::BindGroup>;
    fn resolve_attachment(&self, id: BindingsId) -> Option<&wgpu::TextureView>;
}

impl BindingResolver for () {
    fn resolve_input(&self, _: BindingsId) -> Option<&wgpu::BindGroup> { None }
    fn resolve_attachment(&self, _: BindingsId) -> Option<&wgpu::TextureView> { None }
}

/// Parameters for the renderering stage.
pub struct RenderContext<'a, 'b> {
    pub render_pipelines: &'a RenderPipelines,
    pub resources: &'a resources::GpuResources,
    pub bindings: &'a dyn BindingResolver,
    pub stats: &'b mut RendererStats,
}

pub struct PrepareWorkerData {
    pub pipelines: PrepareRenderPipelines,
    pub f32_buffer: GpuBuffer,
    pub u32_buffer: GpuBuffer,
    pub vertices: GpuBuffer,
    pub indices: GpuStreams,
    pub instances: GpuStreams,
    // allocator
}

pub type PrepareWorkerContext<'a> = worker::Context<'a, (PrepareWorkerData,)>;

/// Parameters for the renderering stage.
pub struct PrepareContext<'a> {
    pub pass: &'a BuiltRenderPass,
    pub transforms: &'a Transforms,
    pub workers: PrepareWorkerContext<'a>,
    pub staging_buffers: Arc<Mutex<StagingBufferPool>>,
}

/// Parameters for the renderering stage.
pub struct UploadContext<'l> {
    pub resources: &'l mut GpuResources,
    pub shaders: &'l Shaders,
    pub wgpu: WgpuContext<'l>,
}

pub struct WgpuContext<'l> {
    pub device: &'l wgpu::Device,
    pub queue: &'l wgpu::Queue,
    pub encoder: &'l mut wgpu::CommandEncoder,
}

#[derive(Clone, Debug, Default)]
pub struct RendererStats {
    pub draw_calls: u32,
    // In bytes.
    pub gpu_data: u32,
    // In ms.
    pub prepare_time: f32,
    // In ms.
    pub upload_time: f32,
    // In ms.
    pub render_time: f32,
}

impl RendererStats {
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

pub trait Renderer {
    fn prepare(&mut self, ctx: &mut PrepareContext);

    fn upload(&mut self, _ctx: &mut UploadContext) -> UploadStats { UploadStats::default() }

    fn render<'pass, 'resources: 'pass, 'tmp>(
        &self,
        _batches: &[batching::BatchId],
        _pass_info: &render_pass::RenderPassConfig,
        _ctx: RenderContext<'resources, 'tmp>,
        _render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
    }
}

pub trait FillPath {
    fn fill_path(&mut self, ctx: &mut RenderPassContext, transforms: &Transforms, path: shape::FilledPath, pattern: BuiltPattern);
}
