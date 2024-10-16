pub mod buffer;
pub mod context;
pub mod gpu;
//pub mod flatten_simd;
pub mod batching;
pub mod cache;
pub mod path;
pub mod pattern;
pub mod resources;
pub mod shape;
pub mod stroke;
pub mod transform;
pub mod render_graph;
//pub mod canvas;

use context::RenderPassContext;
pub use lyon::path::math::{point, vector, Point, Vector};

pub use bitflags;
pub use bytemuck;
pub use lyon::geom;
use pattern::BuiltPattern;
use resources::AsAny;
use transform::Transforms;
pub use wgpu;
pub use etagere;

use std::fmt;

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

pub use crate::context::{
    SurfaceDrawConfig, SurfacePassConfig, SurfaceKind, StencilMode, DepthMode,
};

#[derive(Copy, Clone, Debug, PartialEq)]
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
    index: u16,
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

    #[inline]
    pub(crate) const fn new(namespace: BindingsNamespace, index: u16) -> Self {
        BindingsId { namespace, index }
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
    /// The bindings are created externally and registered to the render graph.
    External,
    /// The bindings are opaque to the core systems. Renderers can use them internally.
    Renderer,
    /// The bindings are unused.
    None,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ResourceKind(u16);

impl ResourceKind {
    pub fn is_texture(self) -> bool {
        self.0 & BufferKind::BUFFER == 0
    }

    pub fn is_buffer(self) -> bool {
        self.0 & BufferKind::BUFFER != 0
    }

    pub fn as_texture(&self) -> Option<TextureKind> {
        if self.is_texture() {
            Some(TextureKind(self.0))
        } else {
            None
        }
    }

    pub fn as_buffer(&self) -> Option<BufferKind> {
        if self.is_buffer() {
            Some(BufferKind(self.0))
        } else {
            None
        }
    }
}

impl fmt::Debug for ResourceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(tex) = self.as_texture() {
            return tex.fmt(f);
        }
        if let Some(buf) = self.as_buffer() {
            return buf.fmt(f);
        }

        write!(f, "<InvalidBufferKind>")
    }
}

impl fmt::Debug for TextureKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Texture(")?;
        if self.is_color() {
            write!(f, "color")?;
        }
        if self.is_alpha() {
            write!(f, "alpha")?;
        }
        if self.is_depth_stencil() {
            write!(f, "depth-stencil")?;
        }
        if self.is_hdr() {
            write!(f, "|hdr")?;
        }
        if self.is_msaa() {
            write!(f, "|msaa")?;
        }
        if self.is_attachment() {
            write!(f, "|attachment")?;
        }
        if self.is_binding() {
            write!(f, "|binding")?;
        }
        if self.is_copy_src() {
            write!(f, "|copy-src")?;
        }
        if self.is_copy_dst() {
            write!(f, "|copy-dst")?;
        }

        write!(f, ")")
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct TextureKind(u16);
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferKind(u16);


impl TextureKind {
    const ALPHA: u16 = 1;
    const DEPTH_STENCIL: u16 = 2;

    const MSAA: u16 = 1 << 3;
    const HDR: u16 = 1 << 4;
    const COPY_SRC: u16 = 1 << 5;
    const COPY_DST: u16 = 1 << 6;
    const BINDING: u16 = 1 << 7;
    const ATTACHMENT: u16 = 1 << 8;

    pub const fn color() -> Self {
        TextureKind(0)
    }

    pub const fn color_attachment() -> Self {
        Self::color().with_attachment()
    }

    pub const fn alpha_attachment() -> Self {
        Self::alpha().with_attachment()
    }

    pub const fn color_binding() -> Self {
        Self::color().with_binding()
    }

    pub const fn alpha_binding() -> Self {
        Self::alpha().with_binding()
    }

    pub const fn alpha() -> Self {
        TextureKind(Self::ALPHA)
    }

    pub const fn depth_stencil() -> Self {
        TextureKind(Self::DEPTH_STENCIL)
    }

    pub const fn with_hdr(self) -> Self {
        TextureKind(self.0 | Self::HDR)
    }

    pub const fn with_attachment(self) -> Self {
        TextureKind(self.0 | Self::ATTACHMENT)
    }

    pub const fn with_binding(self) -> Self {
        TextureKind(self.0 | Self::BINDING)
    }

    pub const fn with_msaa(self, msaa: bool) -> Self {
        if msaa {
            TextureKind(self.0 | Self::MSAA)
        } else {
            TextureKind(self.0 & !Self::MSAA)
        }
    }

    pub const fn with_copy_src(self) -> Self {
        TextureKind(self.0 | Self::COPY_SRC)
    }

    pub const fn with_copy_dst(self) -> Self {
        TextureKind(self.0 | Self::COPY_DST)
    }

    pub const fn as_resource(self) -> ResourceKind {
        ResourceKind(self.0)
    }

    pub const fn is_color(self) -> bool {
        self.0 & (Self::ALPHA | Self::DEPTH_STENCIL)  == 0
    }

    pub const fn is_alpha(self) -> bool {
        self.0 & Self::ALPHA != 0
    }

    pub const fn is_depth_stencil(self) -> bool {
        self.0 & Self::DEPTH_STENCIL != 0
    }

    pub const fn is_hdr(self) -> bool {
        self.0 & Self::HDR != 0
    }

    pub const fn is_attachment(self) -> bool {
        self.0 & Self::ATTACHMENT != 0
    }

    pub const fn is_binding(self) -> bool {
        self.0 & Self::BINDING != 0
    }

    pub const fn is_msaa(self) -> bool {
        self.0 & Self::MSAA != 0
    }

    pub const fn is_copy_src(self) -> bool {
        self.0 & Self::COPY_SRC != 0
    }

    pub const fn is_copy_dst(self) -> bool {
        self.0 & Self::COPY_DST != 0
    }

    pub const fn is_compatible_width(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl BufferKind {
    const BUFFER: u16 = 1 << 15;

    const UNIFORM: u16 = 1 << 0;

    const COPY_SRC: u16 = 1 << 1;
    const COPY_DST: u16 = 1 << 2;

    pub fn as_resource(self) -> ResourceKind {
        ResourceKind(self.0)
    }

    pub fn storage(self) -> Self {
        BufferKind(Self::BUFFER)
    }

    pub fn uniform(self) -> Self {
        BufferKind(Self::BUFFER | Self::UNIFORM)
    }

    pub fn staging(self) -> Self {
        BufferKind(Self::BUFFER | Self::COPY_SRC | Self::COPY_DST)
    }

    pub const fn with_copy_src(self) -> Self {
        BufferKind(self.0 | Self::COPY_SRC)
    }

    pub const fn with_copy_dst(self) -> Self {
        BufferKind(self.0 | Self::COPY_DST)
    }
}

use std::ops::Range;
#[inline]
pub fn u32_range(r: Range<usize>) -> Range<u32> {
    r.start as u32..r.end as u32
}

#[inline]
pub fn usize_range(r: Range<u32>) -> Range<usize> {
    r.start as usize..r.end as usize
}

pub trait BindingResolver {
    fn resolve_input(&self, id: BindingsId) -> Option<&wgpu::BindGroup>;
    fn resolve_attachment(&self, id: BindingsId) -> Option<&wgpu::TextureView>;
}

impl BindingResolver for () {
    fn resolve_input(&self, _: BindingsId) -> Option<&wgpu::BindGroup> { None }
    fn resolve_attachment(&self, _: BindingsId) -> Option<&wgpu::TextureView> { None }
}

/// Parameters for the canvas renderers
pub struct RenderContext<'l> {
    pub render_pipelines: &'l gpu::shader::RenderPipelines,
    pub resources: &'l resources::GpuResources,
    pub bindings: &'l dyn BindingResolver,
}

pub trait Renderer: AsAny {
    fn render_pre_pass(
        &self,
        _index: u32,
        _ctx: RenderContext,
        _encoder: &mut wgpu::CommandEncoder,
    ) {
    }

    fn render<'pass, 'resources: 'pass>(
        &self,
        _batches: &[batching::BatchId],
        _pass_info: &context::SurfacePassConfig,
        _ctx: RenderContext<'resources>,
        _render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
    }
}

pub trait FillPath {
    fn fill_path(&mut self, ctx: &mut RenderPassContext, transforms: &Transforms, path: shape::FilledPath, pattern: BuiltPattern);
}
