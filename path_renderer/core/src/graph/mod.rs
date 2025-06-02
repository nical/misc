mod schedule;

use std::{fmt, u16};
use bitflags::bitflags;

use crate::{instance::RenderStats, resources::GpuResources, shading::RenderPipelines, units::SurfaceIntSize, BindingResolver, Renderer, SurfaceKind};

pub use schedule::{RenderGraph, GraphBindings, GraphError};

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct NodeId(pub u16);

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

impl NodeId {
    pub fn index(self) -> usize {
        self.0 as usize
    }

    pub fn from_index(idx: usize) -> Self {
        debug_assert!(idx < std::u16::MAX as usize);
        NodeId(idx as u16)
    }

    pub fn color(self, slot: u8) -> Dependency {
        Dependency {
            node: self,
            slot: Slot::Color(slot),
        }
    }

    pub fn msaa(self, slot: u8) -> Dependency {
        Dependency {
            node: self,
            slot: Slot::Msaa(slot),
        }
    }

    pub fn depth_stencil(self) -> Dependency {
        Dependency {
            node: self,
            slot: Slot::DepthStencil,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum Slot {
    Color(u8),
    Msaa(u8),
    DepthStencil,
    // Other(u8)
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Dependency {
    pub node: NodeId,
    pub slot: Slot,
}

impl From<NodeId> for Dependency {
    fn from(node: NodeId) -> Self {
        Dependency {
            node,
            slot: Slot::Color(0),
        }
    }
}

impl fmt::Debug for Dependency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.slot {
            Slot::Color(idx) => write!(f, "#{}.color_{}", self.node.0, idx),
            Slot::Msaa(idx) => write!(f, "#{}.msaa_{}", self.node.0, idx),
            Slot::DepthStencil => write!(f, "#{}.depth_stencil", self.node.0),
        }
    }
}

/// TODO: find another name for this.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TaskId(pub u64);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Allocation {
    /// Allocated by the render graph.
    Temporary,
    /// Allocated outside of the render graph.
    External,
    // TODO: a Retained variant for resources that are explicitly created/destroyed
    // outisde of the render graph but still managed by the gpu resource pool.
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

    pub fn from_surface_kind(kind: SurfaceKind) -> Self {
        match kind {
            SurfaceKind::Color => Self::color(),
            SurfaceKind::Alpha => Self::alpha(),
            SurfaceKind::HdrColor => Self::color().with_hdr(),
            SurfaceKind::HdrAlpha => Self::alpha().with_hdr(),
            SurfaceKind::None => unimplemented!(),
        }
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

    pub fn as_key(&self, w: u16, h: u16) -> ResourceKey {
        let w = texture_size_class(w);
        let h = texture_size_class(h);
        ResourceKey(self.0 | (w << 12) | (h << 9))
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

fn texture_size_class(size: u16) -> u16 {
    // Must fit in 3 bits (up to 7 buckets).
    match size {
        0..1025 => 1,
        1025..2049 => 2,
        2049..4097 => 3,
        4097..8193 => 4,
        8193..16385 => 5,
        _ => panic!("Invalid texture size")
    }
}
fn texture_size_from_class(class: u16) -> u16 {
    match class {
        1 => 1024,
        2 => 2048,
        3 => 4096,
        4 => 8192,
        _ => 16384,
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

    pub fn storage() -> Self {
        BufferKind(Self::BUFFER)
    }

    pub fn uniform() -> Self {
        BufferKind(Self::BUFFER | Self::UNIFORM)
    }

    pub fn staging() -> Self {
        BufferKind(Self::BUFFER | Self::COPY_SRC | Self::COPY_DST)
    }

    pub const fn with_copy_src(self) -> Self {
        BufferKind(self.0 | Self::COPY_SRC)
    }

    pub const fn with_copy_dst(self) -> Self {
        BufferKind(self.0 | Self::COPY_DST)
    }
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

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ResourceKey(u16);

impl ResourceKey {
    pub fn is_texture(self) -> bool {
        self.0 & BufferKind::BUFFER == 0
    }

    pub fn is_buffer(self) -> bool {
        self.0 & BufferKind::BUFFER != 0
    }

    pub fn as_texture(&self) -> Option<(TextureKind, u16, u16)> {
        if !self.is_texture() {
            return None;
        }

        let kind = self.0 & 0b111111111;
        let w = self.0 >> 12 & 0b111;
        let h = self.0 >> 9 & 0b111;
        Some((
            TextureKind(kind),
            texture_size_from_class(w),
            texture_size_from_class(h),
        ))
    }
}

impl fmt::Debug for ResourceKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some((kind, w, h)) = self.as_texture() {
            return write!(f, "#{kind:?}({w},{h})");
        }
        let buf = BufferKind(self.0); // TODO: buffer size classes
        write!(f, "#{buf:?}(..)")
    }
}

// The virtual/physical resource distinction is similar to virtual and physical
// registers: Every output port of a node is its own virtual resource, to which
// is assigned a physical resource when scheduling the graph. Physical resources
// are used by as many nodes as possible to minimize memory usage.

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TempResourceKey {
    pub kind: ResourceKind,
    pub size: (u32, u32),
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ColorAttachment {
    pub kind: TextureKind,
    pub non_msaa: Resource,
    pub msaa: Resource,
}

impl ColorAttachment {
    pub const fn color() -> Self {
        ColorAttachment {
            kind: TextureKind::color(),
            non_msaa: Resource::Auto,
            msaa: Resource::Auto,
        }
    }

    pub const fn alpha() -> Self {
        ColorAttachment {
            kind: TextureKind::alpha(),
            non_msaa: Resource::Auto,
            msaa: Resource::Auto,
        }
    }

    pub const fn with_dependency(mut self, dep: Dependency) -> Self {
        match dep.slot {
            Slot::Color(_) => {
                self.non_msaa = Resource::Dependency(dep);
            }
            Slot::Msaa(_) => {
                self.msaa = Resource::Dependency(dep);
            }
            Slot::DepthStencil => {
                panic!("Can't use a depth_stencil slot as a color attachment");
            }
        }

        self
    }

    pub const fn with_external(mut self, index: u16, msaa: bool) -> Self {
        if msaa {
            self.msaa = Resource::External(index);
        } else {
            self.non_msaa = Resource::External(index);
        }

        self
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Resource {
    /// Automatically allocate the resource for this attachment.
    Auto,
    /// Use the specified resource for this attachment.
    External(u16),
    /// Use the same resource as the one provided by the dependency.
    Dependency(Dependency),
}

impl From<Dependency> for Resource {
    fn from(dep: Dependency) -> Resource {
        Resource::Dependency(dep)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum NodeKind {
    Render,
    Compute,
    Transfer,
}

// TODO: clearing
pub struct NodeDescriptor<'l> {
    pub kind: NodeKind,
    pub label: Option<&'static str>,
    pub task: Option<TaskId>,
    pub reads: &'l[Dependency],
    pub attachments: &'l[ColorAttachment],
    pub depth_stencil: Option<(Resource, bool, bool)>,
    pub msaa_resolve_target: Option<Resource>,
    pub msaa: bool,
    pub size: Option<SurfaceIntSize>,
}

impl NodeDescriptor<'static> {
    pub fn new() -> Self {
        NodeDescriptor {
            kind: NodeKind::Render, // TODO
            label: None,
            task: None,
            reads: &[],
            attachments: &[],
            depth_stencil: None,
            msaa_resolve_target: None,
            msaa: false,
            size: None,
        }
    }
}

impl<'l> NodeDescriptor<'l> {

    pub fn read<'a>(self, inputs: &'a [Dependency]) -> NodeDescriptor<'a>
    where 'l : 'a
    {
        NodeDescriptor {
            reads: inputs,
            .. self
        }
    }

    pub fn attachments<'a>(self, attachments: &'a [ColorAttachment]) -> NodeDescriptor<'a>
    where 'l : 'a
    {
        NodeDescriptor {
            attachments,
            .. self
        }
    }

    pub fn depth_stencil(self, attachment: Resource, depth: bool, stencil: bool) -> Self {
        NodeDescriptor {
            depth_stencil: Some((attachment, depth, stencil)),
            .. self
        }
    }

    pub fn task(self, task: TaskId) -> Self {
        NodeDescriptor {
            task: Some(task),
            .. self
        }
    }

    pub fn label(self, label: &'static str) -> Self {
        NodeDescriptor {
            label: Some(label),
            .. self
        }
    }

    pub fn msaa_resolve(self, attachment: Resource) -> Self {
        NodeDescriptor {
            msaa_resolve_target: Some(attachment),
            .. self
        }
    }

    pub fn msaa(self, msaa: bool) -> Self {
        NodeDescriptor {
            msaa,
            .. self
        }
    }

    pub fn size(self, size: SurfaceIntSize) -> Self {
        NodeDescriptor {
            size: Some(size),
            .. self
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OutputKind {
    ColorAttachment,
    DepthStencilAttachment,
    MsaaResolveTarget,
    Other,
}

bitflags! {
    #[derive(Copy, Clone, Debug)]
    pub struct ResourceFlags: u8 {
        /// Allocate only if used.
        const LAZY = 1 << 0;
        const LOAD = 1 << 1;
        const CLEAR = 1 << 2;
    }
}

pub struct CommandContext<'l> {
    pub encoder: &'l mut wgpu::CommandEncoder,
    pub renderers: &'l [&'l dyn Renderer],
    pub resources: &'l GpuResources,
    pub bindings: &'l dyn BindingResolver,
    pub render_pipelines: &'l RenderPipelines,
    pub stats: &'l mut RenderStats,
}

pub trait Command: std::fmt::Debug {
    fn execute(&self, ctx: &mut CommandContext);
}

#[derive(Debug)]
pub struct CommandList<'l> {
    cmds: Vec<Box<dyn Command + 'l>>,
}

impl<'l> CommandList<'l> {
    pub fn new() -> Self {
        CommandList { cmds: Vec::with_capacity(128) }
    }

    pub fn push(&mut self, cmd: Box<dyn Command + 'l>) {
        self.cmds.push(cmd)
    }

    pub fn execute(&self, ctx: &mut CommandContext) {
        for cmd in &self.cmds {
            cmd.execute(ctx);
        }
    }
}
