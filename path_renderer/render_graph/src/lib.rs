#![allow(mismatched_lifetime_syntaxes)]

mod schedule;
mod render_node;

use std::{fmt, sync::{Arc, Mutex}, u16};
use core::bitflags::bitflags;
use core::smallvec::SmallVec;

use core::{BindingsId, RenderPassConfig, SurfaceKind, resources::TextureKind};
use core::gpu::GpuBufferWriter;
use core::instance::{Frame, Passes};
use core::render_pass::RenderPassBuilder;
use core::render_task::{RenderTaskData, RenderTaskHandle, RenderTaskInfo};
use core::units::{SurfaceIntRect, SurfaceIntSize, SurfaceRect, SurfaceVector};

pub use crate::render_node::*;

use schedule::RenderGraph;
pub use schedule::GraphError;

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

// The virtual/physical resource distinction is similar to virtual and physical
// registers: Every output port of a node is its own virtual resource, to which
// is assigned a physical resource when scheduling the graph. Physical resources
// are used by as many nodes as possible to minimize memory usage.

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ColorAttachment {
    pub kind: TextureKind,
    pub non_msaa: Resource,
    pub msaa: Resource,
    pub clear: bool,
}

impl ColorAttachment {
    pub const fn color() -> Self {
        ColorAttachment {
            kind: TextureKind::color(),
            non_msaa: Resource::Auto,
            msaa: Resource::Auto,
            clear: false,
        }
    }

    pub const fn alpha() -> Self {
        ColorAttachment {
            kind: TextureKind::alpha(),
            non_msaa: Resource::Auto,
            msaa: Resource::Auto,
            clear: false,
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

    pub const fn cleared(mut self, clear: bool) -> Self {
        self.clear = clear;

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

pub type GraphSystemId = u16;

// TODO: Do we need both RenderGraph and FrameGraph?

type FrameGraphRef = Arc<Mutex<RenderGraph>>;

pub struct FrameGraph {
    pub(crate) inner: FrameGraphRef,
}

impl FrameGraph {
    pub fn new(_frame: &Frame) -> Self {
        FrameGraph {
            inner: Arc::new(Mutex::new(RenderGraph::new())),
        }
    }

    pub fn new_ref(&self) -> Self {
        FrameGraph { inner: self.inner.clone() }
    }

    // TODO: mark root as a flag in the node instead to avoid accessing/locking
    // the graph.
    pub fn add_root<'l>(&mut self, root: impl Into<NodeDependency<'l>>) {
        self.inner.lock().unwrap().add_root(root.into().as_graph_dependency());
    }

    pub fn add_node(&self, descriptor: &NodeDescriptor) -> Node {
        let mut guard = self.inner.lock().unwrap();

        let node = Node {
            id: guard.add_node(&descriptor),
            graph: self.inner.clone(),
            dependencies: SmallVec::new(),
        };

        node
    }

    pub fn add_render_node(&mut self, f32_buffer: &mut GpuBufferWriter, descriptor: RenderNodeDescriptor) -> RenderNode {
        let descriptor = descriptor.to_node_descriptor();

        let (depth, stencil) = descriptor
            .depth_stencil
            .map(|ds| (ds.1, ds.2))
            .unwrap_or((false, false));
        let mut pass = RenderPassBuilder::new();

        let mut kind = [
            SurfaceKind::None,
            SurfaceKind::None,
            SurfaceKind::None,
        ];
        for (idx, attachment) in descriptor.attachments.iter().enumerate() {
            if attachment.kind.is_color() {
                kind[idx] = SurfaceKind::Color;
            } else if attachment.kind.is_alpha() {
                kind[idx] = SurfaceKind::Alpha;
            }
        }

        let size = descriptor.size.unwrap();
        let sizef = size.to_f32();
        let rect = SurfaceRect::from_size(sizef);
        let render_task = RenderTaskHandle(f32_buffer.push(RenderTaskData {
            clip: rect,
            image_source: rect,
            content_offset: SurfaceVector::zero(),
            rcp_target_width: 1.0 / sizef.width,
            rcp_target_height: 1.0 / sizef.height,
        }));

        let task_info = RenderTaskInfo {
            bounds: SurfaceIntRect::from_size(size),
            target_rect: SurfaceIntRect::from_size(size),
            offset: SurfaceVector::zero(),
            handle: render_task,
        };

        pass.begin(
            &task_info,
            RenderPassConfig {
                depth,
                stencil,
                msaa: descriptor.msaa,
                attachments: kind,
            },
        );

        RenderNode::new(pass, self.add_node(&descriptor))
    }

    pub fn add_atlas_render_node(&mut self, f32_buffer: &mut GpuBufferWriter, descriptor: RenderNodeDescriptor) -> AtlasRenderNode {
        let size = descriptor.size;
        let inner = self.add_render_node(f32_buffer, descriptor);

        AtlasRenderNode {
            inner,
            allocator: guillotiere::SimpleAtlasAllocator::new(size.cast_unit()),
            allocated_px: 0,
        }
    }

    pub fn schedule(&mut self, passes: &mut Passes) -> Result<(), Box<GraphError>> {
        let mut guard = self.inner.lock().unwrap();
        let inner = &mut *guard;

        inner.schedule(passes)
    }
}

pub struct Node {
    id: NodeId,
    pub(crate) graph: FrameGraphRef,
    pub(crate) dependencies: SmallVec<[Dependency; 4]>,
}

impl Node {
    pub fn read(&mut self, dep: NodeDependency) {
        self.dependencies.push(dep.as_graph_dependency())
    }

    pub fn node_id(&self) -> NodeId {
        self.id
    }
}

pub struct NodeDependency<'l> {
    pub(crate) node: &'l Node,
    pub(crate) slot: Slot,
}

impl<'l> NodeDependency<'l> {
    pub(crate) fn as_graph_dependency(&self) -> crate::Dependency {
        crate::Dependency {
            node: self.node.id,
            slot: self.slot,
        }
    }

    pub fn get_binding(&self) -> BindingsId {
        self.node.graph.lock().unwrap().get_binding(self.as_graph_dependency())
    }
}

impl Node {
    pub fn add_dependency<'l>(&mut self, dep: impl Into<NodeDependency<'l>>) {
        let dep = dep.into().as_graph_dependency();
        self.dependencies.push(dep);
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        self.graph.lock().unwrap().finish_node(self.id);
    }
}
