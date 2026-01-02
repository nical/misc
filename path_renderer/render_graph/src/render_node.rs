use crate::{ColorAttachment, Dependency, NodeDescriptor, NodeKind, Resource, Slot};
use crate::{Node, NodeDependency};
use core::render_pass::{RenderPassBuilder, RenderPassContext};
use core::render_task::RenderTask;
use core::units::{SurfaceIntRect, SurfaceIntSize};
use core::{gpu::GpuBufferWriter};

pub struct RenderNodeDescriptor<'l> {
    pub label: Option<&'static str>,
    pub reads: &'l[Dependency],
    pub attachments: &'l[ColorAttachment],
    pub depth_stencil: Option<(Resource, bool, bool)>,
    pub msaa_resolve_target: Option<Resource>,
    pub msaa: bool,
    pub size: SurfaceIntSize,
}

impl RenderNodeDescriptor<'static> {
    pub fn new(size: SurfaceIntSize) -> Self {
        RenderNodeDescriptor {
            label: None,
            reads: &[],
            attachments: &[],
            depth_stencil: None,
            msaa_resolve_target: None,
            msaa: false,
            size,
        }
    }
}

impl<'l> RenderNodeDescriptor<'l> {
    pub fn read<'a>(self, inputs: &'a [Dependency]) -> RenderNodeDescriptor<'a>
    where 'l : 'a
    {
        RenderNodeDescriptor { reads: inputs, .. self }
    }

    pub fn attachments<'a>(self, attachments: &'a [ColorAttachment]) -> RenderNodeDescriptor<'a>
    where 'l : 'a
    {
        RenderNodeDescriptor { attachments, .. self }
    }

    pub fn depth_stencil(self, attachment: Resource, depth: bool, stencil: bool) -> Self {
        RenderNodeDescriptor {
            depth_stencil: Some((attachment, depth, stencil)),
            .. self
        }
    }

    pub fn label(self, label: &'static str) -> Self {
        RenderNodeDescriptor { label: Some(label), .. self }
    }

    pub fn msaa_resolve(self, attachment: Resource) -> Self {
        RenderNodeDescriptor { msaa_resolve_target: Some(attachment), .. self }
    }

    pub fn msaa(self, msaa: bool) -> Self {
        RenderNodeDescriptor { msaa, .. self }
    }

    pub(crate)fn to_node_descriptor(self) -> NodeDescriptor<'l> {
        NodeDescriptor {
            kind: NodeKind::Render,
            label: self.label,
            reads: self.reads,
            attachments: self.attachments,
            depth_stencil: self.depth_stencil,
            msaa_resolve_target: self.msaa_resolve_target,
            msaa: self.msaa,
            size: Some(self.size),
        }
    }
}

pub struct RenderNode {
    pass: RenderPassBuilder,
    node: Node,
}

impl RenderNode {
    pub(crate) fn new(pass: RenderPassBuilder, node: Node) -> Self {
        RenderNode { pass, node }
    }

    pub fn ctx(&mut self) -> RenderPassContext {
        self.pass.ctx()
    }

    pub fn color(&self, idx: u8) -> NodeDependency {
        NodeDependency {
            node: self.as_ref(),
            slot: Slot::Color(idx),
        }
    }

    pub fn msaa(&self, idx: u8) -> NodeDependency {
        NodeDependency {
            node: self.as_ref(),
            slot: Slot::Msaa(idx),
        }
    }

    pub fn depth_stencil(&self) -> NodeDependency {
        NodeDependency {
            node: self.as_ref(),
            slot: Slot::DepthStencil,
        }
    }

    pub fn read(&mut self, dep: NodeDependency) {
        self.node.read(dep);
    }

    pub fn finish(self) {}
}

impl Drop for RenderNode {
    fn drop(&mut self) {
        let built = self.pass.end();

        // TODO: instead of doing a non-trivial amount of work while the lock is held,
        // it may be better to enqueue everything and do the work before RenderGraph::schedule.
        let id = self.node.node_id();
        let mut guard = self.node.graph.lock().unwrap();
        guard.add_built_render_pass(id, built);
        guard.add_dependencies(id, &self.node.dependencies);
    }
}

impl AsRef<Node> for RenderNode {
    fn as_ref(&self) -> &Node {
        &self.node
    }
}

impl<'l> Into<NodeDependency<'l>> for &'l RenderNode {
    fn into(self) -> NodeDependency<'l> {
        self.color(0)
    }
}

pub struct AtlasRenderNode {
    pub(crate) inner: RenderNode,
    pub(crate) allocator: guillotiere::SimpleAtlasAllocator,
    pub(crate) allocated_px: u32,
}

impl AtlasRenderNode {
    pub fn allocate(&mut self, f32_buffer: &mut GpuBufferWriter, bounds: &SurfaceIntRect) -> Option<RenderPassContext> {
        let alloc = self.allocator.allocate(bounds.size().cast_unit())?;
        let offset = alloc.min.cast_unit().to_f32().to_vector();
        self.allocated_px += alloc.area() as u32;
        let task_info = RenderTask::new_sub_task(f32_buffer, bounds, offset, self.size());
        self.inner.pass.set_render_task(&task_info);

        Some(self.inner.pass.ctx())
    }

    pub fn ctx(&mut self) -> RenderPassContext {
        self.inner.pass.ctx()
    }

    pub fn finish(self) {
        self.inner.finish();
    }

    pub fn color(&self, idx: u8) -> NodeDependency {
        self.inner.color(idx)
    }

    pub fn msaa(&self, idx: u8) -> NodeDependency {
        self.inner.msaa(idx)
    }

    pub fn depth_stencil(&self) -> NodeDependency {
        self.inner.depth_stencil()
    }

    pub fn size(&self) -> SurfaceIntSize {
        self.allocator.size().cast_unit()
    }

    pub fn is_empty(&mut self) -> bool {
        self.allocator.is_empty()
    }

    pub fn occupancy(&self) -> f32 {
        let s = self.size();
        let area = s.width as f32 * s.height as f32;
        self.allocated_px as f32 / area
    }
}
