use crate::render_pass::{RenderPassBuilder, RenderPassContext, RenderPasses};
use crate::render_task::{FrameAtlasAllocator, RenderTaskData, RenderTaskHandle, RenderTaskInfo};
use crate::units::{SurfaceIntRect, SurfaceIntSize, SurfaceRect, SurfaceVector};
use crate::{gpu::GpuBufferWriter, RenderPassConfig, SurfaceKind};
use crate::graph::{ColorAttachment, Dependency, NodeDescriptor, NodeKind, Resource, Slot, TaskId};
use crate::graph::{FrameGraph, Node, NodeDependency};

// TODO: Should this be in graph/mod.rs?
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

    pub(crate)fn to_node_descriptor(self, task: TaskId) -> NodeDescriptor<'l> {
        NodeDescriptor {
            kind: NodeKind::Render,
            label: self.label,
            task: Some(task),
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

    pub fn finish(mut self, render_nodes: &mut RenderNodes) {
        // TODO: instead of doing a non-trivial amount of work while the lock is held,
        // it may be better to enqueue everything and do the work before RenderGraph::schedule.
        let built = self.pass.end();
        render_nodes.passes.add_built_render_pass(self.node.task_id(), built);

        let mut guard = self.node.graph.lock().unwrap();
        // TODO
        guard.graph.add_dependencies(self.node.node_id(), &self.node.dependencies);
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
    inner: RenderNode,
    atlas: FrameAtlasAllocator,
}

impl AtlasRenderNode {
    pub fn allocate(&mut self, f32_buffer: &mut GpuBufferWriter, rect: &SurfaceIntRect) -> Option<RenderPassContext> {
        let task_info = self.atlas.allocate(f32_buffer, rect)?;
        self.inner.pass.batcher.set_render_task(&task_info);

        Some(self.inner.pass.ctx())
    }

    pub fn finish(self, render_nodes: &mut RenderNodes) {
        self.inner.finish(render_nodes);
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
}

pub struct RenderNodes {
    pub passes: RenderPasses,
    pub graph: FrameGraph,
}

impl RenderNodes {
    pub fn new(graph: &FrameGraph) -> Self {
        RenderNodes {
            passes: RenderPasses::new(),
            graph: graph.new_ref(),
        }
    }

    pub fn add_node(&mut self, f32_buffer: &mut GpuBufferWriter, descriptor: RenderNodeDescriptor) -> RenderNode {
        let task_id = self.passes.next_render_pass_id();
        let descriptor = descriptor.to_node_descriptor(task_id);

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

        let size = descriptor.size.unwrap().to_f32();
        let render_task = RenderTaskHandle(f32_buffer.push(RenderTaskData {
            rect: SurfaceRect::from_size(size),
            content_offset: SurfaceVector::zero(),
            rcp_target_width: 1.0 / size.width,
            rcp_target_height: 1.0 / size.height,
        }));

        let task_info = RenderTaskInfo {
            bounds: SurfaceIntRect::from_size(descriptor.size.unwrap()),
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

        RenderNode::new(pass, self.graph.add_node(&descriptor, task_id))
    }
}
