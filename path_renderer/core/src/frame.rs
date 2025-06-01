use std::sync::{Arc, Mutex};

use guillotiere::AtlasAllocator;
use smallvec::SmallVec;

use crate::render_pass::{RenderCommands, RenderPassBuilder, RenderPassContext};
use crate::gpu::{GpuBuffer, GpuBufferWriter, GpuStreams};
use crate::graph::{ColorAttachment, Dependency, NodeDescriptor, NodeId, NodeKind, RenderGraph, Resource, Slot, TaskId};
use crate::render_task::{RenderTaskData, RenderTaskInfo, RenderTaskHandle};
use crate::units::{SurfaceIntRect, SurfaceIntSize, SurfaceRect, SurfaceVector};
use crate::{transform::Transforms, SurfaceKind, RenderPassConfig};

type RenderGraphRef = Arc<Mutex<FrameGraphInner>>;

pub struct FrameGraph {
    pub(crate) inner: RenderGraphRef,
}

pub struct Frame {
    pub f32_buffer: GpuBuffer,
    pub u32_buffer: GpuBuffer,
    pub vertices: GpuBuffer,
    pub indices: GpuStreams,
    pub instances: GpuStreams,
    pub transforms: Transforms,
    pub graph: FrameGraph,
    // pub allocator: FrameAllocator, // TODO
    index: u32,
}

pub(crate) struct FrameGraphInner {
    pub graph: RenderGraph,
    pub render_passes: RenderCommands,
}

impl Frame {
    pub(crate) fn new(
        index: u32,
        f32_buffer: GpuBuffer,
        u32_buffer: GpuBuffer,
        vertices: GpuBuffer,
        indices: GpuStreams,
        instances: GpuStreams,
    ) -> Self {
        Frame {
            f32_buffer,
            u32_buffer,
            vertices,
            indices,
            instances,
            transforms: Transforms::new(),
            graph: FrameGraph {
                inner: Arc::new(Mutex::new(FrameGraphInner {
                    graph: RenderGraph::new(),
                    render_passes: RenderCommands::new(),
                })),
            },
            index,
        }
    }

    pub fn index(&self) -> u32 {
        self.index
    }
}

impl FrameGraph {
    pub fn add_render_node(&mut self, f32_buffer: &mut GpuBufferWriter, descriptor: RenderNodeDescriptor) -> RenderNode {
        let mut guard = self.inner.lock().unwrap();

        let task_id = guard.render_passes.next_render_pass_id();
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

        let node = Node {
            id: guard.graph.add_node(&descriptor),
            task: task_id,
            graph: self.inner.clone(),
            dependencies: SmallVec::new(),
        };

        RenderNode::new(pass, node)
    }

    // TODO: mark root as a flag in the node instead to avoid accessing/locking
    // the graph.
    pub fn add_root<'l>(&mut self, root: impl Into<NodeDependency<'l>>) {
        self.inner.lock().unwrap().graph.add_root(root.into().as_graph_dependency());
    }
}

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

    fn to_node_descriptor(self, task: TaskId) -> NodeDescriptor<'l> {
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

pub struct ComputeNodeDescriptor<'l> {
    pub label: Option<&'static str>,
    pub reads: &'l[Dependency],
    pub resources: &'l[Resource],
}

pub struct Node {
    id: NodeId,
    task: TaskId,
    graph: RenderGraphRef,
    dependencies: SmallVec<[Dependency; 4]>,
}

impl Node {
    pub fn task_id(&self) -> TaskId {
        self.task
    }
}

pub struct NodeDependency<'l> {
    pub(crate) node: &'l Node,
    pub(crate) slot: Slot,
}

impl<'l> NodeDependency<'l> {
    pub(crate) fn as_graph_dependency(&self) -> crate::graph::Dependency {
        crate::graph::Dependency {
            node: self.node.id,
            slot: self.slot,
        }
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
        self.graph.lock().unwrap().graph.finish_node(self.id);
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

    pub fn finish(self) {
        // Just a convenience to trigger Drop.
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

impl Drop for RenderNode {
    fn drop(&mut self) {
        // TODO: instead of doing a non-trivial amount of work while the lock is held,
        // it may be better to enqueue everything and do the work before RenderGraph::schedule.
        let built = self.pass.end();
        let mut guard = self.node.graph.lock().unwrap();

        guard.graph.set_dependencies(self.node.id, std::mem::take(&mut self.node.dependencies));
        guard.render_passes.add_built_render_pass(self.node.task, built);
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
    atlas: AtlasAllocator,
}

impl AtlasRenderNode {
    pub fn allocate(&mut self, f32_buffer: &mut GpuBufferWriter, rect: &SurfaceIntRect) -> Option<RenderPassContext> {
        let alloc = self.atlas.allocate(rect.size().cast_unit())?;
        let offset = alloc.rectangle.min.cast_unit() - rect.min;

        let target = self.inner.pass.size.to_f32();
        let data = RenderTaskData {
            rect: alloc.rectangle.to_f32().cast_unit(),
            content_offset: offset.to_f32(),
            rcp_target_width: 1.0 / target.width,
            rcp_target_height: 1.0 / target.height,
        };

        let handle = RenderTaskHandle(f32_buffer.push(data));

        self.inner.pass.batcher.set_render_task(&RenderTaskInfo {
            bounds: *rect,
            offset: offset.to_f32(),
            handle,
        });

        Some(self.inner.pass.ctx())
    }

    pub fn finish(self) {
        // Just a convenience to trigger Drop.
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
