use crate::render_pass::{BuiltRenderPass, RenderPass, RenderPassBuilder};
use crate::gpu::{GpuStore, GpuStreams};
use crate::graph::{Resource, ColorAttachment, Dependency, NodeDescriptor, NodeId, NodeKind, RenderGraph, TaskId};
use crate::units::SurfaceIntSize;
use crate::{transform::Transforms, SurfaceKind, RenderPassConfig};


pub struct Frame {
    pub gpu_store: GpuStore,
    pub vertices: GpuStore,
    pub indices: GpuStreams,
    pub instances: GpuStreams,
    pub transforms: Transforms,
    pub(crate) graph: RenderGraph,
    pub(crate) built_render_passes: Vec<Option<BuiltRenderPass>>,
    // pub allocator: FrameAllocator, // TODO
    index: u32,
}

impl Frame {
    pub(crate) fn new(
        index: u32,
        gpu_store: GpuStore,
        vertices: GpuStore,
        indices: GpuStreams,
        instances: GpuStreams,
    ) -> Self {
        Frame {
            gpu_store,
            vertices,
            indices,
            instances,
            transforms: Transforms::new(),
            graph: RenderGraph::new(),
            built_render_passes: Vec::new(),
            index,
        }
    }

    pub fn begin_render_pass(&mut self, descriptor: RenderNodeDescriptor) -> RenderPass {
        let task_id = TaskId(self.built_render_passes.len() as u64);
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

        pass.begin(
            descriptor.size.unwrap(),
            RenderPassConfig {
                depth,
                stencil,
                msaa: descriptor.msaa,
                attachments: kind,
            }
        );

        let node_id = self.graph.add_node(&descriptor);
        RenderPass::new(pass, node_id)
    }

    pub fn end_render_pass(&mut self, pass: RenderPass) {
        let index = self.graph.get_task_id(pass.node_id()).0 as usize;
        while self.built_render_passes.len() <= index {
            self.built_render_passes.push(None);
        }
        self.built_render_passes[index] = Some(pass.end());
    }

    pub fn add_dependency(&mut self, node: NodeId, dep: Dependency) {
        self.graph.add_dependency(node, dep);
    }

    pub fn add_root(&mut self, root: Dependency) {
        self.graph.add_root(root);
    }

    pub fn index(&self) -> u32 {
        self.index
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
