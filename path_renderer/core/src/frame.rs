use crate::context::{BuiltRenderPass, RenderPassBuilder, RenderPassContext};
use crate::gpu::GpuStore;
use crate::render_graph::{Dependency, NodeDescriptor, NodeId, RenderGraph, TaskId};
use crate::{transform::Transforms, SurfaceKind, SurfacePassConfig};



pub struct Frame {
    pub gpu_store: GpuStore,
    pub transforms: Transforms,
    pub(crate) graph: RenderGraph,
    pub(crate) built_render_passes: Vec<Option<BuiltRenderPass>>,
    // pub allocator: FrameAllocator, // TODO
    index: u32,
}

pub struct RenderSurface {
    node: NodeId,
    pass: RenderPassBuilder,
}

impl RenderSurface {
    pub fn node_id(&self) -> NodeId {
        self.node
    }

    pub fn ctx(&mut self) -> RenderPassContext {
        self.pass.ctx()
    }
}

impl Frame {
    pub(crate) fn new(index: u32) -> Self {
        Frame {
            index,
            graph: RenderGraph::new(),
            built_render_passes: Vec::new(),
            gpu_store: GpuStore::new(512),
            transforms: Transforms::new(),
        }
    }

    pub fn begin_render_surface(&mut self, mut descriptor: NodeDescriptor) -> RenderSurface {
        let task_id = TaskId(self.built_render_passes.len() as u64);
        descriptor = descriptor.task(task_id);
        let (depth, stencil) = descriptor
            .depth_stencil
            .map(|ds| (ds.1, ds.2))
            .unwrap_or((false, false));
        let mut pass = RenderPassBuilder::new();
        pass.begin(
            descriptor.size.unwrap(),
            SurfacePassConfig {
                depth,
                stencil,
                msaa: descriptor.msaa,
                kind: SurfaceKind::Color, // TODO
            }
        );

        RenderSurface {
            pass,
            node: self.graph.add_node(&descriptor),
        }
    }

    pub fn end_render_surface(&mut self, mut surface: RenderSurface) {
        let index = self.graph.get_task_id(surface.node).0 as usize;
        while self.built_render_passes.len() <= index {
            self.built_render_passes.push(None);
        }
        self.built_render_passes[index] = Some(surface.pass.end());
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
