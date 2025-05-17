use crate::batching::{Batcher, BatchId};
use crate::graph::NodeId;
use crate::shading::{DepthMode, RenderPipelines, StencilMode, SurfaceDrawConfig, SurfaceKind};
use crate::instance::RenderStats;
use crate::path::FillRule;
use crate::resources::GpuResources;
use crate::units::{SurfaceIntSize, SurfaceRect};
use crate::{BindingResolver, Renderer, RenderContext};
use std::ops::Range;
use std::time::Instant;

pub struct RenderPass {
    pass: RenderPassBuilder,
    node: NodeId,
}

impl RenderPass {
    pub(crate) fn new(pass: RenderPassBuilder, node: NodeId) -> Self {
        RenderPass { pass, node }
    }

    pub fn node_id(&self) -> NodeId {
        self.node
    }

    pub fn ctx(&mut self) -> RenderPassContext {
        self.pass.ctx()
    }

    pub(crate) fn end(mut self) -> BuiltRenderPass { self.pass.end() }
}

pub type ZIndex = u32;

pub struct ZIndices {
    next: ZIndex,
}

impl ZIndices {
    pub fn new() -> Self {
        ZIndices { next: 1 }
    }

    pub fn push(&mut self) -> ZIndex {
        let result = self.next;
        self.next += 1;

        result
    }

    pub fn push_range(&mut self, count: usize) -> Range<ZIndex> {
        let first = self.next;
        self.next += count as ZIndex;

        first..self.next
    }

    pub fn clear(&mut self) {
        self.next = 1;
    }
}

impl Default for ZIndices {
    fn default() -> Self {
        Self::new()
    }
}

pub type RendererId = u16;

/// The surface parameters for render passes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct RenderPassConfig {
    pub depth: bool,
    pub msaa: bool,
    pub stencil: bool,
    pub attachments: [SurfaceKind; 3],
}

impl Default for RenderPassConfig {
    fn default() -> Self {
        RenderPassConfig {
            depth: false,
            msaa: false,
            stencil: false,
            attachments: [
                SurfaceKind::Color,
                SurfaceKind::None,
                SurfaceKind::None,
            ],
        }
    }
}

impl RenderPassConfig {
    pub fn msaa(&self) -> bool {
        self.msaa
    }
    pub fn depth(&self) -> bool {
        self.depth
    }
    pub fn stencil(&self) -> bool {
        self.stencil
    }
    pub fn depth_or_stencil(&self) -> bool {
        self.depth || self.stencil
    }
    pub fn draw_config(&self, use_depth: bool, stencil: Option<FillRule>) -> SurfaceDrawConfig {
        SurfaceDrawConfig {
            msaa: self.msaa,
            depth: match (use_depth, self.depth) {
                (true, true) => DepthMode::Enabled,
                (false, true) => DepthMode::Ignore,
                (_, false) => DepthMode::None,
            },
            stencil: match (stencil, self.stencil) {
                (None, false) => StencilMode::None,
                (None, true) => StencilMode::Ignore,
                (Some(FillRule::EvenOdd), true) => StencilMode::EvenOdd,
                (Some(FillRule::NonZero), true) => StencilMode::NonZero,
                (Some(_), false) => panic!(
                    "Attempting to use the stencil buffer on a surface that does not have one"
                ),
            },
            attachments: self.attachments,
        }
    }

    pub fn color() -> Self {
        RenderPassConfig {
            msaa: false,
            depth: false,
            stencil: false,
            attachments: [
                SurfaceKind::Color,
                SurfaceKind::None,
                SurfaceKind::None,
            ],
        }
    }

    pub fn alpha() -> Self {
        RenderPassConfig {
            msaa: false,
            depth: false,
            stencil: false,
            attachments: [
                SurfaceKind::Alpha,
                SurfaceKind::None,
                SurfaceKind::None,
            ],
        }
    }

    pub fn with_msaa(mut self, msaa: bool) -> Self {
        self.msaa = msaa;
        self
    }

    pub fn with_stencil(mut self, stencil: bool) -> Self {
        self.stencil = stencil;
        self
    }

    pub fn with_depth(mut self, depth: bool) -> Self {
        self.depth = depth;
        self
    }
}

// Paramater for the renderers's prepare methods.
// For now it matches the RenderPassBuilder it is created from
// but it may become a subset of it.
pub struct RenderPassContext<'l> {
    pub z_indices: &'l mut ZIndices,
    pub batcher: &'l mut Batcher,
    pub config: RenderPassConfig,
}

pub(crate) struct RenderPassBuilder {
    z_indices: ZIndices,
    config: RenderPassConfig,
    size: SurfaceIntSize,
    batcher: Batcher,
}

impl RenderPassBuilder {
    pub fn new() -> Self {
        RenderPassBuilder {
            z_indices: ZIndices::default(),
            config: RenderPassConfig::default(),
            size: SurfaceIntSize::new(0, 0),
            batcher: Batcher::new(),
        }
    }

    pub fn ctx(&mut self) -> RenderPassContext {
        RenderPassContext {
            z_indices: &mut self.z_indices,
            batcher: &mut self.batcher,
            config: self.config,
        }
    }

    pub fn begin(&mut self, size: SurfaceIntSize, config: RenderPassConfig) {
        self.z_indices.clear();
        self.config = config;
        self.size = size;
        self.batcher.begin(&SurfaceRect::from_size(size.to_f32()));
    }

    pub fn end(&mut self) -> BuiltRenderPass {
        let mut batches = Vec::new();
        self.batcher.finish(&mut batches);

        BuiltRenderPass {
            batches,
            config: self.config,
            size: self.size,
        }
    }
}

pub struct BuiltRenderPass {
    batches: Vec<BatchId>,
    config: RenderPassConfig,
    size: SurfaceIntSize,
}

impl BuiltRenderPass {
    pub fn batches(&self) -> &[BatchId] {
        &self.batches
    }

    pub fn config(&self) -> RenderPassConfig {
        self.config
    }

    pub fn surface_size(&self) -> SurfaceIntSize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }

    pub fn render<'pass, 'resources: 'pass>(
        &self,
        pass_index: u16,
        renderers: &[&'resources dyn Renderer],
        resources: &'resources GpuResources,
        bindings: &'resources dyn BindingResolver,
        render_pipelines: &'resources RenderPipelines,
        wgpu_pass: &mut wgpu::RenderPass<'pass>,
        stats: &mut RenderStats,
    ) {
        if self.batches.is_empty() {
            return;
        }

        //if let Some(bind_group) = blit_src {
        //    let common = &resources[common_resources];
        //    wgpu_pass.set_bind_group(0, bind_group, &[]);
        //    wgpu_pass.set_index_buffer(common.quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
        //    wgpu_pass.set_pipeline(if surface_cfg.depth || surface_cfg.stencil {
        //        &common.msaa_blit_with_depth_stencil_pipeline
        //    } else {
        //        &common.msaa_blit_pipeline
        //    });
        //    wgpu_pass.draw_indexed(0..6, 0, 0..1);
        //}

        let base_bind_group = resources.graph.get_base_bindgroup(pass_index);
        wgpu_pass.set_bind_group(0, base_bind_group, &[]);

        // Traverse batches, grouping consecutive items with the same renderer.
        let mut start = 0;
        let mut end = 0;
        let mut renderer = self.batches[0].renderer;
        while start < self.batches.len() {
            let done = end >= self.batches.len();
            if !done && renderer == self.batches[end].renderer {
                end += 1;
                continue;
            }

            let start_time = Instant::now();
            let renderer_stats = &mut stats.renderers[renderer as usize];
            renderers[renderer as usize].render(
                &self.batches[start..end],
                &self.config,
                RenderContext {
                    render_pipelines,
                    bindings,
                    resources,
                    stats: renderer_stats,
                },
                wgpu_pass
            );
            let time = crate::instance::ms(Instant::now() - start_time);
            renderer_stats.render_time += time;

            if done {
                break;
            }
            start = end;
            renderer = self.batches[start].renderer;
        }
    }
}

#[test]
fn surface_draw_config() {
    for msaa in [true, false] {
        for depth in [DepthMode::Enabled, DepthMode::Ignore, DepthMode::None] {
            for stencil in [StencilMode::EvenOdd, StencilMode::NonZero, StencilMode::Ignore, StencilMode::None] {
                for kind in [SurfaceKind::Color, SurfaceKind::Alpha] {
                    let kind = [kind, SurfaceKind::None, SurfaceKind::None];
                    let surface = SurfaceDrawConfig { msaa, depth, stencil, attachments: kind};
                    assert_eq!(SurfaceDrawConfig::from_hash(surface.hash()), surface);
                }
            }
        }
    }
}
