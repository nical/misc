use crate::batching::{BatchId, Batcher};
use crate::graph::PassRenderContext;
use crate::render_task::{RenderTaskHandle, RenderTaskInfo};
use crate::shading::{DepthMode, RenderPipelines, StencilMode, SurfaceDrawConfig, SurfaceKind};
use crate::path::FillRule;
use crate::resources::GpuResources;
use crate::units::SurfaceIntSize;
use crate::{BindingResolver, BindingsId, RenderContext, Renderer, RendererStats};
use std::ops::Range;
use std::time::Instant;

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

// Parameter for the renderers's bathing methods.
// For now it matches the RenderPassBuilder it is created from
// but it may become a subset of it.
pub struct RenderPassContext<'l> {
    pub z_indices: &'l mut ZIndices,
    pub batcher: &'l mut Batcher,
    pub config: RenderPassConfig,
    pub render_task: RenderTaskHandle,
}

pub(crate) struct RenderPassBuilder {
    z_indices: ZIndices,
    config: RenderPassConfig,
    pub(crate) size: SurfaceIntSize,
    render_task: RenderTaskHandle,
    pub(crate) batcher: Batcher,
}

impl RenderPassBuilder {
    pub fn new() -> Self {
        RenderPassBuilder {
            z_indices: ZIndices::default(),
            config: RenderPassConfig::default(),
            size: SurfaceIntSize::new(0, 0),
            batcher: Batcher::new(),
            render_task: RenderTaskHandle::INVALID,
        }
    }

    pub fn ctx(&mut self) -> RenderPassContext {
        RenderPassContext {
            z_indices: &mut self.z_indices,
            batcher: &mut self.batcher,
            config: self.config,
            render_task: self.render_task,
        }
    }

    pub fn begin(&mut self, render_task: &RenderTaskInfo, config: RenderPassConfig) {
        self.z_indices.clear();
        self.config = config;
        self.size = render_task.bounds.size();
        self.render_task = render_task.handle;
        self.batcher.begin(&render_task);
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

// TODO: better name. This is the information that is generated by the graph.
pub struct RenderPassIo {
    pub label: Option<&'static str>,
    pub color_attachments: [ColorAttachment; 3],
    pub depth_stencil_attachment: Option<BindingsId>,
}

#[derive(Copy, Clone, Debug)]
pub struct ColorAttachment {
    pub non_msaa: Option<BindingsId>,
    pub msaa: Option<BindingsId>,
    pub flags: AttathchmentFlags,
}

pub struct BuiltRenderPass {
    batches: Vec<BatchId>,
    config: RenderPassConfig,
    size: SurfaceIntSize,
}

impl BuiltRenderPass {
    pub(crate) fn empty() -> Self {
        BuiltRenderPass {
            batches: Vec::new(),
            config: RenderPassConfig::default(),
            size: SurfaceIntSize::new(0, 0),
        }
    }

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

    pub fn render(
        &self,
        io: &RenderPassIo,
        ctx: &mut PassRenderContext
    ) {
        let mut wgpu_attachments = Vec::new();

        let msaa = self.config().msaa();
        for attachment in &io.color_attachments {
            let view;
            let resolve_target;
            if msaa {
                view = attachment.msaa.map(|id| ctx.bindings.resolve_attachment(id).unwrap());
                resolve_target = attachment.non_msaa.map(|id| ctx.bindings.resolve_attachment(id).unwrap());
            } else {
                view = attachment.non_msaa.map(|id| ctx.bindings.resolve_attachment(id).unwrap());
                resolve_target = None;
            }

            if let Some(view) = view {
                wgpu_attachments.push(Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: if attachment.flags.load {
                            wgpu::LoadOp::Load
                        } else {
                            wgpu::LoadOp::Clear(wgpu::Color::BLACK)
                        },
                        store: if attachment.flags.store {
                            wgpu::StoreOp::Store
                        } else {
                            wgpu::StoreOp::Discard
                        },
                    },
                    depth_slice: None,
                }))
            } else {
                wgpu_attachments.push(None);
            }
        }
        while let Some(attachment) = wgpu_attachments.last() {
            if attachment.is_none() {
                wgpu_attachments.pop();
            } else {
                break;
            }
        }

        let depth_stencil_attachment = io.depth_stencil_attachment.map(|id| {
            let view = ctx.bindings.resolve_attachment(id).unwrap();
            wgpu::RenderPassDepthStencilAttachment {
                view,
                depth_ops: if self.config.depth {
                    Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Discard,
                    })
                } else {
                    None
                },
                stencil_ops: if self.config.stencil {
                    Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(128),
                        store: wgpu::StoreOp::Discard,
                    })
                } else {
                    None
                },
            }
        });

        let pass_descriptor = &wgpu::RenderPassDescriptor {
            label: io.label,
            color_attachments: &wgpu_attachments,
            depth_stencil_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
        };

        let mut wgpu_pass = ctx.encoder.begin_render_pass(&pass_descriptor);

        self.render_in_pass(
            ctx.renderers,
            ctx.resources,
            ctx.bindings,
            ctx.render_pipelines,
            &mut wgpu_pass,
            ctx.gpu_profiler,
            ctx.stats,
        );
    }

    pub fn render_in_pass<'pass, 'resources: 'pass>(
        &self,
        renderers: &[&'resources dyn Renderer],
        resources: &'resources GpuResources,
        bindings: &'resources dyn BindingResolver,
        render_pipelines: &'resources RenderPipelines,
        wgpu_pass: &mut wgpu::RenderPass<'pass>,
        gpu_profiler: &'pass mut wgpu_profiler::GpuProfiler,
        stats: &mut [RendererStats],
    ) {
        if self.batches.is_empty() {
            return;
        }

        let base_bind_group = resources.common.get_base_bindgroup();
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
            let renderer_stats = &mut stats[renderer as usize];
            renderers[renderer as usize].render(
                &self.batches[start..end],
                &self.config,
                RenderContext {
                    render_pipelines,
                    bindings,
                    resources,
                    stats: renderer_stats,
                    gpu_profiler,
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct AttathchmentFlags {
    pub load: bool,
    pub store: bool,
}
