use crate::batching::{BatchId, Batcher};
use crate::graph::{Command, CommandContext, TaskId};
use crate::render_task::{RenderTaskHandle, RenderTaskInfo};
use crate::shading::{DepthMode, RenderPipelines, StencilMode, SurfaceDrawConfig, SurfaceKind};
use crate::instance::RenderStats;
use crate::path::FillRule;
use crate::resources::GpuResources;
use crate::units::SurfaceIntSize;
use crate::{BindingResolver, BindingsId, RenderContext, Renderer};
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

        let base_bind_group = resources.graph.get_base_bindgroup();
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct AttathchmentFlags {
    pub load: bool,
    pub store: bool,
}

pub struct RenderCommand<'l> {
    label: Option<&'static str>,
    commands: &'l RenderCommands,
    built_pass: u32,
    // [(non-msaa, msaa, flags); 3]
    color_attachments: [(Option<BindingsId>, Option<BindingsId>, AttathchmentFlags); 3],
    depth_stencil_attachment: Option<BindingsId>,
}

impl<'l> std::fmt::Debug for RenderCommand<'l> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(label) = self.label {
            write!(f, "RenderCommand({})", label)
        } else {
            write!(f, "RenderCommand")
        }
    }
}

impl<'l> Command for RenderCommand<'l> {
    fn execute(&self, ctx: &mut CommandContext) {
        let mut wgpu_attachments = Vec::new();

        let built_pass = &self.commands.built_render_passes[self.built_pass as usize].as_ref().unwrap();
        let msaa = built_pass.config().msaa();
        for (non_msaa_attachment, msaa_attachment, flags) in &self.color_attachments {
            let view;
            let resolve_target;
            if msaa {
                view = msaa_attachment.map(|id| ctx.bindings.resolve_attachment(id).unwrap());
                resolve_target = non_msaa_attachment.map(|id| ctx.bindings.resolve_attachment(id).unwrap());
            } else {
                view = non_msaa_attachment.map(|id| ctx.bindings.resolve_attachment(id).unwrap());
                resolve_target = None;
            }

            if let Some(view) = view {
                wgpu_attachments.push(Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: if flags.load {
                            wgpu::LoadOp::Load
                        } else {
                            wgpu::LoadOp::Clear(wgpu::Color::BLACK)
                        },
                        store: if flags.store {
                            wgpu::StoreOp::Store
                        } else {
                            wgpu::StoreOp::Discard
                        },
                    },
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

        let depth_stencil_attachment = self.depth_stencil_attachment.map(|id| {
            let view = ctx.bindings.resolve_attachment(id).unwrap();
            wgpu::RenderPassDepthStencilAttachment {
                view,
                depth_ops: if built_pass.config.depth {
                    Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Discard,
                    })
                } else {
                    None
                },
                stencil_ops: if built_pass.config.stencil {
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
            label: self.label,
            color_attachments: &wgpu_attachments,
            depth_stencil_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
        };

        let mut wgpu_pass = ctx.encoder.begin_render_pass(&pass_descriptor);

        built_pass.render(
            ctx.renderers,
            ctx.resources,
            ctx.bindings,
            ctx.render_pipelines,
            &mut wgpu_pass,
            ctx.stats,
        );
    }
}

pub struct RenderCommands {
    built_render_passes: Vec<Option<BuiltRenderPass>>,
    next_id: u64,
}

impl RenderCommands {
    pub fn new() -> Self {
        RenderCommands { built_render_passes: Vec::with_capacity(128), next_id: 0, }
    }

    pub fn next_render_pass_id(&mut self) -> TaskId {
        let id = TaskId(self.next_id);
        self.next_id += 1;

        id
    }

    pub fn add_built_render_pass(&mut self, id: TaskId, pass: BuiltRenderPass) {
        let index = id.0 as usize;
        if self.built_render_passes.len() <= index {
            let n = index + 1 - self.built_render_passes.len();
            self.built_render_passes.reserve(n);
        }
        while self.built_render_passes.len() <= index {
            self.built_render_passes.push(None);
        }
        self.built_render_passes[index] = Some(pass);
    }

    pub fn passes(&self) -> &[Option<BuiltRenderPass>] {
        &self.built_render_passes
    }

    pub fn create_command<'l>(
        &'l self,
        label: Option<&'static str>,
        built_pass: u32,
        color_attachments: &[(Option<BindingsId>, Option<BindingsId>, AttathchmentFlags); 3],
        depth_stencil_attachment: Option<BindingsId>,
    ) -> Box<dyn Command + 'l> {
        Box::new(RenderCommand {
            label,
            commands: self,
            built_pass,
            color_attachments: *color_attachments,
            depth_stencil_attachment,
        })
    }
}
