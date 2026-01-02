use wgpu_profiler::GpuProfiler;

use crate::batching::{BatchId, Batcher, ScissorRect};
use crate::render_task::{RenderTaskAdress, RenderTask};
use crate::resources::GpuResources;
use crate::shading::{DepthMode, RenderPipelines, StencilMode, SurfaceDrawConfig, SurfaceKind};
use crate::path::FillRule;
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
    pub render_task: RenderTaskAdress,
}

// TODO: Should this move into render_graph?
pub struct RenderPassBuilder {
    z_indices: ZIndices,
    config: RenderPassConfig,
    pub(crate) size: SurfaceIntSize,
    render_task: RenderTaskAdress,
    pub batcher: Batcher,
}

impl RenderPassBuilder {
    pub fn new() -> Self {
        RenderPassBuilder {
            z_indices: ZIndices::default(),
            config: RenderPassConfig::default(),
            size: SurfaceIntSize::new(0, 0),
            batcher: Batcher::new(),
            render_task: RenderTaskAdress::NONE,
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

    pub fn begin(&mut self, render_task: &RenderTask, config: RenderPassConfig) {
        self.z_indices.clear();
        self.config = config;
        self.size = render_task.bounds.size().to_i32();
        self.render_task = render_task.gpu_address;
        self.batcher.begin(&render_task);
    }

    // TODO: This isn't a very good API.
    pub fn set_render_task(&mut self, render_task: &RenderTask) {
        self.render_task = render_task.gpu_address;
        self.batcher.set_render_task(render_task);
    }

    pub fn end(&mut self) -> BuiltRenderPass {
        let mut batches = Vec::new();
        self.batcher.finish(&mut batches);

        BuiltRenderPass {
            batches,
            config: self.config,
            size: self.size,
            io: RenderPassIo::default(),
        }
    }
}

// TODO: better name. This is the information that is generated by the graph.
pub struct RenderPassIo {
    pub label: Option<&'static str>,
    pub color_attachments: [ColorAttachment; 3],
    pub depth_stencil_attachment: Option<BindingsId>,
}

impl Default for RenderPassIo {
    fn default() -> Self {
        let attachment = ColorAttachment {
            msaa: None,
            non_msaa: None,
            load: false,
            store: true,
            clear: false,
        };
        RenderPassIo {
            label: None,
            color_attachments: [attachment, attachment, attachment],
            depth_stencil_attachment: None,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ColorAttachment {
    pub non_msaa: Option<BindingsId>,
    pub msaa: Option<BindingsId>,
    pub load: bool,
    pub store: bool,
    pub clear: bool,
}

pub struct BuiltRenderPass {
    batches: Vec<BatchId>,
    config: RenderPassConfig,
    size: SurfaceIntSize,
    io: RenderPassIo,
}

impl BuiltRenderPass {
    pub fn empty() -> Self {
        BuiltRenderPass {
            batches: Vec::new(),
            config: RenderPassConfig::default(),
            size: SurfaceIntSize::new(0, 0),
            io: RenderPassIo::default(),
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
        ctx: &mut PassRenderContext,
        renderers: &[&mut dyn Renderer],
    ) {
        let mut commands = RenderCommands::new();
        for batch in &self.batches {
            commands.set_scissor_rect(&batch.scissor);
            let renderer_prepare_start = Instant::now();
            let renderer_idx = batch.renderer as usize;
            renderers[renderer_idx].add_render_commands(*batch, &mut commands);
            let stats = &mut ctx.stats[renderer_idx];
            stats.prepare_time += crate::instance::ms(Instant::now() - renderer_prepare_start);
        }

        commands.end_render_pass();

        commands.render_pass(ctx, renderers, self.size, &self.io, &self.config());
    }

    pub fn set_label(&mut self, label: &'static str) {
        self.io.label = Some(label);
    }

    pub fn set_color_attachments(&mut self, attachments: &[ColorAttachment]) {
        debug_assert!(attachments.len() <= 3);
        for (idx, attachment) in attachments.iter().enumerate() {
            self.io.color_attachments[idx] = *attachment;
        }
    }

    pub fn set_depth_stencil_attachment(&mut self, attachment: Option<BindingsId>) {
        self.io.depth_stencil_attachment = attachment;
    }
}

impl Default for BuiltRenderPass {
    fn default() -> Self {
        Self::empty()
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
pub struct AttachmentFlags {
    pub load: bool,
    pub clear: bool,
    pub store: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct RenderCommandId {
    pub renderer: RendererId,
    pub request_pre_pass: bool,
    /// Spare bits that can be used by the renderer.
    pub flags: u8,
    /// Interpreted by the renderer.
    ///
    /// If Renderer::add_render_commands is not implemented manually,
    /// defaults to the batch index.
    pub index: u32,

    pub scissor: Option<ScissorRect>,
}

impl From<BatchId> for RenderCommandId {
    fn from(batch: BatchId) -> Self {
        RenderCommandId {
            renderer: batch.renderer,
            request_pre_pass: false,
            flags: 0,
            index: batch.index,
            scissor: None,
        }
    }
}

struct SubPass {
    ordered: Range<usize>,
    order_independent: Range<usize>,
    pre_passes: Range<usize>,
}

pub struct RenderCommands {
    ordered: Vec<RenderCommandId>,
    order_independent: Vec<RenderCommandId>,
    pre_passes: Vec<RenderCommandId>,
    sub_passes: Vec<SubPass>,
    requested_pre_pass: Vec<bool>,
    current_pass_start: u16,
    current_scissor: Option<ScissorRect>,
}

impl RenderCommands {
    pub fn new() -> Self {
        RenderCommands {
            ordered: Vec::with_capacity(256),
            order_independent: Vec::with_capacity(64),
            sub_passes: Vec::new(),
            pre_passes: Vec::new(),
            requested_pre_pass: Vec::new(),
            current_pass_start: 0,
            current_scissor: None,
        }
    }

    pub fn push(&mut self, mut command: RenderCommandId) {
        if command.request_pre_pass {
            self.request_pre_pass(command);
        }
        command.scissor = self.current_scissor;
        self.ordered.push(command);
    }

    pub fn push_order_independent(&mut self, mut command: RenderCommandId) {
        if command.request_pre_pass {
            self.request_pre_pass(command);
        }
        command.scissor = self.current_scissor;
        self.order_independent.push(command)
    }

    pub(crate) fn set_scissor_rect(&mut self, scissor: &Option<ScissorRect>) {
        self.current_scissor = *scissor;
    }

    pub fn end_render_pass(&mut self) {
        self.current_pass_start = self.sub_passes.len() as u16;
        self.finish_sub_pass();
    }

    fn request_pre_pass(&mut self, command: RenderCommandId) {
        let renderer = command.renderer as usize;
        while self.requested_pre_pass.len() <= renderer {
            self.requested_pre_pass.push(false);
        }
        if self.requested_pre_pass[renderer] {
            self.finish_sub_pass();
            self.pre_passes.push(command)
        }

        self.requested_pre_pass[renderer] = true;
    }

    fn finish_sub_pass(&mut self) {
        let (o_start, oi_start, pp_start) = self.sub_passes.last()
            .map(|sp| (sp.ordered.end, sp.order_independent.end, sp.pre_passes.end))
            .unwrap_or((0, 0, 0));
        let o_end = self.ordered.len();
        let oi_end = self.order_independent.len();
        let pp_end = self.pre_passes.len();

        let sub_pass = SubPass {
            ordered: o_start..o_end,
            order_independent: oi_start..oi_end,
            pre_passes: pp_start..pp_end,
        };

        if sub_pass.ordered.is_empty()
            && sub_pass.order_independent.is_empty()
            && sub_pass.pre_passes.is_empty() {
            return;
        }

        if !sub_pass.order_independent.is_empty() {
            // Process order-independent batches front-to-back. This is
            // done under the assumption that we are using the depth buffer
            // to ensure order-independence, and therefore rendering font
            // to back minimizes the number of pixels that will be shaded.
            self.order_independent[sub_pass.order_independent.clone()].reverse();
        }

        self.sub_passes.push(sub_pass);
    }

    pub fn render_pass(
        &self,
        ctx: &mut PassRenderContext,
        renderers: &[&mut dyn Renderer],
        size: SurfaceIntSize,
        io: &RenderPassIo,
        config: &RenderPassConfig,
    ) {
        for sub_pass in &self.sub_passes {
            self.render_sub_pass(ctx, renderers, sub_pass, size, io, config);
        }
    }

    fn render_pre_pass(
        &self,
        _pre_pass: &RenderCommandId,
        _ctx: &mut PassRenderContext,
    ) {
        // TODO
    }

    fn render_sub_pass(
        &self,
        ctx: &mut PassRenderContext,
        renderers: &[&mut dyn Renderer],
        sub_pass: &SubPass,
        size: SurfaceIntSize,
        io: &RenderPassIo,
        config: &RenderPassConfig,
    ) {
        for pre_pass in &self.pre_passes[sub_pass.pre_passes.clone()] {
            self.render_pre_pass(pre_pass, ctx);
        }

        let mut wgpu_attachments = Vec::new();

        let msaa = config.msaa();
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
                        load: if attachment.load {
                            wgpu::LoadOp::Load
                        } else if attachment.clear {
                            wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0, g: 0.0, b: 0.0, a: 0.0,
                            })
                        } else {
                            wgpu::LoadOp::DontCare(unsafe {
                                wgpu::LoadOpDontCare::enabled()
                            })
                        },
                        store: if attachment.store {
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
                depth_ops: if config.depth {
                    Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Discard,
                    })
                } else {
                    None
                },
                stencil_ops: if config.stencil {
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
            multiview_mask: None,
        };

        let mut wgpu_pass = ctx.encoder.begin_render_pass(&pass_descriptor);

        let base_bind_group = ctx.resources.common.get_base_bindgroup();
        wgpu_pass.set_bind_group(0, base_bind_group, &[]);

        let mut current_scissor = None;
        let set_scissor = &mut |scissor: &Option<ScissorRect>, wgpu_pass: &mut wgpu::RenderPass| {
            let scissor = match scissor {
                Some(r) => *r,
                None => ScissorRect {
                    x: 0, y: 0,
                    w: size.width as u16, h: size.height as u16,
                }
            };
            wgpu_pass.set_scissor_rect(
                scissor.x as u32, scissor.y as u32,
                scissor.w as u32, scissor.h as u32,
            );
        };

        let mut prev_renderer = None;
        let mut start_idx = sub_pass.ordered.start;

        for (idx, command) in self.order_independent[sub_pass.order_independent.clone()].iter().enumerate() {
            let scissor_changed = command.scissor != current_scissor;
            if (Some(command.renderer) != prev_renderer || scissor_changed) && prev_renderer.is_some() {
                let commands = &self.order_independent[start_idx..idx];
                let renderer_idx = prev_renderer.unwrap() as usize;
                self.render_commands(
                    commands,
                    renderer_idx,
                    renderers,
                    ctx.render_pipelines,
                    ctx.resources,
                    ctx.bindings,
                    ctx.stats,
                    config,
                    ctx.gpu_profiler,
                    &mut wgpu_pass,
                );
                start_idx = idx;
            }

            if scissor_changed {
                set_scissor(&command.scissor, &mut wgpu_pass);
            }

            current_scissor = command.scissor;
            prev_renderer = Some(command.renderer);
        }

        if start_idx != sub_pass.order_independent.end {
            let commands = &self.order_independent[start_idx..sub_pass.order_independent.end];
            let renderer_idx = prev_renderer.unwrap() as usize;
            self.render_commands(
                commands,
                renderer_idx,
                renderers,
                ctx.render_pipelines,
                ctx.resources,
                ctx.bindings,
                ctx.stats,
                config,
                ctx.gpu_profiler,
                &mut wgpu_pass,
            );
        }

        let mut prev_renderer = None;
        let mut start_idx = sub_pass.ordered.start;
        for (idx, command) in self.ordered[sub_pass.ordered.clone()].iter().enumerate() {
            let scissor_changed = command.scissor != current_scissor;
            if (Some(command.renderer) != prev_renderer || scissor_changed) && prev_renderer.is_some() {
                let commands = &self.ordered[start_idx..idx];
                let renderer_idx = prev_renderer.unwrap() as usize;
                self.render_commands(
                    commands,
                    renderer_idx,
                    renderers,
                    ctx.render_pipelines,
                    ctx.resources,
                    ctx.bindings,
                    ctx.stats,
                    config,
                    ctx.gpu_profiler,
                    &mut wgpu_pass,
                );
                start_idx = idx;
            }

            if scissor_changed {
                set_scissor(&command.scissor, &mut wgpu_pass);
            }

            current_scissor = command.scissor;
            prev_renderer = Some(command.renderer);
        }

        if start_idx != sub_pass.ordered.end {
            let commands = &self.ordered[start_idx..sub_pass.ordered.end];
            let renderer_idx = prev_renderer.unwrap() as usize;
            self.render_commands(
                commands,
                renderer_idx,
                renderers,
                ctx.render_pipelines,
                ctx.resources,
                ctx.bindings,
                ctx.stats,
                config,
                ctx.gpu_profiler,
                &mut wgpu_pass,
            );
        }
    }

    pub fn render_commands<'pass, 'resources: 'pass>(
        &self,
        commands: &[RenderCommandId],
        renderer_idx: usize,
        renderers: &[&mut dyn Renderer],
        render_pipelines: &'resources RenderPipelines,
        resources: &'resources crate::resources::GpuResources,
        bindings: &'resources dyn BindingResolver,
        stats: &mut [RendererStats],
        config: &RenderPassConfig,
        gpu_profiler: &'resources GpuProfiler,
        wgpu_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        let start_time = std::time::Instant::now();
        let stats = &mut stats[renderer_idx];
        renderers[renderer_idx].render(
            commands,
            config,
            RenderContext {
                render_pipelines,
                bindings,
                resources,
                stats,
                gpu_profiler,
            },
            wgpu_pass
        );

        let time = crate::instance::ms(std::time::Instant::now() - start_time);
        stats.render_time += time;
    }
}
pub struct PassRenderContext<'l> {
    pub encoder: &'l mut wgpu::CommandEncoder,
    pub resources: &'l GpuResources,
    pub bindings: &'l dyn BindingResolver,
    pub render_pipelines: &'l RenderPipelines,
    pub stats: &'l mut [RendererStats],
    pub gpu_profiler: &'l mut wgpu_profiler::GpuProfiler,
}
