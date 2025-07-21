use crate::batching::{BatchId, Batcher};
use crate::graph::{Command, CommandContext, TaskId};
use crate::render_task::{RenderTaskHandle, RenderTaskInfo};
use crate::shading::{DepthMode, RenderPipelines, StencilMode, SurfaceDrawConfig, SurfaceKind};
use crate::instance::{ms, RenderStats};
use crate::path::FillRule;
use crate::units::SurfaceIntSize;
use crate::{BindingResolver, BindingsId, PrepareContext, RenderContext, Renderer};
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

pub struct GraphRenderCommand {
    label: Option<&'static str>,
    built_pass: BuiltRenderPass,
    // [(non-msaa, msaa, flags); 3]
    info: RenderPassInfo,
    //color_attachments: [(Option<BindingsId>, Option<BindingsId>, AttathchmentFlags); 3],
    //depth_stencil_attachment: Option<BindingsId>,
    render_commands: RenderCommands,
}

impl std::fmt::Debug for GraphRenderCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(label) = self.label {
            write!(f, "RenderCommand({})", label)
        } else {
            write!(f, "RenderCommand")
        }
    }
}

impl Command for GraphRenderCommand {
    fn prepare(
        &mut self,
        ctx: &mut PrepareContext,
        renderers: &mut[&mut dyn Renderer],
    ) {
        let batches = self.built_pass.batches();

        for (idx, renderer) in renderers.iter_mut().enumerate() {
            let renderer_prepare_start = Instant::now();
            renderer.prepare_pass(ctx, &self.built_pass);
            let stats = &mut ctx.stats.renderers[idx];
            stats.prepare_time += ms(Instant::now() - renderer_prepare_start);
        }

        for batch in batches.iter().rev() {
            let renderer_prepare_start = Instant::now();
            let renderer_idx = batch.renderer as usize;
            renderers[renderer_idx].prepare_batch_backward(ctx, &self.built_pass, *batch);
            let stats = &mut ctx.stats.renderers[renderer_idx];
            stats.prepare_time += ms(Instant::now() - renderer_prepare_start);
        }

        for batch in batches {
            let renderer_prepare_start = Instant::now();
            let renderer_idx = batch.renderer as usize;
            renderers[renderer_idx].prepare_batch_forward(ctx, &self.built_pass, *batch, &mut self.render_commands);
            let stats = &mut ctx.stats.renderers[renderer_idx];
            stats.prepare_time += ms(Instant::now() - renderer_prepare_start);
        }

        self.render_commands.end_render_pass();
    }

    fn execute(&self, ctx: &mut CommandContext) {
        self.render_commands.schedule(ctx, &self.info);
    }
}

pub struct RenderPasses {
    built_render_passes: Vec<Option<BuiltRenderPass>>,
    next_id: u64,
}

impl RenderPasses {
    pub fn new() -> Self {
        RenderPasses {
            built_render_passes: Vec::with_capacity(128),
            next_id: 0,
        }
    }

    pub(crate) fn next_render_pass_id(&mut self) -> TaskId {
        let id = TaskId(self.next_id);
        self.next_id += 1;

        id
    }

    pub(crate) fn add_built_render_pass(&mut self, id: TaskId, pass: BuiltRenderPass) {
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

    pub fn create_command(
        &mut self,
        label: Option<&'static str>,
        built_pass_idx: u32,
        color_attachments: &[(Option<BindingsId>, Option<BindingsId>, AttathchmentFlags); 3],
        depth_stencil_attachment: Option<BindingsId>,
    ) -> Box<dyn Command> {
        let built_pass = self.built_render_passes[built_pass_idx as usize].take().unwrap();
        Box::new(GraphRenderCommand {
            label,
            info: RenderPassInfo {
                label,
                config: built_pass.config(),
                color_attachments: *color_attachments,
                depth_stencil_attachment,
            },
            built_pass,
            render_commands: RenderCommands::new(),
        })
    }
}


#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct RenderCommandId {
    pub renderer: RendererId,
    pub request_pre_pass: bool,
    pub index: u32,
}

impl From<BatchId> for RenderCommandId {
    fn from(batch: BatchId) -> Self {
        RenderCommandId {
            renderer: batch.renderer,
            request_pre_pass: false,
            index: batch.index,
        }
    }
}

struct SubPass {
    ordered: Range<usize>,
    order_independent: Range<usize>,
    pre_passes: Range<usize>,
}

pub struct RenderPassInfo {
    pub label: Option<&'static str>,
    pub config: RenderPassConfig,
    pub color_attachments: [(Option<BindingsId>, Option<BindingsId>, AttathchmentFlags); 3],
    pub depth_stencil_attachment: Option<BindingsId>,
}

pub struct RenderCommands {
    ordered: Vec<RenderCommandId>,
    order_independent: Vec<RenderCommandId>,
    pre_passes: Vec<RenderCommandId>,
    sub_passes: Vec<SubPass>,
    requested_pre_pass: Vec<bool>,
    current_pass_start: u16,
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
        }
    }

    pub fn push(&mut self, command: RenderCommandId) {
        if command.request_pre_pass {
            self.request_pre_pass(command);
        }
        self.ordered.push(command);
    }

    pub fn push_order_independent(&mut self, command: RenderCommandId) {
        if command.request_pre_pass {
            self.request_pre_pass(command);
        }
        self.order_independent.push(command)
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

        self.sub_passes.push(sub_pass);
    }

    pub fn schedule(
        &self,
        ctx: &mut CommandContext,
        info: &RenderPassInfo,
    ) {
        for sub_pass in &self.sub_passes {
            self.render_sub_pass(ctx, sub_pass, info);
        }
    }

    fn render_pre_pass(
        &self,
        _pre_pass: &RenderCommandId,
        _ctx: &mut CommandContext,
    ) {
        // TODO
    }

    fn render_sub_pass(
        &self,
        ctx: &mut CommandContext,
        sub_pass: &SubPass,
        info: &RenderPassInfo,
    ) {
        for pre_pass in &self.pre_passes[sub_pass.pre_passes.clone()] {
            self.render_pre_pass(pre_pass, ctx);
        }

        let mut wgpu_attachments = Vec::new();

        let msaa = info.config.msaa();
        for (non_msaa_attachment, msaa_attachment, flags) in &info.color_attachments {
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

        let depth_stencil_attachment = info.depth_stencil_attachment.map(|id| {
            let view = ctx.bindings.resolve_attachment(id).unwrap();
            wgpu::RenderPassDepthStencilAttachment {
                view,
                depth_ops: if info.config.depth {
                    Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Discard,
                    })
                } else {
                    None
                },
                stencil_ops: if info.config.stencil {
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
            label: info.label,
            color_attachments: &wgpu_attachments,
            depth_stencil_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
        };

        let mut wgpu_pass = ctx.encoder.begin_render_pass(&pass_descriptor);

        let base_bind_group = ctx.resources.common.get_base_bindgroup();
        wgpu_pass.set_bind_group(0, base_bind_group, &[]);

        let mut prev_renderer = None;
        let mut end_idx = sub_pass.order_independent.end;
        for (idx, command) in self.order_independent[sub_pass.order_independent.clone()].iter().enumerate().rev() {
            if Some(command.renderer) != prev_renderer && prev_renderer.is_some() {
                let commands = &self.order_independent[idx..end_idx];
                let renderer_idx = prev_renderer.unwrap() as usize;
                self.render_commands(
                    commands,
                    renderer_idx,
                    ctx.renderers,
                    ctx.render_pipelines,
                    ctx.resources,
                    ctx.bindings,
                    ctx.stats,
                    &info.config,
                    &mut wgpu_pass,
                );
                end_idx = idx;
            }

            prev_renderer = Some(command.renderer);
        }

        if end_idx != sub_pass.order_independent.start {
            let commands = &self.order_independent[sub_pass.order_independent.start..end_idx];
            let renderer_idx = prev_renderer.unwrap() as usize;
            self.render_commands(
                commands,
                renderer_idx,
                ctx.renderers,
                ctx.render_pipelines,
                ctx.resources,
                ctx.bindings,
                ctx.stats,
                &info.config,
                &mut wgpu_pass,
            );
        }

        let mut prev_renderer = None;
        let mut start_idx = sub_pass.ordered.start;
        for (idx, command) in self.ordered[sub_pass.ordered.clone()].iter().enumerate() {
            if Some(command.renderer) != prev_renderer && prev_renderer.is_some() {
                let commands = &self.ordered[start_idx..idx];
                let renderer_idx = prev_renderer.unwrap() as usize;
                self.render_commands(
                    commands,
                    renderer_idx,
                    ctx.renderers,
                    ctx.render_pipelines,
                    ctx.resources,
                    ctx.bindings,
                    ctx.stats,
                    &info.config,
                    &mut wgpu_pass,
                );
                start_idx = idx;
            }

            prev_renderer = Some(command.renderer);
        }

        if start_idx != sub_pass.ordered.end {
            let commands = &self.ordered[start_idx..sub_pass.ordered.end];
            let renderer_idx = prev_renderer.unwrap() as usize;
            self.render_commands(
                commands,
                renderer_idx,
                ctx.renderers,
                ctx.render_pipelines,
                ctx.resources,
                ctx.bindings,
                ctx.stats,
                &info.config,
                &mut wgpu_pass,
            );
        }
    }

    pub fn render_commands<'pass, 'resources: 'pass>(
        &self,
        commands: &[RenderCommandId],
        renderer_idx: usize,
        renderers: &[&dyn Renderer],
        render_pipelines: &'resources RenderPipelines,
        resources: &'resources crate::resources::GpuResources,
        bindings: &'resources dyn BindingResolver,
        stats: &mut RenderStats,
        config: &RenderPassConfig,
        wgpu_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        let start_time = std::time::Instant::now();
        let stats = &mut stats.renderers[renderer_idx];
        renderers[renderer_idx].render(
            commands,
            config,
            RenderContext {
                render_pipelines,
                bindings,
                resources,
                stats,
            },
            wgpu_pass
        );

        let time = crate::instance::ms(std::time::Instant::now() - start_time);
        stats.render_time += time;
    }
}
