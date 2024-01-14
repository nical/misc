// TODO: this module lumps together a bunch of structural concepts some of which
// are poorly named and/or belong elsewhere.

use crate::batching::{BatchId, Batcher, SurfaceIndex};
use crate::gpu::shader::RenderPipelines;
use crate::path::FillRule;
use crate::pattern::{BindingsId, BuiltPattern};
use crate::resources::{AsAny, CommonGpuResources, GpuResources, ResourcesHandle};
use crate::shape::FilledPath;
use crate::stroke::StrokeOptions;
use crate::transform::Transforms;
use crate::units::SurfaceIntSize;
use crate::{BindingResolver, Color};
use std::ops::Range;

pub type ZIndex = u32;

pub struct ZIndices {
    next: ZIndex,
}

impl ZIndices {
    pub fn new() -> Self {
        ZIndices { next: 0 }
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
        self.next = 0;
    }
}

impl Default for ZIndices {
    fn default() -> Self {
        Self::new()
    }
}

pub type RendererId = u16;
pub type RenderPassId = u32;

pub struct RenderPassesRequirements {
    pub msaa: bool,
    pub depth_stencil: bool,
    pub msaa_depth_stencil: bool,
    // Temporary color target used in place of the main target if we need
    // to read from it but can't.
    pub temporary: bool,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct ContextSurface {
    states: Vec<SurfacePassConfig>,
    size: SurfaceIntSize,
    state: SurfacePassConfig,
    clear: Option<Color>,
    /// If true, the main target cannot be sampled (for example a swapchain's target).
    write_only_target: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum StencilMode {
    EvenOdd,
    NonZero,
    Ignore,
    None,
}

impl From<FillRule> for StencilMode {
    fn from(fill_rule: FillRule) -> Self {
        match fill_rule {
            FillRule::EvenOdd => StencilMode::EvenOdd,
            FillRule::NonZero => StencilMode::NonZero,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DepthMode {
    Enabled,
    Ignore,
    None,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SurfaceKind {
    Color,
    Alpha,
    // TODO: HDRColor, color spaces?
}

// The surface parameters for draw calls.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SurfaceDrawConfig {
    pub msaa: bool,
    pub depth: DepthMode,
    pub stencil: StencilMode,
    pub kind: SurfaceKind,
}

impl SurfaceDrawConfig {
    pub(crate) fn hash(&self) -> u8 {
        return if self.msaa { 1 } else { 0 }
            + (match self.depth {
                DepthMode::Enabled => 1,
                DepthMode::Ignore => 2,
                DepthMode::None => 0,
            } << 1)
            + (match self.stencil {
                StencilMode::EvenOdd => 1,
                StencilMode::NonZero => 2,
                StencilMode::Ignore => 3,
                StencilMode::None => 0,
            } << 3)
            + (match self.kind {
                SurfaceKind::Color => 0,
                SurfaceKind::Alpha => 1,
            } << 6);
    }

    pub(crate) fn from_hash(val: u8) -> Self {
        SurfaceDrawConfig {
            msaa: val & 1 != 0,
            depth: match (val >> 1) & 3 {
                1 => DepthMode::Enabled,
                2 => DepthMode::Ignore,
                _ => DepthMode::None,
            },
            stencil: match (val >> 3) & 3 {
                1 => StencilMode::EvenOdd,
                2 => StencilMode::NonZero,
                3 => StencilMode::Ignore,
                _ => StencilMode::None,
            },
            kind: match (val >> 6) & 1 {
                0 => SurfaceKind::Color,
                _ => SurfaceKind::Alpha,
            }
        }
    }

    pub fn color() -> Self {
        SurfaceDrawConfig {
            msaa: false,
            depth: DepthMode::None,
            stencil: StencilMode::None,
            kind: SurfaceKind::Color,
        }
    }

    pub fn alpha() -> Self {
        SurfaceDrawConfig {
            msaa: false,
            depth: DepthMode::None,
            stencil: StencilMode::None,
            kind: SurfaceKind::Color,
        }
    }

    pub fn with_msaa(mut self, msaa: bool) -> Self {
        self.msaa = msaa;
        self
    }

    pub fn with_stencil(mut self, stencil: StencilMode) -> Self {
        self.stencil = stencil;
        self
    }

    pub fn with_depth(mut self, depth: DepthMode) -> Self {
        self.depth = depth;
        self
    }
}

impl Default for SurfaceDrawConfig {
    fn default() -> Self {
        SurfaceDrawConfig::color()
    }
}

#[test]
fn surface_draw_config() {
    for msaa in [true, false] {
        for depth in [DepthMode::Enabled, DepthMode::Ignore, DepthMode::None] {
            for stencil in [StencilMode::EvenOdd, StencilMode::NonZero, StencilMode::Ignore, StencilMode::None] {
                for kind in [SurfaceKind::Color, SurfaceKind::Alpha] {
                    let surface = SurfaceDrawConfig { msaa, depth, stencil, kind};
                    assert_eq!(SurfaceDrawConfig::from_hash(surface.hash()), surface);
                }                
            }
        }
    }
}

/// The surface parameters for render passes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SurfacePassConfig {
    pub depth: bool,
    pub msaa: bool,
    pub stencil: bool,
    pub kind: SurfaceKind,
}

impl Default for SurfacePassConfig {
    fn default() -> Self {
        SurfacePassConfig {
            depth: false,
            msaa: false,
            stencil: false,
            kind: SurfaceKind::Color,
        }
    }
}

impl SurfacePassConfig {
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
            kind: self.kind,
        }
    }

    pub fn color() -> Self {
        SurfacePassConfig {
            msaa: false,
            depth: false,
            stencil: false,
            kind: SurfaceKind::Color,
        }
    }

    pub fn alpha() -> Self {
        SurfacePassConfig {
            msaa: false,
            depth: false,
            stencil: false,
            kind: SurfaceKind::Color,
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

impl ContextSurface {
    #[inline]
    fn new(size: SurfaceIntSize, state: SurfacePassConfig) -> Self {
        ContextSurface {
            states: vec![state],
            size,
            state,
            clear: Some(Color::BLACK),
            write_only_target: true,
        }
    }

    #[inline]
    pub fn size(&self) -> SurfaceIntSize {
        self.size
    }

    #[inline]
    pub fn msaa(&self) -> bool {
        self.state.msaa
    }

    #[inline]
    pub fn opaque_pass(&self) -> bool {
        self.state.depth
    }

    #[inline]
    pub fn get_config(&self, surface: SurfaceIndex) -> SurfacePassConfig {
        self.states[surface as usize]
    }

    pub fn current_config(&self) -> SurfacePassConfig {
        self.state
    }
}

// TODO: should this go in RenderContext?
/// Information about the current render pass.
pub struct RenderPassState {
    pub output_type: SurfaceKind,
    pub surface: SurfacePassConfig,
}

impl RenderPassState {
    #[inline]
    pub fn surface_config(&self, use_depth: bool, stencil: Option<FillRule>) -> SurfaceDrawConfig {
        self.surface.draw_config(use_depth, stencil)
    }
}

pub struct CanvasParams {
    pub tolerance: f32,
}

impl Default for CanvasParams {
    fn default() -> Self {
        CanvasParams { tolerance: 0.25 }
    }
}

pub struct Context {
    pub z_indices: ZIndices,
    pub surface: ContextSurface,
    pub params: CanvasParams,
    pub batcher: Batcher,
}

impl Context {
    pub fn new(params: CanvasParams) -> Self {
        Context {
            z_indices: ZIndices::default(),
            surface: ContextSurface::default(),
            params,
            batcher: Batcher::new(),
        }
    }

    pub fn begin_frame(&mut self, size: SurfaceIntSize, surface: SurfacePassConfig) {
        self.z_indices.clear();
        self.surface = ContextSurface::new(size, surface);
        self.batcher.begin();
    }

    pub fn prepare(&mut self) {
        self.batcher.finish();
        self.surface.states.push(self.surface.state);
    }

    pub fn render(
        &self,
        renderers: &[&dyn Renderer],
        resources: &GpuResources,
        bindings: &dyn BindingResolver,
        render_pipelines: &RenderPipelines,
        _device: &wgpu::Device,
        common_resources: ResourcesHandle<CommonGpuResources>,
        target: &SurfaceResources,
        resolve_to_main: bool,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        // TODO: Pre-passes


        // Sub-passes

        let surface_cfg = self.surface.current_config();

        // Whether to blit the previous non-msaa pass into this one (msaa).
        let msaa_blit = false; // TODO
        // Whether to resolve this msaa pass pass to a non-msaa target.
        let msaa_resolve = surface_cfg.msaa && resolve_to_main;
        let clear = !msaa_resolve && self.surface.clear.is_some();

        let (view, label) = if surface_cfg.msaa {
            (target.msaa_color.unwrap(), "MSAA color target")
        } else  {
            (target.main, "Color target")
        };

        let ops = wgpu::Operations {
            load: if clear {
                wgpu::LoadOp::Clear(
                    self.surface
                        .clear
                        .map(Color::to_wgpu)
                        .unwrap_or(wgpu::Color::BLACK),
                )
            } else {
                wgpu::LoadOp::Load
            },
            store: if msaa_resolve {
                wgpu::StoreOp::Discard
            } else {
                wgpu::StoreOp::Store
            },
        };

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: if msaa_resolve {
                    Some(&target.main)
                } else {
                    None
                },
                ops,
            })],
            depth_stencil_attachment: if surface_cfg.depth_or_stencil() {
                Some(wgpu::RenderPassDepthStencilAttachment {
                    view: if surface_cfg.msaa {
                        target.msaa_depth
                    } else {
                        target.depth
                    }
                    .unwrap(),
                    depth_ops: if surface_cfg.depth {
                        Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0.0),
                            store: wgpu::StoreOp::Discard,
                        })
                    } else {
                        None
                    },
                    stencil_ops: if surface_cfg.stencil {
                        Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(128),
                            store: wgpu::StoreOp::Discard,
                        })
                    } else {
                        None
                    },
                })
            } else {
                None
            },
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        if msaa_blit {
            let common = &resources[common_resources];
            render_pass.set_bind_group(0, target.temporary_src_bind_group.unwrap(), &[]);
            render_pass.set_index_buffer(common.quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.set_pipeline(if surface_cfg.depth || surface_cfg.stencil {
                &common.msaa_blit_with_depth_stencil_pipeline
            } else {
                &common.msaa_blit_pipeline
            });
            render_pass.draw_indexed(0..6, 0, 0..1);
        }

        let pass_info = RenderPassState {
            output_type: SurfaceKind::Color,
            surface: surface_cfg,
        };

        // Traverse batches, grouping consecutive items with the same renderer.
        let batches = self.batcher.batches();
        let mut start = 0;
        let mut end = 0;
        let mut renderer = batches[0].renderer;
        while start < batches.len() {
            let done = end >= batches.len();
            if !done && renderer == batches[end].renderer {
                end += 1;
                continue;
            }

            renderers[renderer as usize].render(
                &batches[start..end],
                &pass_info,
                RenderContext {
                    render_pipelines,
                    bindings,
                    resources,
                },
                &mut render_pass
            );

            if done {
                break;
            }
            start = end;
            renderer = batches[start].renderer;
        }
    }
}

/// Parameters for the canvas renderers
pub struct RenderContext<'l> {
    pub render_pipelines: &'l RenderPipelines,
    pub resources: &'l GpuResources,
    pub bindings: &'l dyn BindingResolver,
}

pub trait Renderer: AsAny {
    fn render_pre_pass(
        &self,
        _index: u32,
        _ctx: RenderContext,
        _encoder: &mut wgpu::CommandEncoder,
    ) {
    }

    fn render<'pass, 'resources: 'pass>(
        &self,
        _batches: &[BatchId],
        _pass_info: &RenderPassState,
        _ctx: RenderContext<'resources>,
        _render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
    }
}

pub trait FillPath {
    fn fill_path(&mut self, ctx: &mut Context, transforms: &Transforms, path: FilledPath, pattern: BuiltPattern);
}

pub trait StrokePath {
    fn stroke_path(&mut self, ctx: &mut Context, path: crate::path::Path, options: StrokeOptions, pattern: BuiltPattern);
}


/// A bock of commands from a specific renderer within a render pass.
#[derive(Copy, Clone, Debug)]
pub struct SubPass {
    pub renderer_id: RendererId,
    // This index is provided by the render in `add_render_pass` and passed back
    // to it in `render`. It can be anything, it is usually a batch index.
    pub internal_index: u32,
    pub require_pre_pass: bool,
    // Used when generating the render passes to decide when to split render
    // passes. It can be obtained from the batch id.
    pub surface: SurfaceIndex,
}

/// Work a renderer can schedule before a render pass.
#[derive(Debug)]
pub struct PrePass {
    pub renderer_id: RendererId,
    pub internal_index: u32,
}

pub struct SurfaceResources<'a> {
    pub main: &'a wgpu::TextureView,
    pub temporary_color: Option<&'a wgpu::TextureView>,
    pub temporary_src_bind_group: Option<&'a wgpu::BindGroup>,
    pub depth: Option<&'a wgpu::TextureView>,
    pub msaa_color: Option<&'a wgpu::TextureView>,
    pub msaa_depth: Option<&'a wgpu::TextureView>,
}

/// A helper struct to resolve bind groups and avoid redundant bindings.
pub struct DrawHelper {
    current_bindings: [BindingsId; 4],
}

impl DrawHelper {
    pub fn new() -> Self {
        DrawHelper {
            current_bindings: [
                BindingsId::NONE,
                BindingsId::NONE,
                BindingsId::NONE,
                BindingsId::NONE,
            ],
        }
    }

    pub fn resolve_and_bind<'pass, 'resources: 'pass>(
        &mut self,
        group_index: u32,
        id: BindingsId,
        resolver: &'resources dyn BindingResolver,
        pass: &mut wgpu::RenderPass<'pass>,
    ) {
        let idx = group_index as usize;
        if id.is_some() && id != self.current_bindings[idx] {
            if let Some(bind_group) = resolver.resolve(id) {
                pass.set_bind_group(group_index, bind_group, &[]);
            }
            self.current_bindings[idx] = id;
        }
    }

    pub fn bind<'pass, 'resources: 'pass>(
        &mut self,
        group_index: u32,
        id: BindingsId,
        bind_group: &'resources wgpu::BindGroup,
        pass: &mut wgpu::RenderPass<'pass>,
    ) {
        let idx = group_index as usize;
        if self.current_bindings[idx] != id {
            pass.set_bind_group(group_index, bind_group, &[]);
            self.current_bindings[idx] = id
        }
    }

    pub fn reset_binding(&mut self, group_index: u32) {
        self.current_bindings[group_index as usize] = BindingsId::NONE;
    }
}
