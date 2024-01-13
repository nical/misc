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
use crate::{u32_range, usize_range, BindingResolver, Color};
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

#[derive(Debug)]
struct RenderPass {
    pre_passes: Range<u32>,
    sub_passes: Range<u32>,
    surface: SurfacePassConfig,
    msaa_resolve: bool,
    msaa_blit: bool,
    temporary: bool,
}

#[derive(Debug)]
struct RenderPassSlice<'a> {
    pre_passes: &'a [PrePass],
    sub_passes: &'a [SubPass],
    surface: SurfacePassConfig,
    msaa_resolve: bool,
    msaa_blit: bool,
    temporary: bool,
}

pub struct RenderPassesRequirements {
    pub msaa: bool,
    pub depth_stencil: bool,
    pub msaa_depth_stencil: bool,
    // Temporary color target used in place of the main target if we need
    // to read from it but can't.
    pub temporary: bool,
}

impl RenderPassesRequirements {
    fn add_pass(&mut self, pass: SurfacePassConfig) {
        self.msaa |= pass.msaa;
        self.msaa_depth_stencil |= pass.msaa && (pass.depth || pass.stencil);
        self.depth_stencil |= !pass.msaa && (pass.depth || pass.stencil);
    }
}

#[derive(Default)]
pub struct RenderPasses {
    sub_passes: Vec<SubPass>,
    pre_passes: Vec<PrePass>,
    passes: Vec<RenderPass>,
}

impl RenderPasses {
    pub fn push(&mut self, pass: SubPass) {
        self.sub_passes.push(pass);
    }

    pub fn clear(&mut self) {
        self.sub_passes.clear();
        self.passes.clear();
        self.pre_passes.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }

    fn build(&mut self, surface: &ContextSurface) -> RenderPassesRequirements {
        let mut requirements = RenderPassesRequirements {
            msaa: false,
            depth_stencil: false,
            msaa_depth_stencil: false,
            temporary: false,
        };

        // Assume we can't read from the main target and therefore not blit it
        // into an msaa arget.
        let disable_msaa_blit_from_main_target = surface.write_only_target;

        if self.sub_passes.is_empty() {
            return requirements;
        }

        // A bit per system specifying whether they have been used in the current
        // render pass yet.
        let mut renderer_bits: u64 = 0;

        let mut start = 0;
        let mut pre_start = 0;

        let mut current = self.sub_passes.first().unwrap().surface;
        let mut msaa_blit = false;

        for (idx, pass) in self.sub_passes.iter().enumerate() {
            let req_bit: u64 = 1 << pass.renderer_id;
            let flush_pass =
                pass.surface != current || (pass.require_pre_pass && renderer_bits & req_bit != 0);
            //println!(" - sub pass msaa:{}, changed:{msaa_changed}", pass.use_msaa);
            if flush_pass {
                // When transitioning from msaa to non-msaa targets, resolve the msaa
                // target into the non msaa one.
                let current_surface = surface.states[current as usize];
                let pass_surface = surface.states[pass.surface as usize];
                let msaa_resolve = current_surface.msaa && !pass_surface.msaa;
                //println!("   -> flush msaa resolve {msaa_resolve} blit {msaa_blit}");
                self.passes.push(RenderPass {
                    pre_passes: u32_range(pre_start..self.pre_passes.len()),
                    sub_passes: u32_range(start..idx),
                    surface: current_surface,
                    msaa_resolve,
                    msaa_blit,
                    temporary: false,
                });
                requirements.add_pass(current_surface);
                requirements.temporary |= disable_msaa_blit_from_main_target && msaa_blit;

                // When transitioning from non-msaa to msaa, blit the non-msaa target
                // into the msaa one.
                msaa_blit = !current_surface.msaa && pass_surface.msaa;
                if msaa_blit && disable_msaa_blit_from_main_target {
                    for pass in self.passes.iter_mut().rev() {
                        if pass.surface.msaa {
                            break;
                        }
                        pass.temporary = true;
                    }
                }

                start = idx;
                pre_start = self.pre_passes.len();
                current = pass.surface;
                renderer_bits = 0;
            }

            if pass.require_pre_pass {
                self.pre_passes.push(PrePass {
                    renderer_id: pass.renderer_id,
                    internal_index: pass.internal_index,
                });

                renderer_bits |= req_bit;
            }
        }

        if start < self.sub_passes.len() {
            let surface = surface.states[current as usize];
            self.passes.push(RenderPass {
                pre_passes: u32_range(pre_start..self.pre_passes.len()),
                sub_passes: u32_range(start..self.sub_passes.len()),
                surface,
                msaa_resolve: surface.msaa,
                msaa_blit,
                temporary: false,
            });

            requirements.add_pass(surface);
            requirements.temporary |= msaa_blit;
        }

        requirements
    }

    fn iter(&self) -> impl Iterator<Item = RenderPassSlice> {
        self.passes.iter().map(|pass| RenderPassSlice {
            pre_passes: &self.pre_passes[usize_range(pass.pre_passes.clone())],
            sub_passes: &self.sub_passes[usize_range(pass.sub_passes.clone())],
            surface: pass.surface,
            msaa_resolve: pass.msaa_resolve,
            msaa_blit: pass.msaa_blit,
            temporary: pass.temporary,
        })
    }
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
    // TODO: maybe split off the push/pop transforms builder thing so that this remains more compatible
    // with a retained scene model as well.
    pub transforms: Transforms,
    pub z_indices: ZIndices,
    pub surface: ContextSurface,
    render_passes: RenderPasses,
    pub params: CanvasParams,
    pub batcher: Batcher,
}

impl Context {
    pub fn new(params: CanvasParams) -> Self {
        Context {
            transforms: Transforms::default(),
            z_indices: ZIndices::default(),
            surface: ContextSurface::default(),
            render_passes: RenderPasses::default(),
            params,
            batcher: Batcher::new(),
        }
    }

    pub fn begin_frame(&mut self, size: SurfaceIntSize, surface: SurfacePassConfig) {
        self.transforms.clear();
        self.z_indices.clear();
        self.render_passes.clear();
        self.surface = ContextSurface::new(size, surface);
        self.batcher.begin();
    }

    pub fn prepare(&mut self) {
        self.batcher.finish();
        self.surface.states.push(self.surface.state);
    }

    pub fn reconfigure_surface(&mut self, state: SurfacePassConfig) {
        if self.surface.state == state {
            return;
        }
        self.batcher
            .set_render_pass(self.surface.states.len() as u16);
        self.surface.states.push(state);
        self.surface.state = state;
    }

    // TODO: Maybe remove this step and pass the batch id directly duirng rendering.
    // This removes the ability to split render passes in the middle of a batch but that
    // should probably be a higher-level thing.
    // Would need another way for the renderers to request for pre-passes.
    pub fn build_render_passes(
        &mut self,
        renderers: &mut [&mut dyn Renderer],
    ) -> RenderPassesRequirements {
        for batch in self.batcher.batches() {
            renderers[batch.renderer as usize].add_render_passes(*batch, &mut self.render_passes)
        }

        self.render_passes.build(&self.surface)
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
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let mut need_clear = self.surface.clear.is_some();
        #[cfg(debug_assertions)]
        if self.surface.clear.is_none()
            && self.surface.write_only_target
            && !self.render_passes.is_empty()
        {
            let first = self.render_passes.iter().next().unwrap();
            if first.surface.msaa || first.temporary {
                println!("Can't load content if the first pass is a temporary or msaa target");
            }
        }

        for pass in self.render_passes.iter() {
            for pre_pass in pass.pre_passes {
                renderers[pre_pass.renderer_id as usize].render_pre_pass(
                    pre_pass.internal_index,
                    RenderContext {
                        render_pipelines,
                        resources,
                        bindings,
                    },
                    encoder,
                );
            }

            let (view, label) = if pass.surface.msaa {
                (target.msaa_color.unwrap(), "MSAA color target")
            } else if pass.temporary {
                (target.temporary_color.unwrap(), "Temporary color target")
            } else {
                (target.main, "Color target")
            };

            //println!("{}: {:#?}", label, pass);

            // If the first thing we do is a full-target blit, no nead to load the
            // contents of the target.
            let mut clear = pass.msaa_blit;
            // After resolving the msaa target, we don't want to clear the contents of the
            // main target
            if pass.msaa_resolve {
                need_clear = false;
            }

            if need_clear {
                need_clear = false;
                clear = true;
            }

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
                store: if pass.msaa_resolve {
                    wgpu::StoreOp::Discard
                } else {
                    wgpu::StoreOp::Store
                },
            };

            //println!("{label}: {pass:#?}");

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: if pass.msaa_resolve {
                        Some(&target.main)
                    } else {
                        None
                    },
                    ops,
                })],
                depth_stencil_attachment: if pass.surface.depth_or_stencil() {
                    Some(wgpu::RenderPassDepthStencilAttachment {
                        view: if pass.surface.msaa {
                            target.msaa_depth
                        } else {
                            target.depth
                        }
                        .unwrap(),
                        depth_ops: if pass.surface.depth {
                            Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(0.0),
                                store: wgpu::StoreOp::Discard,
                            })
                        } else {
                            None
                        },
                        stencil_ops: if pass.surface.stencil {
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

            if pass.msaa_blit {
                let common = &resources[common_resources];
                render_pass.set_bind_group(0, target.temporary_src_bind_group.unwrap(), &[]);
                render_pass.set_index_buffer(common.quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.set_pipeline(if pass.surface.depth || pass.surface.stencil {
                    &common.msaa_blit_with_depth_stencil_pipeline
                } else {
                    &common.msaa_blit_pipeline
                });
                render_pass.draw_indexed(0..6, 0, 0..1);
            }

            let pass_info = RenderPassState {
                output_type: SurfaceKind::Color,
                surface: pass.surface,
            };

            let mut start = 0;
            let mut renderer = pass.sub_passes[0].renderer_id;
            for (idx, sub_pass) in pass.sub_passes.iter().enumerate() {
                if renderer != sub_pass.renderer_id && idx > start {
                    renderers[renderer as usize].render(
                        &pass.sub_passes[start..idx],
                        &pass_info,
                        RenderContext {
                            render_pipelines,
                            bindings,
                            resources,
                        },
                        &mut render_pass,
                    );
                    start = idx;
                    renderer = sub_pass.renderer_id;
                }
            }
            if pass.sub_passes.len() > start {
                renderers[renderer as usize].render(
                    &pass.sub_passes[start..pass.sub_passes.len()],
                    &pass_info,
                    RenderContext {
                        render_pipelines,
                        bindings,
                        resources,
                    },
                    &mut render_pass,
                );
            }
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
    fn add_render_passes(&mut self, batch: BatchId, render_passes: &mut RenderPasses) {
        render_passes.push(SubPass {
            renderer_id: batch.renderer,
            internal_index: batch.index,
            require_pre_pass: false,
            surface: batch.surface,
        });
    }

    fn render_pre_pass(
        &self,
        _index: u32,
        _ctx: RenderContext,
        _encoder: &mut wgpu::CommandEncoder,
    ) {
    }

    fn render<'pass, 'resources: 'pass>(
        &self,
        _sub_passes: &[SubPass],
        _pass_info: &RenderPassState,
        _ctx: RenderContext<'resources>,
        _render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
    }
}

pub trait FillPath {
    fn fill_path(&mut self, ctx: &mut Context, path: FilledPath, pattern: BuiltPattern);
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
