use crate::batching::{Batcher, BatchId};
use crate::gpu::shader::RenderPipelines;
use crate::path::FillRule;
use crate::resources::GpuResources;
use crate::units::SurfaceIntSize;
use crate::{BindingsId, BindingResolver, Renderer, RenderContext};
use std::ops::Range;

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
pub type RenderPassId = u32;

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

#[derive(Debug, Default)]
pub struct SurfaceInfo {
    size: SurfaceIntSize,
    config: SurfacePassConfig,
}

impl SurfaceInfo {
    #[inline]
    fn new(size: SurfaceIntSize, surface_cfg: SurfacePassConfig) -> Self {
        SurfaceInfo {
            size,
            config: surface_cfg,
        }
    }

    #[inline]
    pub fn size(&self) -> SurfaceIntSize {
        self.size
    }

    #[inline]
    pub fn msaa(&self) -> bool {
        self.config.msaa
    }

    #[inline]
    pub fn depth(&self) -> bool {
        self.config.depth
    }

    pub fn config(&self) -> SurfacePassConfig {
        self.config
    }
}

// Paramater for the renderers's prepare methods.
// For now it matches the RenderPassBuilder it is created from
// but it may become a subset of it.
pub struct RenderPassContext<'l> {
    pub z_indices: &'l mut ZIndices,
    pub batcher: &'l mut Batcher,
    pub surface: SurfacePassConfig,
}

pub struct RenderPassBuilder {
    pub z_indices: ZIndices,
    pub surface: SurfaceInfo,
    pub batcher: Batcher,
}

impl RenderPassBuilder {
    pub fn new() -> Self {
        RenderPassBuilder {
            z_indices: ZIndices::default(),
            surface: SurfaceInfo::default(),
            batcher: Batcher::new(),
        }
    }

    pub fn ctx(&mut self) -> RenderPassContext {
        RenderPassContext {
            z_indices: &mut self.z_indices,
            batcher: &mut self.batcher,
            surface: self.surface.config(),
        }
    }

    pub fn begin(&mut self, size: SurfaceIntSize, surface: SurfacePassConfig) {
        self.z_indices.clear();
        self.surface = SurfaceInfo::new(size, surface);
        self.batcher.begin();
    }

    pub fn end(&mut self) -> BuiltRenderPass {
        let mut batches = Vec::new();
        self.batcher.finish(&mut batches);

        BuiltRenderPass {
            batches,
            config: self.surface.config,
            size: self.surface.size(),
        }
    }
}

pub struct BuiltRenderPass {
    batches: Vec<BatchId>,
    config: SurfacePassConfig,
    size: SurfaceIntSize,
}

impl BuiltRenderPass {
    pub fn batches(&self) -> &[BatchId] {
        &self.batches
    }

    pub fn surface(&self) -> SurfacePassConfig {
        self.config
    }

    pub fn surface_size(&self) -> SurfaceIntSize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }

    pub fn encode<'pass, 'resources: 'pass>(
        &self,
        pass_index: u16,
        renderers: &[&'resources dyn Renderer],
        resources: &'resources GpuResources,
        bindings: &'resources dyn BindingResolver,
        render_pipelines: &'resources RenderPipelines,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        if self.batches.is_empty() {
            return;
        }

        //if let Some(bind_group) = blit_src {
        //    let common = &resources[common_resources];
        //    render_pass.set_bind_group(0, bind_group, &[]);
        //    render_pass.set_index_buffer(common.quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
        //    render_pass.set_pipeline(if surface_cfg.depth || surface_cfg.stencil {
        //        &common.msaa_blit_with_depth_stencil_pipeline
        //    } else {
        //        &common.msaa_blit_pipeline
        //    });
        //    render_pass.draw_indexed(0..6, 0, 0..1);
        //}

        let base_bind_group = resources.graph.get_base_bindgroup(pass_index);
        render_pass.set_bind_group(0, base_bind_group, &[]);

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

            renderers[renderer as usize].render(
                &self.batches[start..end],
                &self.config,
                RenderContext {
                    render_pipelines,
                    bindings,
                    resources,
                },
                render_pass
            );

            if done {
                break;
            }
            start = end;
            renderer = self.batches[start].renderer;
        }
    }
}

/// Work a renderer can schedule before a render pass.
#[derive(Debug)]
pub struct PrePass {
    pub renderer_id: RendererId,
    pub internal_index: u32,
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
            if let Some(bind_group) = resolver.resolve_input(id) {
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
