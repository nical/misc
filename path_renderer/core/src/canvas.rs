// TODO: this module lumps together a bunch of structural concepts some of which
// are poorly named and/or belong elsewhere.

use std::sync::Arc;
use lyon::geom::euclid::vec2;
use lyon::math::{Point};
use lyon::path::{Path, FillRule};
use lyon::geom::euclid::default::{Transform2D, Box2D, Size2D};
use crate::gpu::shader::{OutputType, SurfaceConfig, DepthMode, StencilMode};
use crate::resources::{GpuResources, CommonGpuResources, ResourcesHandle, AsAny};
use crate::{Color, u32_range, usize_range, BindingResolver};
use crate::pattern::{BuiltPattern, BindingsId};
use std::{ops::Range};
use crate::gpu::{Shaders};

pub type TransformId = u32;

struct Transform {
    transform: Transform2D<f32>,
    parent: Option<TransformId>,
}

pub struct Transforms {
    current_transform: TransformId,
    transforms: Vec<Transform>,
}

impl Transforms {
    pub fn new() -> Self {
        Transforms {
            current_transform: 0,
            transforms: vec![
                Transform {
                    transform: Transform2D::identity(),
                    parent: None,
                },
            ],
        }
    }

    pub fn push(&mut self, transform: &Transform2D<f32>) {
        let id = self.transforms.len() as TransformId;
        if self.current_transform == 0 {
            self.transforms.push(Transform {
                transform: *transform,
                parent: Some(self.current_transform),
            });
        } else {
            let transform = self.transforms[self.current_transform as usize].transform.then(transform);
            self.transforms.push(Transform {
                transform,
                parent: Some(self.current_transform),
            });
        }

        self.current_transform = id;
    }

    pub fn pop(&mut self) {
        assert!(self.current_transform != 0);
        self.current_transform = self.transforms[self.current_transform as usize].parent.unwrap_or(0);
    }

    pub fn current(&self) -> TransformId {
        self.current_transform
    }

    pub fn get(&self, id: TransformId) -> &Transform2D<f32> {
        &self.transforms[id as usize].transform
    }

    pub fn get_current(&self) -> &Transform2D<f32> {
        self.get(self.current())
    }

    pub fn clear(&mut self) {
        self.current_transform = 0;
        self.transforms.shrink_to(1);
    }
}

impl Default for Transforms {
    fn default() -> Self {
        Self::new()
    }
}

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

        first .. self.next
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

pub type RendererId = u32;
pub type RendererCommandIndex = u32;

#[derive(Clone, Debug)]
pub struct CommandRange {
    pub commands: Range<RendererCommandIndex>,
    pub surface: SurfaceState,
}

pub struct Commands {
    commands: Vec<(RendererId, CommandRange)>,
    current: Option <(RendererId, CommandRange)>,
    surface: SurfaceState,
}

impl Commands {
    pub fn new(surface: SurfaceState) -> Self {
        Commands { commands: Vec::new(), current: None, surface }
    }

    pub fn push(&mut self, renderer: RendererId, internal_index: RendererCommandIndex) {
        if let Some((sys, range)) = &mut self.current {
            if *sys == renderer {
                range.commands.end += 1;
                return
            }

            self.commands.push((*sys, range.clone()));
        }

        self.current = Some((
            renderer,
            CommandRange {
                commands: internal_index..(internal_index + 1),
                surface: self.surface,
            }
        ));
    }

    fn flush(&mut self) {
        if let Some(commands) = self.current.take() {
            self.commands.push(commands);
        }
    }

    pub fn clear(&mut self) {
        self.commands.clear();
    }


    pub fn with_renderer(&self, id: RendererId) -> impl Iterator<Item = &CommandRange> {
        self.commands.iter().filter(move |cmd| cmd.0 == id).map(|cmd| &cmd.1)
    }

    pub fn with_renderer_rev(&self, id: RendererId) -> impl Iterator<Item = &CommandRange> {
        self.commands.iter().rev().filter(move |cmd| cmd.0 == id).map(|cmd| &cmd.1)
    }
}

impl Default for Commands {
    fn default() -> Self {
        Self::new(SurfaceState::default())
    }
}

pub type RenderPassId = u32;

#[derive(Debug)]
struct RenderPass {
    pre_passes: Range<u32>,
    sub_passes: Range<u32>,
    surface: SurfaceState,
    msaa_resolve: bool,
    msaa_blit: bool,
    temporary: bool,
}

#[derive(Debug)]
struct RenderPassSlice<'a> {
    pre_passes: &'a [PrePass],
    sub_passes: &'a [SubPass],
    surface: SurfaceState,
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
    fn add_pass(&mut self, pass: SurfaceState) {
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

    pub fn build(&mut self, surface: &SurfaceParameters) -> RenderPassesRequirements {
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

        self.sub_passes.sort_by_key(|pass| pass.z_index);

        // A bit per system specifying whether they have been used in the current
        // render pass yet.
        let mut renderer_bits: u64 = 0;

        let mut start = 0;
        let mut pre_start = 0;

        let mut current = self.sub_passes.first().unwrap().surface;
        let mut msaa_blit = false;

        for (idx, pass) in self.sub_passes.iter().enumerate() {
            let req_bit: u64 = 1 << pass.renderer_id;
            let flush_pass = pass.surface != current
                || (pass.require_pre_pass && renderer_bits & req_bit != 0);
 
            //println!(" - sub pass msaa:{}, changed:{msaa_changed}", pass.use_msaa);
            if flush_pass {
                // When transitioning from msaa to non-msaa targets, resolve the msaa
                // target into the non msaa one.
                let msaa_resolve = current.msaa && !pass.surface.msaa;
                //println!("   -> flush msaa resolve {msaa_resolve} blit {msaa_blit}");
                self.passes.push(RenderPass {
                    pre_passes: u32_range(pre_start..self.pre_passes.len()),
                    sub_passes: u32_range(start..idx),
                    surface: current,
                    msaa_resolve,
                    msaa_blit,
                    temporary: false,
                });
                requirements.add_pass(current);
                requirements.temporary |= disable_msaa_blit_from_main_target && msaa_blit;

                // When transitioning from non-msaa to msaa, blit the non-msaa target
                // into the msaa one.
                msaa_blit = !current.msaa && pass.surface.msaa;
                if msaa_blit && disable_msaa_blit_from_main_target {
                    for pass in self.passes.iter_mut().rev() {
                        if pass.surface.msaa {
                            break
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
            self.passes.push(RenderPass {
                pre_passes: u32_range(pre_start..self.pre_passes.len()),
                sub_passes: u32_range(start..self.sub_passes.len()),
                surface: current,
                msaa_resolve: current.msaa,
                msaa_blit,
                temporary: false,
            });

            requirements.add_pass(current);
            requirements.temporary |= msaa_blit;
        }

        requirements
    }

    fn iter(&self) -> impl Iterator<Item = RenderPassSlice> {
        self.passes.iter().map(|pass|
            RenderPassSlice {
                pre_passes: &self.pre_passes[usize_range(pass.pre_passes.clone())],
                sub_passes: &self.sub_passes[usize_range(pass.sub_passes.clone())],
                surface: pass.surface,
                msaa_resolve: pass.msaa_resolve,
                msaa_blit: pass.msaa_blit,
                temporary: pass.temporary,
            }
        )
    }
}

// TODO: when building render passes we support transitioning between different surface states. At the moment
// renderers make a best effort to match the initial surface state read at the beginning of the frame, but it
// would make more sense to explicitly allow transitioning the surface state or allow renderers to express
// constraints. The problem with resolving constraints while building the render passes is that the renderers
// need to know about the presence of an opaque pass earlier.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SurfaceParameters {
    size: Size2D<u32>,
    state: SurfaceState,
    clear: Option<Color>,
    /// If true, the main target cannot be sampled (for example a swapchain's target).
    write_only_target: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SurfaceState {
    pub depth: bool,
    pub msaa: bool,
    pub stencil: bool,
}

impl Default for SurfaceState {
    fn default() -> Self {
        SurfaceState { depth: false, msaa: false, stencil: false }
    }
}

impl SurfaceState {
    pub fn msaa(&self) -> bool { self.msaa }
    pub fn depth(&self) -> bool { self.depth }
    pub fn stencil(&self) -> bool { self.stencil }
    pub fn depth_or_stencil(&self) -> bool { self.depth || self.stencil }
    pub fn surface_config(&self, use_depth: bool, stencil: Option<FillRule>) -> SurfaceConfig {
        SurfaceConfig {
            msaa: self.msaa,
            depth: match (use_depth, self.depth) {
                (true, true) => DepthMode::Enabled,
                (false, true) => DepthMode::Ignore,
                (false, false) => DepthMode::None,
                (true, false) => panic!("Attempting to use the depth buffer on a surface that does not have one"),
            },
            stencil: match (stencil, self.stencil) {
                (None, false) => StencilMode::None,
                (None, true) => StencilMode::Ignore,
                (Some(FillRule::EvenOdd), true) => StencilMode::EvenOdd,
                (Some(FillRule::NonZero), true) => StencilMode::NonZero,
                (Some(_), false) => panic!("Attempting to use the stencil buffer on a surface that does not have one"),
            },
        }
    }
}

impl SurfaceParameters {
    #[inline]
    pub fn new(size: Size2D<u32>) -> Self {
        SurfaceParameters {
            size,
            state: SurfaceState::default(),
            clear: Some(Color::BLACK),
            write_only_target: true,
        }
    }

    #[inline]
    pub fn with_opaque_pass(mut self, enabled: bool) -> Self {
        self.state.depth = enabled;
        self
    }

    #[inline]
    pub fn with_msaa(mut self, enabled: bool) -> Self {
        self.state.msaa = enabled;
        self
    }

    #[inline]
    pub fn with_clear(mut self, clear: Option<Color>) -> Self {
        self.clear = clear;
        self
    }

    #[inline]
    pub fn size(&self) -> Size2D<u32> {
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
    pub fn state(&self) -> SurfaceState {
        self.state
    }
}

pub struct RenderPassState {
    pub output_type: OutputType,
    pub surface: SurfaceState,
}

impl RenderPassState {
    #[inline]
    pub fn surface_config(&self, use_depth: bool, stencil: Option<FillRule>) -> SurfaceConfig {
        self.surface.surface_config(use_depth, stencil)
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

// TODO: canvas isn't a great name for this.
#[derive(Default)]
pub struct Canvas {
    // TODO: maybe split off the push/pop transforms builder thing so that this remains more compatible
    // with a retained scene model as well.
    pub transforms: Transforms,
    pub z_indices: ZIndices,
    pub commands: Commands,
    pub surface: SurfaceParameters,
    render_passes: RenderPasses,
    pub params: CanvasParams,
}

impl Canvas {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn begin_frame(&mut self, surface: SurfaceParameters) {
        self.transforms.clear();
        self.z_indices.clear();
        self.render_passes.clear();
        self.commands.clear();
        self.commands.surface = surface.state;
        self.surface = surface;
    }

    pub fn prepare(&mut self) {
        self.commands.flush();
    }

    pub fn reconfigure_surface(&mut self, state: SurfaceState) {
        if self.commands.surface == state {
            return;
        }

        self.commands.flush();
        self.commands.surface = state;
    }

    pub fn build_render_passes(&mut self, renderers: &mut[&mut dyn CanvasRenderer]) -> RenderPassesRequirements {
        for renderer in renderers {
            renderer.add_render_passes(&mut self.render_passes);
        }
        self.render_passes.build(&self.surface)
    }

    pub fn render(
        &self,
        renderers: &[&dyn CanvasRenderer],
        resources: &GpuResources,
        bindings: &dyn BindingResolver,
        shaders: &mut Shaders,
        _device: &wgpu::Device,
        common_resources: ResourcesHandle<CommonGpuResources>,
        target: &SurfaceResources,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let mut need_clear = self.surface.clear.is_some();
        #[cfg(debug_assertions)]
        if self.surface.clear.is_none() && self.surface.write_only_target && !self.render_passes.is_empty() {
            let first = self.render_passes.iter().next().unwrap();
            if first.surface.msaa || first.temporary {
                println!("Can't load content if the first pass is a temporary or msaa target");
            }
        }

        for pass in self.render_passes.iter() {
            for pre_pass in pass.pre_passes {
                renderers[pre_pass.renderer_id as usize].render_pre_pass(pre_pass.internal_index, shaders, resources, bindings, encoder);
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
                        self.surface.clear.map(Color::to_wgpu).unwrap_or(wgpu::Color::BLACK)
                    )
                } else {
                    wgpu::LoadOp::Load
                },
                store: !pass.msaa_resolve,
            };

            //println!("{label}: {pass:#?}");

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: if pass.msaa_resolve { Some(&target.main) } else { None },
                    ops,
                })],
                depth_stencil_attachment: if pass.surface.depth_or_stencil() {
                    Some(wgpu::RenderPassDepthStencilAttachment {
                        view: if pass.surface.msaa {
                            target.msaa_depth
                        } else {
                            target.depth
                        }.unwrap(),
                        depth_ops: if pass.surface.depth {
                            Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(0.0),
                                store: false,
                            })
                        } else {
                            None
                        },
                        stencil_ops: if pass.surface.stencil {
                            Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(128),
                                store: false,
                            })
                        } else {
                            None
                        },
                    })
                } else {
                    None
                }
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
                output_type: OutputType::Color,
                surface: pass.surface,
            };

            for sub_pass in pass.sub_passes {
                renderers[sub_pass.renderer_id as usize].render(
                    sub_pass.internal_index,
                    &pass_info,
                    shaders,
                    resources,
                    bindings,
                    &mut render_pass,
                );
            }
        }
    }
}

pub trait CanvasRenderer: AsAny {
    fn add_render_passes(&mut self, render_passes: &mut RenderPasses);
    fn render_pre_pass(
        &self,
        _index: u32,
        _shaders: &Shaders,
        _renderers: &GpuResources,
        _bindings: &dyn BindingResolver,
        _encoder: &mut wgpu::CommandEncoder,
    ) {}
    fn render<'pass, 'resources: 'pass>(
        &self,
        _index: u32,
        _pass_info: &RenderPassState,
        _shaders: &'resources Shaders,
        _renderers: &'resources GpuResources,
        _bindings: &'resources dyn BindingResolver,
        _render_pass: &mut wgpu::RenderPass<'pass>,
    ) {}
}

#[derive(Debug)]
pub struct SubPass {
    pub renderer_id: RendererId,
    pub internal_index: u32,
    pub z_index: ZIndex,
    pub require_pre_pass: bool,
    pub surface: SurfaceState,
}

#[derive(Debug)]
pub struct PrePass {
    pub renderer_id: RendererId,
    pub internal_index: u32,
}


pub struct DummyRenderer {
    pub system_id: RendererId,
    passes: Vec<ZIndex>,
}

impl DummyRenderer {
    pub fn new(system_id: RendererId) -> Self {
        DummyRenderer {
            system_id,
            passes: Vec::new(),
        }
    }

    pub fn begin_frame(&mut self) {
        self.passes.clear();
    }

    pub fn command(&mut self, canvas: &mut Canvas) {
        let z_index = canvas.z_indices.push();
        let index = self.passes.len() as RendererCommandIndex;
        self.passes.push(z_index);
        canvas.commands.push(self.system_id, index);
    }

    pub fn prepare(&mut self, canvas: &Canvas) {
        for _range in canvas.commands.with_renderer(self.system_id) {
            // A typical system would build batches here but this one does nothing.
        }
    }
}

impl CanvasRenderer for DummyRenderer {
    fn add_render_passes(&mut self, render_passes: &mut RenderPasses) {
        for (idx, z_index) in self.passes.iter().enumerate() {
            render_passes.push(SubPass {
                renderer_id: self.system_id,
                internal_index: idx as u32,
                z_index: *z_index,
                require_pre_pass: false,
                surface: SurfaceState::default(),
            });
        }
    }
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
            current_bindings: [BindingsId::NONE, BindingsId::NONE, BindingsId::NONE, BindingsId::NONE],
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



pub trait Shape {
    fn to_command(self) -> RecordedShape;
}

pub struct PathShape {
    pub path: Arc<Path>,
    pub fill_rule: FillRule,
    pub inverted: bool,
}

impl PathShape {
    pub fn new(path: Arc<Path>) -> Self {
        PathShape { path, fill_rule: FillRule::EvenOdd, inverted: false }
    }
    pub fn with_fill_rule(mut self, fill_rule: FillRule) -> Self {
        self.fill_rule = fill_rule;
        self
    }
    pub fn inverted(mut self) -> Self {
        self.inverted = !self.inverted;
        self
    }
}

impl Shape for PathShape {
    fn to_command(self) -> RecordedShape {
        RecordedShape::Path(self)
    }
}

impl Shape for Arc<Path> {
    fn to_command(self) -> RecordedShape {
        RecordedShape::Path(PathShape::new(self))
    }
}

pub struct Circle {
    pub center: Point,
    pub radius: f32,
    pub inverted: bool,
}

impl Circle {
    pub fn new(center: Point, radius: f32) -> Self {
        Circle { center, radius, inverted: false }
    }

    pub fn inverted(mut self) -> Self {
        self.inverted = !self.inverted;
        self
    }

    pub fn aabb(&self) -> Box2D<f32> {
        Box2D {
            min: self.center - vec2(self.radius, self.radius),
            max: self.center + vec2(self.radius, self.radius),
        }
    }
}

impl Shape for Circle {
    fn to_command(self) -> RecordedShape {
        RecordedShape::Circle(self)
    }
}

impl Shape for Box2D<f32> {
    fn to_command(self) -> RecordedShape {
        RecordedShape::Rect(self)
    }
}

impl Shape for Box2D<i32> {
    fn to_command(self) -> RecordedShape {
        RecordedShape::Rect(self.to_f32())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct All;

impl Shape for All {
    fn to_command(self) -> RecordedShape {
        RecordedShape::Canvas
    }
}

// TODO: the enum prevents other types of shapes from being added externally.
pub enum RecordedShape {
    Path(PathShape),
    Rect(Box2D<f32>),
    Circle(Circle),
    Canvas,
}

pub struct Fill {
    pub shape: RecordedShape,
    pub pattern: BuiltPattern,
    pub transform: TransformId,
    pub z_index: ZIndex,
}
