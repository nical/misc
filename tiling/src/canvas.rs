// TODO: this module lumps together a bunch of structural concepts some of which
// are poorly named and/or belong elsewhere.

use std::sync::Arc;
use lyon::math::{Point};
use lyon::path::{Path, FillRule};
use lyon::geom::euclid::default::{Transform2D, Box2D, Size2D};
use wgpu::util::DeviceExt;
use crate::Color;
use crate::pattern::checkerboard::*;
use crate::pattern::simple_gradient::*;
use std::{ops::Range};
use crate::gpu::{DynamicStore, GpuStore, GpuTargetDescriptor};
use std::{any::Any, marker::PhantomData};

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

pub struct Commands {
    commands: Vec<(RendererId, Range<RendererCommandIndex>)>,
}

impl Commands {
    pub fn new() -> Self {
        Commands { commands: Vec::new() }
    }

    pub fn push(&mut self, renderer: RendererId, internal_index: RendererCommandIndex) {
        if let Some((sys, range)) = self.commands.last_mut() {
            if *sys == renderer && range.end == internal_index {
                range.end += 1;
                return;
            }
        }

        self.commands.push((renderer, internal_index..(internal_index + 1)));
    }

    pub fn clear(&mut self) {
        self.commands.clear();
    }


    pub fn with_renderer(&self, id: RendererId) -> impl Iterator<Item = &Range<RendererCommandIndex>> {
        self.commands.iter().filter(move |cmd| cmd.0 == id).map(|cmd| &cmd.1)
    }

    pub fn with_renderer_rev(&self, id: RendererId) -> impl Iterator<Item = &Range<RendererCommandIndex>> {
        self.commands.iter().filter(move |cmd| cmd.0 == id).map(|cmd| &cmd.1)
    }
}

impl Default for Commands {
    fn default() -> Self {
        Self::new()
    }
}

pub type RenderPassId = u32;

#[derive(Default)]
pub struct RenderPasses {
    sub_passes: Vec<SystemRenderPass>,
    pre_passes: Vec<SystemPrePass>,
    passes: Vec<(Range<usize>, Range<usize>)>,
}

impl RenderPasses {
    pub fn push(&mut self, pass: SystemRenderPass) {
        self.sub_passes.push(pass);
    }

    pub fn clear(&mut self) {
        self.sub_passes.clear();
        self.passes.clear();
        self.pre_passes.clear();
    }

    pub fn build(&mut self) {
        self.sub_passes.sort_by_key(|pass| pass.z_index);

        // A bit per system specifying whether they have been used in the current
        // render pass yet.
        let mut req: u64 = 0;

        let mut start = 0;
        let mut pre_start = 0;

        for (idx, pass) in self.sub_passes.iter().enumerate() {
            let req_bit: u64 = 1 << pass.renderer_id;
            if pass.require_pre_pass {
                if req & req_bit != 0 {
                    self.passes.push((start..idx, pre_start..self.pre_passes.len()));
                    start = idx;
                    pre_start = self.pre_passes.len();
                    req = 0;
                }

                self.pre_passes.push(SystemPrePass {
                    renderer_id: pass.renderer_id,
                    internal_index: pass.internal_index,
                });
            }

            req |= req_bit;
        }

        if start < self.sub_passes.len() {
            self.passes.push((start .. self.sub_passes.len(), pre_start..self.pre_passes.len()));
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&[SystemPrePass], &[SystemRenderPass])> {
        self.passes.iter().map(|(sub, pre)| (&self.pre_passes[pre.clone()], &self.sub_passes[sub.clone()]))
    }
}

#[derive(Clone, Debug, Default)]
pub struct SurfaceState {
    size: Size2D<u32>,
    opaque_pass: bool,
}

impl SurfaceState {
    #[inline]
    pub fn new(size: Size2D<u32>) -> Self {
        SurfaceState { size, opaque_pass: false }
    }

    #[inline]
    pub fn with_opaque_pass(mut self) -> Self {
        self.opaque_pass = true;
        self
    }

    #[inline]
    pub fn size(&self) -> Size2D<u32> {
        self.size
    }

    #[inline]
    pub fn msaa(&self) -> Option<u8> {
        None
    }

    #[inline]
    pub fn opaque_pass(&self) -> bool {
        self.opaque_pass
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
    pub surface: SurfaceState,
    render_passes: RenderPasses,
}

impl Canvas {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn begin_frame(&mut self, surface: SurfaceState) {
        self.transforms.clear();
        self.z_indices.clear();
        self.commands.clear();
        self.render_passes.clear();
        self.surface = surface;
    }

    pub fn build_render_passes(&mut self, renderers: &mut[&mut dyn CanvasRenderer]) {
        for renderer in renderers {
            renderer.add_render_passes(&mut self.render_passes);
        }
        self.render_passes.build();
    }

    pub fn render(
        &self,
        renderers: &[&dyn CanvasRenderer],
        resources: &GpuResources,
        target: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        for (pre_passes, sub_passes) in self.render_passes.iter() {
            for pre_pass in pre_passes {
                renderers[pre_pass.renderer_id as usize].render_pre_pass(pre_pass.internal_index, resources, encoder);
            }

            // TODO: avoid hard-coding this.
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Color target"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                    resolve_target: None,
                })],
                depth_stencil_attachment: None,
            });

            for sub_pass in sub_passes {
                renderers[sub_pass.renderer_id as usize].render(sub_pass.internal_index, resources, &mut render_pass);
            }
        }
    }
}

pub trait CanvasRenderer: AsAny {
    fn add_render_passes(&mut self, render_passes: &mut RenderPasses);
    fn render_pre_pass(
        &self,
        _index: u32,
        _renderers: &GpuResources,
        _encoder: &mut wgpu::CommandEncoder,
    ) {}
    fn render<'pass, 'resources: 'pass>(
        &self,
        _index: u32,
        _renderers: &'resources GpuResources,
        _render_pass: &mut wgpu::RenderPass<'pass>,
    ) {}
}

pub struct SystemRenderPass {
    pub renderer_id: RendererId,
    pub internal_index: u32,
    pub z_index: ZIndex,
    pub require_pre_pass: bool,
    // TODO: add support for optional order-independent pass.
}

pub struct SystemPrePass {
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
            render_passes.push(SystemRenderPass {
                renderer_id: self.system_id,
                internal_index: idx as u32,
                z_index: *z_index,
                require_pre_pass: false,
            });
        }
    }

    fn render_pre_pass(
        &self,
        _index: u32,
        _renderers: &GpuResources,
        _encoder: &mut wgpu::CommandEncoder,
    ) {}

    fn render<'pass, 'resources: 'pass>(
        &self,
        _index: u32,
        _renderers: &'resources GpuResources,
        _render_pass: &mut wgpu::RenderPass<'pass>,
    ) {}
}



pub struct CommonGpuResources {
    pub quad_ibo: wgpu::Buffer,
    pub vertices: DynamicStore,
    // TODO: right now there can be only one target per instance of this struct
    pub target_and_gpu_store_layout: wgpu::BindGroupLayout,
    pub main_target_and_gpu_store_bind_group: wgpu::BindGroup,
    pub main_target_descriptor_ubo: wgpu::Buffer,
    pub gpu_store_view: wgpu::TextureView,
}

impl CommonGpuResources {
    pub fn new(device: &wgpu::Device, target_size: Size2D<u32>, gpu_store: &GpuStore) -> Self {

        let atlas_desc_buffer_size = std::mem::size_of::<GpuTargetDescriptor>() as u64;
        let target_and_gpu_store_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Target and gpu store"),
            entries: &[
                // target descriptor
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(atlas_desc_buffer_size),
                    },
                    count: None,
                },
                // GPU store
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                }
            ],
        });

        let main_target_descriptor_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mask params"),
            contents: bytemuck::cast_slice(&[
                GpuTargetDescriptor::new(target_size.width, target_size.height)
            ]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let quad_indices = [0u16, 1, 2, 0, 2, 3];
        let quad_ibo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Quad indices"),
            contents: bytemuck::cast_slice(&quad_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let gpu_store_view = gpu_store.texture().create_view(&wgpu::TextureViewDescriptor {
            label: Some("gpu store"),
            format: Some(wgpu::TextureFormat::Rgba32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            base_array_layer: 0,
            mip_level_count: Some(1),
            array_layer_count: Some(1),
        });

        let main_target_and_gpu_store_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Main target & gpu store"),
            layout: &target_and_gpu_store_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(main_target_descriptor_ubo.as_entire_buffer_binding())
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&gpu_store_view),
                }
            ],
        });

        let vertices = DynamicStore::new_vertices(4096 * 32);

        CommonGpuResources {
            quad_ibo,
            vertices,
            target_and_gpu_store_layout,
            main_target_and_gpu_store_bind_group,
            main_target_descriptor_ubo,
            gpu_store_view,
        }
    }

    pub fn resize_target(&self, size: Size2D<u32>, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.main_target_descriptor_ubo,
            0,
            bytemuck::cast_slice(&[GpuTargetDescriptor::new(size.width, size.height)]),
        );
    }
}

impl RendererResources for CommonGpuResources {
    fn name(&self) -> &'static str { "CommonGpuResources" }

    fn begin_frame(&mut self) {
    }

    fn begin_rendering(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.vertices.unmap(encoder);
    }

    fn end_frame(&mut self) {
        self.vertices.end_frame();
    }

}

pub trait RendererResources: AsAny {
    fn name(&self) -> &'static str;
    fn begin_frame(&mut self) {}
    fn begin_rendering(&mut self, _encoder: &mut wgpu::CommandEncoder) {}
    fn end_frame(&mut self) {}
}

pub trait AsAny {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: 'static> AsAny for T {
    fn as_any(&self) -> &dyn Any { self as _ }
    fn as_any_mut(&mut self) -> &mut dyn Any { self as _ }
}

pub struct ResourcesHandle<T> {
    index: u8,
    _marker: PhantomData<T>,
}

impl<T> ResourcesHandle<T> {
    pub fn new(index: u8) -> Self {
        ResourcesHandle { index, _marker: PhantomData }
    }

    pub fn index(&self) -> usize { self.index as usize }
}

impl<T> Copy for ResourcesHandle<T> {}
impl<T> Clone for ResourcesHandle<T> {
    fn clone(&self) -> Self { *self }
}

pub struct GpuResources {
    systems: Vec<Box<dyn RendererResources>>,
}

impl GpuResources {
    pub fn new(systems: Vec<Box<dyn RendererResources>>) -> Self {
        GpuResources {
            systems,
        }
    }

    pub fn get<T: 'static>(&self, handle: ResourcesHandle<T>) -> &T {
        let result: Option<&T> = (*self.systems[handle.index()]).as_any().downcast_ref();
        #[cfg(debug_assertions)]
        if result.is_none() {
            panic!("Invalid type, got {:?}", self.systems[handle.index()].name());
        }
        result.unwrap()
    }

    pub fn get_mut<T: 'static>(&mut self, handle: ResourcesHandle<T>) -> &mut T {
        #[cfg(debug_assertions)]
        let name = self.systems[handle.index()].name();
        let result: Option<&mut T> = (*self.systems[handle.index()]).as_any_mut().downcast_mut();
        #[cfg(debug_assertions)]
        if result.is_none() {
            panic!("Invalid type, got {:?}", name);
        }

        result.unwrap()
    }

    pub fn begin_frame(&mut self) {
        for sys in &mut self.systems {
            sys.begin_frame();
        }
    }

    pub fn begin_rendering(&mut self, encoder: &mut wgpu::CommandEncoder) {
        for sys in &mut self.systems {
            sys.begin_rendering(encoder);
        }
    }

    pub fn end_frame(&mut self) {
        for sys in &mut self.systems {
            sys.end_frame();
        }
    }
}

impl<T: 'static> std::ops::Index<ResourcesHandle<T>> for GpuResources {
    type Output = T;
    fn index(&self, handle: ResourcesHandle<T>) -> &T {
        self.get(handle)
    }
}

impl<T: 'static> std::ops::IndexMut<ResourcesHandle<T>> for GpuResources {
    fn index_mut(&mut self, handle: ResourcesHandle<T>) -> &mut T {
        self.get_mut(handle)
    }
}





/*
 Ideally, in order to take advantage of occusion culling (currently not supporting nested groups)

 1--A-----------4--C-----6
    |              |
    +--2--B        +--5
          |
          +--3
.

tile order: 6 C* 5 C 4 A* B* 3 B 2 A 1

renderer order: 3, 2 B, 5, 1 A 4 C 6

for each render pass, first render associated push group(s).
if multiple groups don't fit into an atlas, create a new render pass.

*/







pub trait Shape {
    fn to_command(self) -> RecordedShape;
}

pub trait Pattern {
    fn to_command(self) -> RecordedPattern;
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

impl Pattern for Color {
    fn to_command(self) -> RecordedPattern {
        RecordedPattern::Color(self)
    }
}

impl Pattern for Gradient {
    fn to_command(self) -> RecordedPattern {
        RecordedPattern::Gradient(self)
    }
}

impl Pattern for Checkerboard {
    fn to_command(self) -> RecordedPattern {
        RecordedPattern::Checkerboard(self)
    }
}

pub enum RecordedPattern {
    Color(Color),
    Gradient(Gradient),
    Image(u32),
    Checkerboard(Checkerboard),
}

impl RecordedPattern {
    pub fn is_opaque(&self) -> bool {
        match self {
            Self::Color(color) => color.is_opaque(),
            Self::Gradient(g) => g.color0.is_opaque() && g.color1.is_opaque(),
            Self::Checkerboard(c) => c.color0.is_opaque() && c.color1.is_opaque(),
            Self::Image(_) => false, // TODO
        }
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
    pub pattern: RecordedPattern,
    pub transform: TransformId,
    pub z_index: ZIndex,
}
