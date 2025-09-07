use core::render_task::RenderTaskHandle;
use core::{bytemuck, wgpu, BindingsId, PrepareContext};
use core::batching::{BatchFlags, BatchId, BatchList};
use core::render_pass::{BuiltRenderPass, RenderPassConfig, RenderPassContext, RendererId, ZIndex};
use core::gpu::GpuBufferWriter;
use core::shading::{GeometryId, BlendMode, RenderPipelineIndex, RenderPipelineKey};
use core::utils::{DrawHelper, usize_range};

use core::pattern::BuiltPattern;
use core::shape::FilledPath;
use core::transform::{TransformId, Transforms};
use core::units::{LocalRect, SurfaceIntSize};

use lyon_path::{FillRule, traits::PathIterator};
use wpf_gpu_raster::{PathBuilder, FillMode};
use std::ops::Range;

pub const PATTERN_KIND_COLOR: u32 = 0;
pub const PATTERN_KIND_SIMPLE_LINEAR_GRADIENT: u32 = 1;


pub struct PrimitiveInfo {
    pub z_index: u32,
    pub pattern: u32,
    pub opacity: f32,
    pub render_task: RenderTaskHandle,
}

pub type EncodedPrimitiveInfo = [u32; 4];

impl PrimitiveInfo {
    pub fn encode(&self) -> EncodedPrimitiveInfo {
        [
            self.z_index,
            self.pattern,
            (self.opacity.max(0.0).min(1.0) * 65535.0) as u32,
            self.render_task.to_u32(),
        ]
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vertex {
    // Note: x and y chould be encoded into a pair of u16.
    pub x: f32,
    pub y: f32,
    // Note: coverage could be a u8.
    pub coverage: f32,
    pub prim_address: u32,
}

unsafe impl bytemuck::Zeroable for Vertex {}
unsafe impl bytemuck::Pod for Vertex {}

struct BatchInfo {
    draws: Range<u32>,
    blend_mode: BlendMode,
}

#[derive(Clone)]
enum Shape {
    Path(FilledPath),
}

impl Shape {
    pub fn aabb(&self) -> LocalRect {
        match self {
            // TODO: return the correct aabb for inverted shapes.
            Shape::Path(shape) => *shape.path.aabb(),
        }
    }
}

struct Fill {
    shape: Shape,
    render_task: RenderTaskHandle,
    pattern: BuiltPattern,
    transform: TransformId,
    z_index: ZIndex,
}

struct Draw {
    vertices: Range<u32>,
    pattern_inputs: BindingsId,
    pipeline_idx: RenderPipelineIndex,
}

pub struct WpfMeshRenderer {
    renderer_id: RendererId,
    builder: PathBuilder,

    batches: BatchList<Fill, BatchInfo>,
    draws: Vec<Draw>,
    geometry: GeometryId,
}

impl WpfMeshRenderer {
    pub(crate) fn new(
        renderer_id: RendererId,
        geometry: GeometryId,
    ) -> Self {
        WpfMeshRenderer {
            renderer_id,
            builder: PathBuilder::new(),

            draws: Vec::new(),
            batches: BatchList::new(renderer_id),
            geometry,
        }
    }

    pub fn supports_surface(&self, surface: RenderPassConfig) -> bool {
        !surface.msaa
    }

    pub fn begin_frame(&mut self) {
        self.draws.clear();
        self.batches.clear();
    }

    pub fn fill_path<P: Into<FilledPath>>(
        &mut self,
        ctx: &mut RenderPassContext,
        transforms: &Transforms,
        path: P,
        pattern: BuiltPattern,
    ) {
        self.fill_shape(ctx, transforms, Shape::Path(path.into()), pattern);
    }

    fn fill_shape(&mut self, ctx: &mut RenderPassContext, transforms: &Transforms, shape: Shape, pattern: BuiltPattern) {
        let transform = transforms.current_id();
        let z_index = ctx.z_indices.push();

        let aabb = transforms
            .get_current()
            .matrix()
            .outer_transformed_box(&shape.aabb());

        self.batches.add(
            ctx,
            &pattern.batch_key(),
            &aabb,
            BatchFlags::empty(),
            &mut || BatchInfo {
                draws: 0..0,
                blend_mode: pattern.blend_mode.with_alpha(true),
            },
            &mut |mut batch, task| {
                batch.push(Fill {
                    shape: shape.clone(),
                    render_task: task.handle,
                    pattern,
                    transform,
                    z_index,
                });
            }
        );
    }

    pub fn prepare_impl(&mut self, ctx: &mut PrepareContext, passes: &[BuiltRenderPass]) {
        if self.batches.is_empty() {
            return;
        }

        let transforms = &ctx.transforms;
        let worker_data = &mut ctx.workers.data();
        let shaders = &mut worker_data.pipelines;
        let mut vertices = worker_data.vertices.write_items::<Vertex>();
        let mut prim_buffer = worker_data.u32_buffer.write_items::<EncodedPrimitiveInfo>();

        let id = self.renderer_id;
        let mut batches = self.batches.take();
        for pass in passes {
            for batch_id in pass
                .batches()
                .iter()
                .filter(|batch| batch.renderer == id)
            {
                let (commands, surface, info) = &mut batches.get_mut(batch_id.index);
                let surface_size = pass.surface_size();

                let draw_start = self.draws.len() as u32;
                let mut key = commands
                    .first()
                    .as_ref()
                    .unwrap()
                    .pattern
                    .shader_and_bindings();

                let mut geom_start = vertices.pushed_items();
                for fill in commands.iter() {
                    if key != fill.pattern.shader_and_bindings() {
                        let end = vertices.pushed_items();
                        if end > geom_start {
                            self.draws.push(Draw {
                                vertices: geom_start..end,
                                pattern_inputs: key.1,
                                pipeline_idx: shaders.prepare(RenderPipelineKey::new(
                                    self.geometry,
                                    key.0,
                                    info.blend_mode,
                                    surface.draw_config(false, None),
                                )),
                            });
                        }
                        geom_start = end;
                        key = fill.pattern.shader_and_bindings();
                    }
                    self.prepare_fill(fill, surface_size, transforms, &mut vertices, &mut prim_buffer);
                }

                let end = vertices.pushed_items();
                if end > geom_start {
                    self.draws.push(Draw {
                        vertices: geom_start..end,
                        pattern_inputs: key.1,
                        pipeline_idx: shaders.prepare(RenderPipelineKey::new(
                            self.geometry,
                            key.0,
                            info.blend_mode,
                            surface.draw_config(false, None),
                        )),
                    });
                }

                let draws = draw_start..self.draws.len() as u32;
                info.draws = draws;
            }
        }

        self.batches = batches;
    }

    fn prepare_fill(&mut self, fill: &Fill, surface_size: SurfaceIntSize, transforms: &Transforms, vertices: &mut GpuBufferWriter, u32_buffer: &mut GpuBufferWriter) {
        let transform = transforms.get(fill.transform).matrix();

        let prim_address = u32_buffer.push(PrimitiveInfo {
            z_index: fill.z_index,
            pattern: fill.pattern.data,
            opacity: 1.0, // TODO,
            render_task: fill.render_task,
        }.encode()).to_u32();

        match &fill.shape {
            Shape::Path(shape) => {
                let transform = transform.to_untyped();

                for evt in shape.path.iter().transformed(&transform) {
                    use lyon_path::PathEvent;
                    match evt {
                        PathEvent::Begin { at } => {
                            self.builder.move_to(at.x, at.y);
                        }
                        PathEvent::Line { to, .. } => {
                            self.builder.line_to(to.x, to.y);
                        }
                        PathEvent::Quadratic { ctrl, to, .. } => {
                            self.builder.quad_to(ctrl.x, ctrl.y, to.x, to.y);
                        }
                        PathEvent::Cubic { ctrl1, ctrl2, to, .. } => {
                            self.builder.curve_to(ctrl1.x, ctrl1.y, ctrl2.x, ctrl2.y, to.x, to.y);
                        }
                        PathEvent::End { close, .. } => {
                            if close {
                                self.builder.close();
                            }
                        }
                    }
                }

                //self.builder.set_rasterization_truncates(true);
                self.builder.set_fill_mode(match shape.fill_rule {
                    FillRule::EvenOdd => FillMode::EvenOdd,
                    FillRule::NonZero => FillMode::Winding,
                });

                let output = self.builder.rasterize_to_tri_list(0, 0, surface_size.width, surface_size.height);

                // TODO: wpf-gpu-raster does not have a way to reset the builder to avoid memory allocations.
                self.builder = PathBuilder::new();

                for v in output.chunks(3) {
                    let v2 = &v[2];
                    let v1 = &v[1];
                    let v0 = &v[0];
                    // Note: We can't push the 3 vertices in a single slice because slices are
                    // guaranteed to go in the same chunk which means that the last slice of a
                    // chunk may be moved to another chunk and leave a hole if it does not fit
                    // exactly.
                    vertices.push(Vertex { x: v2.x, y: v2.y, coverage: v2.coverage, prim_address });
                    vertices.push(Vertex { x: v1.x, y: v1.y, coverage: v1.coverage, prim_address });
                    vertices.push(Vertex { x: v0.x, y: v0.y, coverage: v0.coverage, prim_address });
                }
            }
        }
    }
}

impl core::FillPath for WpfMeshRenderer {
    fn fill_path(
        &mut self,
        ctx: &mut RenderPassContext,
        transforms: &Transforms,
        path: FilledPath,
        pattern: BuiltPattern,
    ) {
        self.fill_shape(ctx, transforms, Shape::Path(path), pattern);
    }
}

impl core::Renderer for WpfMeshRenderer {
    fn prepare(&mut self, ctx: &mut PrepareContext, passes: &[BuiltRenderPass]) {
        self.prepare_impl(ctx, passes);
    }

    fn render<'pass, 'resources: 'pass, 'tmp>(
        &self,
        batches: &[BatchId],
        _surface_info: &RenderPassConfig,
        ctx: core::RenderContext<'resources, 'tmp>,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        render_pass.set_vertex_buffer(
            0,
            ctx.resources.common.vertices.as_buffer().unwrap().slice(..),
        );

        let mut helper = DrawHelper::new();

        for batch_id in batches {
            let (_, _, batch_info) = self.batches.get(batch_id.index);
            for draw in &self.draws[usize_range(batch_info.draws.clone())] {
                let pipeline = ctx.render_pipelines.get(draw.pipeline_idx).unwrap();

                helper.resolve_and_bind(1, draw.pattern_inputs, ctx.bindings, render_pass);

                render_pass.set_pipeline(pipeline);
                render_pass.draw(draw.vertices.clone(), 0..1);
            }
        }
    }
}
