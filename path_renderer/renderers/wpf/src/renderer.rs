use core::{
    batching::{BatchFlags, BatchList, BatchId},
    bytemuck,
    context::{
        DrawHelper, RendererId,
        SurfacePassConfig, RenderPassContext, BuiltRenderPass,
    },
    gpu::{
        shader::{
            PrepareRenderPipelines, RenderPipelineIndex, RenderPipelineKey, BlendMode, BaseShaderId,
        },
        DynBufferRange,
    },
    pattern::{BindingsId, BuiltPattern},
    resources::GpuResources,
    shape::FilledPath,
    transform::{TransformId, Transforms},
    units::{LocalRect, SurfaceIntSize},
    usize_range, wgpu,
};
use lyon_path::{FillRule, traits::PathIterator};
use wpf_gpu_raster::{PathBuilder, FillMode};
use std::ops::Range;

pub const PATTERN_KIND_COLOR: u32 = 0;
pub const PATTERN_KIND_SIMPLE_LINEAR_GRADIENT: u32 = 1;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vertex {
    pub x: f32,
    pub y: f32,
    pub coverage: f32,
    pub pattern: u32,
}

unsafe impl bytemuck::Zeroable for Vertex {}
unsafe impl bytemuck::Pod for Vertex {}

struct BatchInfo {
    draws: Range<u32>,
    blend_mode: BlendMode,
}

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
    pattern: BuiltPattern,
    transform: TransformId,
}

struct Draw {
    vertices: Range<u32>,
    pattern_inputs: BindingsId,
    pipeline_idx: RenderPipelineIndex,
}

pub struct WpfMeshRenderer {
    renderer_id: RendererId,
    builder: PathBuilder,
    vertices: Vec<Vertex>,

    batches: BatchList<Fill, BatchInfo>,
    draws: Vec<Draw>,
    vbo_range: Option<DynBufferRange>,
    ibo_range: Option<DynBufferRange>,
    base_shader: BaseShaderId,
}

impl WpfMeshRenderer {
    pub(crate) fn new(
        renderer_id: RendererId,
        base_shader: BaseShaderId,
    ) -> Self {
        WpfMeshRenderer {
            renderer_id,
            builder: PathBuilder::new(),
            vertices: Vec::new(),

            draws: Vec::new(),
            batches: BatchList::new(renderer_id),
            vbo_range: None,
            ibo_range: None,
            base_shader,
        }
    }

    pub fn supports_surface(&self, surface: SurfacePassConfig) -> bool {
        !surface.msaa
    }

    pub fn begin_frame(&mut self) {
        self.vertices.clear();
        self.draws.clear();
        self.batches.clear();
        self.vbo_range = None;
        self.ibo_range = None;
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
        let _ = ctx.z_indices.push();

        let aabb = transforms
            .get_current()
            .matrix()
            .outer_transformed_box(&shape.aabb());

        self.batches.find_or_add_batch(
            ctx,
            &pattern.batch_key(),
            &aabb,
            BatchFlags::empty(),
            &mut || BatchInfo {
                draws: 0..0,
                blend_mode: pattern.blend_mode.with_alpha(true),
            },
        ).push(Fill {
            shape,
            pattern,
            transform,
        });
    }

    pub fn prepare(&mut self, pass: &BuiltRenderPass, transforms: &Transforms, shaders: &mut PrepareRenderPipelines) {
        if self.batches.is_empty() {
            return;
        }

        let surface_size = pass.surface_size();

        let id = self.renderer_id;
        let mut batches = self.batches.take();
        for batch_id in pass
            .batches()
            .iter()
            .filter(|batch| batch.renderer == id)
        {
            let (commands, surface, info) = &mut batches.get_mut(batch_id.index);

            let draw_start = self.draws.len() as u32;
            let mut key = commands
                .first()
                .as_ref()
                .unwrap()
                .pattern
                .shader_and_bindings();

            let mut geom_start = self.vertices.len() as u32;
            for fill in commands.iter() {
                if key != fill.pattern.shader_and_bindings() {
                    let end = self.vertices.len() as u32;
                    if end > geom_start {
                        self.draws.push(Draw {
                            vertices: geom_start..end,
                            pattern_inputs: key.1,
                            pipeline_idx: shaders.prepare(RenderPipelineKey::new(
                                self.base_shader,
                                key.0,
                                info.blend_mode,
                                surface.draw_config(false, None),
                            )),
                        });
                    }
                    geom_start = end;
                    key = fill.pattern.shader_and_bindings();
                }
                self.prepare_fill(fill, surface_size, transforms);
            }

            let end = self.vertices.len() as u32;
            if end > geom_start {
                self.draws.push(Draw {
                    vertices: geom_start..end,
                    pattern_inputs: key.1,
                    pipeline_idx: shaders.prepare(RenderPipelineKey::new(
                        self.base_shader,
                        key.0,
                        info.blend_mode,
                        surface.draw_config(false, None),
                    )),
                });
            }

            let draws = draw_start..self.draws.len() as u32;
            info.draws = draws;
        }

        self.batches = batches;
    }

    fn prepare_fill(&mut self, fill: &Fill, surface_size: SurfaceIntSize, transforms: &Transforms) {
        let transform = transforms.get(fill.transform).matrix();
        let pattern = fill.pattern.data;

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
                    self.vertices.push(Vertex {
                        x: v2.x,
                        y: v2.y,
                        coverage: v2.coverage,
                        pattern,
                    });
                    self.vertices.push(Vertex {
                        x: v1.x,
                        y: v1.y,
                        coverage: v1.coverage,
                        pattern,
                    });
                    self.vertices.push(Vertex {
                        x: v0.x,
                        y: v0.y,
                        coverage: v0.coverage,
                        pattern,
                    });
                }
            }
        }
    }

    pub fn upload(
        &mut self,
        resources: &mut GpuResources,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        self.vbo_range = resources.common
            .vertices
            .upload(device, bytemuck::cast_slice(&self.vertices));
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
    fn render<'pass, 'resources: 'pass>(
        &self,
        batches: &[BatchId],
        _surface_info: &SurfacePassConfig,
        ctx: core::RenderContext<'resources>,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        render_pass.set_vertex_buffer(
            0,
            ctx.resources.common
                .vertices
                .get_buffer_slice(self.vbo_range.as_ref().unwrap()),
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
