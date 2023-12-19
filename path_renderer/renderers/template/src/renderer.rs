use core::{
    batching::{BatchFlags, BatchList},
    canvas::{
        CanvasRenderer, Context, DrawHelper, RenderContext, RenderPassState, RendererId, SubPass,
        SurfacePassConfig, ZIndex, FillPath,
    },
    pattern::{BindingsId, BuiltPattern},
    resources::{CommonGpuResources, GpuResources, ResourcesHandle},
    shape::FilledPath,
    transform::TransformId,
    units::LocalRect,
    usize_range, wgpu, gpu::shader::{RenderPipelineIndex, PrepareRenderPipelines},
};
use std::ops::Range;

use crate::TemplateGpuResources;

struct BatchInfo {
    draws: Range<u32>,
    surface: SurfacePassConfig,
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
    z_index: ZIndex,
}

struct Draw {
    indices: Range<u32>,
    pattern_inputs: BindingsId,
    pipeline_idx: RenderPipelineIndex,
}

pub struct MeshRenderer {
    renderer_id: RendererId,
    common_resources: ResourcesHandle<CommonGpuResources>,
    _resources: ResourcesHandle<TemplateGpuResources>,
    tolerenace: f32,

    batches: BatchList<Fill, BatchInfo>,
    draws: Vec<Draw>,
}

impl MeshRenderer {
    pub fn new(
        renderer_id: RendererId,
        common_resources: ResourcesHandle<CommonGpuResources>,
        resources: ResourcesHandle<TemplateGpuResources>,
        res: &TemplateGpuResources,
    ) -> Self {
        MeshRenderer {
            renderer_id,
            common_resources,
            _resources: resources,
            tolerenace: 0.25,

            draws: Vec::new(),
            batches: BatchList::new(renderer_id),
        }
    }

    pub fn supports_surface(&self, _surface: SurfacePassConfig) -> bool {
        true
    }

    pub fn begin_frame(&mut self, ctx: &Context) {
        self.draws.clear();
        self.batches.clear();
    }

    pub fn fill_path<P: Into<FilledPath>>(
        &mut self,
        ctx: &mut Context,
        path: P,
        pattern: BuiltPattern,
    ) {
        self.fill_shape(ctx, Shape::Path(path.into()), pattern);
    }

    fn fill_shape(&mut self, ctx: &mut Context, shape: Shape, pattern: BuiltPattern) {
        let transform = ctx.transforms.current_id();
        let z_index = ctx.z_indices.push();

        let aabb = ctx
            .transforms
            .get_current()
            .matrix()
            .outer_transformed_box(&shape.aabb());

        let mut batch_flags = BatchFlags::empty();
        if pattern.is_opaque && ctx.surface.current_config().depth {
            batch_flags |= BatchFlags::ORDER_INDEPENDENT;
        }

        let (commands, info) = self.batches.find_or_add_batch(
            &mut ctx.batcher,
            &pattern.batch_key(),
            &aabb,
            batch_flags,
            &mut || BatchInfo {
                draws: 0..0,
                surface: ctx.surface.current_config(),
            },
        );
        info.surface = ctx.surface.current_config();
        commands.push(Fill {
            shape,
            pattern,
            transform,
            z_index,
        });
    }

    pub fn prepare(&mut self, ctx: &Context, shaders: &mut PrepareRenderPipelines) {
        if self.batches.is_empty() {
            return;
        }

        let id = self.renderer_id;
        let mut batches = self.batches.take();
        for batch_id in ctx
            .batcher
            .batches()
            .iter()
            .filter(|batch| batch.renderer == id)
        {
            let (commands, info) = &mut batches.get_mut(batch_id.index);

            let surface = info.surface;

            let draw_start = self.draws.len() as u32;
            let mut key = commands
                .first()
                .as_ref()
                .unwrap()
                .pattern
                .shader_and_bindings();


            for fill in commands
                .iter()
                .filter(|fill| !surface.depth || !fill.pattern.is_opaque)
            {
                if key != fill.pattern.shader_and_bindings() {
                    // self.draws.push(...)
                }

                self.prepare_fill(fill, ctx);
            }

            // if commands to flush...

            let draws = draw_start..self.draws.len() as u32;
            info.draws = draws;
        }

        self.batches = batches;
    }

    fn prepare_fill(&mut self, fill: &Fill, ctx: &Context) {
        let transform = ctx.transforms.get(fill.transform).matrix();
        let z_index = fill.z_index;
        let pattern = fill.pattern.data;

        match &fill.shape {
            Shape::Path(shape) => {
            }
        }
    }

    pub fn upload(
        &mut self,
        resources: &mut GpuResources,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        let res = &mut resources[self.common_resources];
    }
}

impl CanvasRenderer for MeshRenderer {
    fn render<'pass, 'resources: 'pass>(
        &self,
        sub_passes: &[SubPass],
        _surface_info: &RenderPassState,
        ctx: RenderContext<'resources>,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        let common_resources = &ctx.resources[self.common_resources];

        render_pass.set_bind_group(
            0,
            &common_resources.main_target_and_gpu_store_bind_group,
            &[],
        );

        let mut helper = DrawHelper::new();

        for sub_pass in sub_passes {
            let (_, batch_info) = self.batches.get(sub_pass.internal_index);
            for draw in &self.draws[usize_range(batch_info.draws.clone())] {
                let pipeline = ctx.render_pipelines.get(draw.pipeline_idx).unwrap();

                helper.resolve_and_bind(1, draw.pattern_inputs, ctx.bindings, render_pass);

                render_pass.set_pipeline(pipeline);
                render_pass.draw_indexed(draw.indices.clone(), 0, 0..1);
            }
        }
    }
}

impl FillPath for MeshRenderer {
    fn fill_path(
        &mut self,
        ctx: &mut Context,
        path: FilledPath,
        pattern: BuiltPattern,
    ) {
        self.fill_shape(ctx, Shape::Path(path), pattern);
    }
}
