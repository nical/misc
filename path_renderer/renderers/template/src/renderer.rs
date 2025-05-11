#![allow(unused)]
//! An empty render that serves as a template for quickly adding new renderers

use core::{
    batching::{BatchFlags, BatchId, BatchList}, context::{
        BuiltRenderPass, DrawHelper, RenderPassBuilder, RenderPassContext, RendererId, SurfacePassConfig, ZIndex
    }, gpu::shader::{PrepareRenderPipelines, RenderPipelineIndex}, pattern::BuiltPattern, resources::{CommonGpuResources, GpuResources}, shape::FilledPath, transform::{TransformId, Transforms}, units::LocalRect, usize_range, wgpu, BindingsId, PrepareContext, UploadContext
};
use std::ops::Range;

struct BatchInfo {
    draws: Range<u32>,
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

pub struct TemplateRenderer {
    renderer_id: RendererId,

    batches: BatchList<Fill, BatchInfo>,
    draws: Vec<Draw>,
}

impl TemplateRenderer {
    pub(crate) fn new(
        renderer_id: RendererId,
    ) -> Self {
        TemplateRenderer {
            renderer_id,

            draws: Vec::new(),
            batches: BatchList::new(renderer_id),
        }
    }

    pub fn supports_surface(&self, _surface: SurfacePassConfig) -> bool {
        true
    }

    pub fn begin_frame(&mut self, _ctx: &RenderPassBuilder) {
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

        let mut batch_flags = BatchFlags::empty();
        if pattern.is_opaque && ctx.surface.depth {
            batch_flags |= BatchFlags::ORDER_INDEPENDENT;
        }

        self.batches.find_or_add_batch(
            ctx,
            &pattern.batch_key(),
            &aabb,
            batch_flags,
            &mut || BatchInfo {
                draws: 0..0,
            },
        ).push(Fill {
            shape,
            pattern,
            transform,
            z_index,
        });
    }

    fn prepare_fill(&mut self, fill: &Fill, transforms: &Transforms) {
        let transform = transforms.get(fill.transform).matrix();
        let z_index = fill.z_index;
        let pattern = fill.pattern.data;

        // ...
    }
}

impl core::Renderer for TemplateRenderer {
    fn prepare(&mut self, ctx: &mut PrepareContext) {
        let pass = &ctx.pass;

        if self.batches.is_empty() {
            return;
        }

        let id = self.renderer_id;
        let mut batches = self.batches.take();
        for batch_id in pass
            .batches()
            .iter()
            .filter(|batch| batch.renderer == id)
        {
            let (commands, surface, info) = &mut batches.get_mut(batch_id.index);

            let draw_start = self.draws.len() as u32;
            let key = commands
                .first()
                .as_ref()
                .unwrap()
                .pattern
                .shader_and_bindings();

            for fill in commands.iter() {
                if key != fill.pattern.shader_and_bindings() {
                    // self.draws.push(...)
                }

                self.prepare_fill(fill, &ctx.transforms);
            }

            // if commands to flush...

            let draws = draw_start..self.draws.len() as u32;
            info.draws = draws;
        }

        self.batches = batches;
    }

    fn render<'pass, 'resources: 'pass, 'tmp>(
        &self,
        batches: &[BatchId],
        _surface_info: &SurfacePassConfig,
        ctx: core::RenderContext<'resources, 'tmp>,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        let mut helper = DrawHelper::new();

        for batch_id in batches {
            let (_, _, batch_info) = self.batches.get(batch_id.index);

            for draw in &self.draws[usize_range(batch_info.draws.clone())] {
                let pipeline = ctx.render_pipelines.get(draw.pipeline_idx).unwrap();

                helper.resolve_and_bind(1, draw.pattern_inputs, ctx.bindings, render_pass);

                render_pass.set_pipeline(pipeline);
                render_pass.draw_indexed(draw.indices.clone(), 0, 0..1);
            }
        }
    }
}

impl core::FillPath for TemplateRenderer {
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
