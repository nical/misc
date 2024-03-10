#![allow(unused)]
//! An empty render that serves as a template for quickly adding new renderers

use core::{
    batching::{BatchFlags, BatchList, BatchId},
    context::{
        RenderPassBuilder, DrawHelper, RendererId,
        SurfacePassConfig, ZIndex, RenderPassContext, BuiltRenderPass,
    },
    pattern::{BindingsId, BuiltPattern},
    resources::{CommonGpuResources, GpuResources, ResourcesHandle},
    shape::FilledPath,
    transform::{TransformId, Transforms},
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

pub struct TemplateRenderer {
    renderer_id: RendererId,
    common_resources: ResourcesHandle<CommonGpuResources>,
    resources: ResourcesHandle<TemplateGpuResources>,

    batches: BatchList<Fill, BatchInfo>,
    draws: Vec<Draw>,
}

impl TemplateRenderer {
    pub fn new(
        renderer_id: RendererId,
        common_resources: ResourcesHandle<CommonGpuResources>,
        resources: ResourcesHandle<TemplateGpuResources>,
        res: &TemplateGpuResources,
    ) -> Self {
        TemplateRenderer {
            renderer_id,
            common_resources,
            resources: resources,

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

        let (commands, info) = self.batches.find_or_add_batch(
            &mut ctx.batcher,
            &pattern.batch_key(),
            &aabb,
            batch_flags,
            &mut || BatchInfo {
                draws: 0..0,
                surface: ctx.surface,
            },
        );
        info.surface = ctx.surface;
        commands.push(Fill {
            shape,
            pattern,
            transform,
            z_index,
        });
    }

    pub fn prepare(&mut self, pass: &BuiltRenderPass, transforms: &Transforms, shaders: &mut PrepareRenderPipelines) {
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
            let (commands, info) = &mut batches.get_mut(batch_id.index);

            let surface = info.surface;

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

                self.prepare_fill(fill, transforms);
            }

            // if commands to flush...

            let draws = draw_start..self.draws.len() as u32;
            info.draws = draws;
        }

        self.batches = batches;
    }

    fn prepare_fill(&mut self, fill: &Fill, transforms: &Transforms) {
        let transform = transforms.get(fill.transform).matrix();
        let z_index = fill.z_index;
        let pattern = fill.pattern.data;

        // ...
    }

    pub fn upload(
        &mut self,
        resources: &mut GpuResources,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
    }
}

impl core::Renderer for TemplateRenderer {
    fn render<'pass, 'resources: 'pass>(
        &self,
        batches: &[BatchId],
        _surface_info: &SurfacePassConfig,
        ctx: core::RenderContext<'resources>,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        let common_resources = &ctx.resources[self.common_resources];

        render_pass.set_bind_group(
            0,
            &common_resources.main_target_and_gpu_store_bind_group,
            &[],
        );

        let mut helper = DrawHelper::new();

        for batch_id in batches {
            let (_, batch_info) = self.batches.get(batch_id.index);

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
