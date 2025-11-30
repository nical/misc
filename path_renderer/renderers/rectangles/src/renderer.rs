use crate::{resources::Instance, InstanceFlags};
use core::transform::Transform;
use core::wgpu;
use core::{
    pattern::BuiltPattern,
    units::LocalRect,
    PrepareContext
};
use core::shading::{RenderPipelineIndex, RenderPipelineKey};
use core::batching::{BatchFlags, BatchList};
use core::render_pass::{BuiltRenderPass, RenderCommandId, RenderPassConfig, RenderPassContext, RendererId};
use core::gpu::{GpuBufferAddress, StreamId};
use core::utils::DrawHelper;

use std::ops::Range;

use super::resources::Geometries;

pub const PATTERN_KIND_COLOR: u32 = 0;
pub const PATTERN_KIND_SIMPLE_LINEAR_GRADIENT: u32 = 1;

// The bits are shifted by 20 to pack with the gpu store handle.
core::bitflags::bitflags! {
    #[repr(transparent)]
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub struct Aa: u32 {
        const TOP =     1 << 20;
        const RIGHT =   2 << 20;
        const BOTTOM =  4 << 20;
        const LEFT =    8 << 20;
        const ALL =     (1|2|4|8) << 20;
        const NONE = 0;
    }
}

pub struct Batch {
    instances: Range<u32>,
    edge_aa: bool,
    // TODO: pattern, surface and opaque are only kept so that we can
    // create the pipeline index later in the prepare pass.
    pattern: BuiltPattern,
    opaque: bool,
    pipeline_idx: Option<RenderPipelineIndex>,
}

pub struct RectangleRenderer {
    batches: BatchList<Instance, Batch>,
    pipelines: crate::resources::Geometries,
    instances: Option<StreamId>,
}

impl RectangleRenderer {
    pub(crate) fn new(
        renderer_id: RendererId,
        pipelines: Geometries,
    ) -> Self {
        RectangleRenderer {
            batches: BatchList::new(renderer_id),
            pipelines,
            instances: None,
        }
    }

    pub fn begin_frame(&mut self) {
        self.batches.clear();
        self.instances = None;
    }

    pub fn supports_surface(&self, _surface: RenderPassConfig) -> bool {
        true
    }

    pub fn geometries(&self) -> &Geometries {
        &self.pipelines
    }

    pub fn fill_rect(
        &mut self,
        ctx: &mut RenderPassContext,
        transform: &Transform,
        local_rect: &LocalRect,
        mut aa: Aa,
        pattern: BuiltPattern,
        // TODO: get this from Transforms.
        transform_handle: GpuBufferAddress,
    ) {
        let z_index = ctx.z_indices.push();
        let aabb = transform
            .matrix()
            .outer_transformed_box(&local_rect.cast_unit());

        let instance_flags = InstanceFlags::from_bits(aa.bits()).unwrap();
        let pass_cfg = ctx.config;

        if pass_cfg.msaa {
            aa = Aa::NONE;
        }

        // Inner rect.
        // This one is pushed as an optimization to avoid rendering large opaque patterns
        // with blending enabled when we only require blending near the edges for anti-aliasing.
        // We push a rect to render in the opaque pass with AaCenter bit set so that the shader
        // deflate the rect to only the opaque portion. The depth test will prevent the subsequent
        // blended rect from evaluating and blending fragments in that area.
        if pattern.is_opaque
            && aa != Aa::NONE
            && pass_cfg.depth
            && aabb.width() > 50.0
            && aabb.height() > 50.0
        {
            self.batches.add(
                ctx,
                &pattern.batch_key(),
                &aabb,
                BatchFlags::ORDER_INDEPENDENT,
                &mut || Batch {
                    instances: 0..0,
                    pattern,
                    opaque: true,
                    edge_aa: true,
                    pipeline_idx: None,
                },
                &mut |mut batch, task| {
                    batch.push(Instance {
                        local_rect: *local_rect,
                        z_index,
                        pattern: pattern.data,
                        flags_transform: transform_handle.to_u32()
                            | (instance_flags | InstanceFlags::AaCenter).bits(),
                        render_task: task.handle.to_u32(),
                    });
                }
            );
        }

        // Main rect
        let use_opaque_pass =
            pattern.is_opaque && (aa == Aa::NONE || pass_cfg.msaa) && pass_cfg.depth;

        let batch_flags = if use_opaque_pass {
            BatchFlags::ORDER_INDEPENDENT
        } else {
            BatchFlags::empty()
        };

        self.batches.add(
            ctx,
            &pattern.batch_key(),
            &aabb,
            batch_flags,
            &mut || Batch {
                instances: 0..0,
                pattern,
                opaque: use_opaque_pass,
                edge_aa: false,
                pipeline_idx: None,
            },
            &mut |mut batch, task| {
                batch.batch_data().edge_aa |= aa != Aa::NONE;
                batch.push(Instance {
                    local_rect: *local_rect,
                    z_index,
                    pattern: pattern.data,
                    flags_transform: transform_handle.to_u32() | instance_flags.bits(),
                    render_task: task.handle.to_u32(),
                });
            }
        );
    }
}

impl core::Renderer for RectangleRenderer {
    fn prepare(&mut self, ctx: &mut PrepareContext, _passes: &[BuiltRenderPass]) {
        if self.batches.is_empty() {
            return;
        }

        let worker_data = &mut ctx.workers.data();
        let shaders = &mut worker_data.pipelines;
        let stream = worker_data.instances.next_stream_id();
        let mut instances = worker_data.instances.write(stream, 0);
        self.instances = Some(stream);

        for (items, surface, batch) in self.batches.iter_mut() {
            if batch.opaque && surface.depth {
                items.reverse();
            }
            const SIZE: u32 = std::mem::size_of::<Instance>() as u32;
            let start = instances.pushed_bytes() / SIZE;
            instances.push_slice(items);
            let end = instances.pushed_bytes() / SIZE;
            batch.instances = start..end;

            let pipeline = self.pipelines.get(batch.opaque, batch.edge_aa);
            let idx = shaders.prepare(RenderPipelineKey::new(
                pipeline,
                batch.pattern.shader,
                batch.pattern.blend_mode.with_alpha(!batch.opaque),
                surface.draw_config(true, None),
            ));
            batch.pipeline_idx = Some(idx);
        }
    }

    fn render<'pass, 'resources: 'pass, 'stats>(
        &self,
        commands: &[RenderCommandId],
        _surface_info: &RenderPassConfig,
        ctx: core::RenderContext<'resources, 'stats>,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        let common_resources = &ctx.resources.common;
        let Some(instance_buffer) = common_resources.instances.resolve_buffer_slice(self.instances)
            else { return; };

        let mut helper = DrawHelper::new();
        render_pass.set_index_buffer(
            common_resources.quad_ibo.slice(..),
            wgpu::IndexFormat::Uint16,
        );

        render_pass.set_vertex_buffer(0, instance_buffer);

        for batch_id in commands {
            let (_, _, batch) = self.batches.get(batch_id.index);
            let pipeline = ctx
                .render_pipelines
                .get(batch.pipeline_idx.unwrap())
                .unwrap();

            helper.resolve_and_bind(1, batch.pattern.bindings, ctx.bindings, render_pass);

            let query = ctx.gpu_profiler.begin_query("rectangle batch", render_pass);

            render_pass.set_pipeline(pipeline);
            render_pass.draw_indexed(0..6, 0, batch.instances.clone());
            ctx.stats.draw_calls += 1;

            ctx.gpu_profiler.end_query(render_pass, query);
        }
    }
}
