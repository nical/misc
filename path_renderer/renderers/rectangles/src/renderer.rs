use crate::{resources::Instance, InstanceFlags};
use core::{
    batching::{BatchFlags, BatchId, BatchList}, bytemuck, context::{
        BuiltRenderPass, DrawHelper, RenderPassContext, RendererId, SurfacePassConfig
    }, gpu::{
        shader::{PrepareRenderPipelines, RenderPipelineIndex, RenderPipelineKey},
        DynBufferRange, GpuStoreHandle,
    }, pattern::BuiltPattern, resources::GpuResources, transform::Transforms, units::LocalRect, wgpu, PrepareContext, UploadContext
};

use super::resources::Pipelines;

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
    vbo_range: Option<DynBufferRange>,
    edge_aa: bool,
    // TODO: pattern, surface and opaque are only kept so that we can
    // create the pipeline index later in the prepare pass.
    pattern: BuiltPattern,
    opaque: bool,
    pipeline_idx: Option<RenderPipelineIndex>,
}

pub struct RectangleRenderer {
    batches: BatchList<Instance, Batch>,
    pipelines: crate::resources::Pipelines,
}

impl RectangleRenderer {
    pub(crate) fn new(
        renderer_id: RendererId,
        pipelines: Pipelines,
    ) -> Self {
        RectangleRenderer {
            batches: BatchList::new(renderer_id),
            pipelines,
        }
    }

    pub fn begin_frame(&mut self) {
        self.batches.clear();
    }

    pub fn supports_surface(&self, _surface: SurfacePassConfig) -> bool {
        true
    }

    pub fn fill_rect(
        &mut self,
        ctx: &mut RenderPassContext,
        transforms: &Transforms,
        local_rect: &LocalRect,
        mut aa: Aa,
        pattern: BuiltPattern,
        // TODO: get this from Transforms.
        transform_handle: GpuStoreHandle,
    ) {
        let z_index = ctx.z_indices.push();
        let aabb = transforms
            .get_current()
            .matrix()
            .outer_transformed_box(&local_rect.cast_unit());

        let instance_flags = InstanceFlags::from_bits(aa.bits()).unwrap();
        let surface = ctx.surface;

        if surface.msaa {
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
            && surface.depth
            && aabb.width() > 50.0
            && aabb.height() > 50.0
        {
            self.batches.find_or_add_batch(
                ctx,
                &pattern.batch_key(),
                &aabb,
                BatchFlags::ORDER_INDEPENDENT,
                &mut || Batch {
                    pattern,
                    opaque: true,
                    edge_aa: true,
                    vbo_range: None,
                    pipeline_idx: None,
                },
            ).push(Instance {
                local_rect: *local_rect,
                z_index,
                pattern: pattern.data,
                flags_transform: transform_handle.to_u32()
                    | (instance_flags | InstanceFlags::AaCenter).bits(),
                mask: 0,
            });
        }

        // Main rect
        let use_opaque_pass =
            pattern.is_opaque && (aa == Aa::NONE || surface.msaa) && surface.depth;

        let batch_flags = if use_opaque_pass {
            BatchFlags::ORDER_INDEPENDENT
        } else {
            BatchFlags::empty()
        };

        let mut batch = self.batches.find_or_add_batch(
            ctx,
            &pattern.batch_key(),
            &aabb,
            batch_flags,
            &mut || Batch {
                pattern,
                opaque: use_opaque_pass,
                edge_aa: false,
                vbo_range: None,
                pipeline_idx: None,
            },
        );

        batch.batch_data().edge_aa |= aa != Aa::NONE;
        batch.push(Instance {
            local_rect: *local_rect,
            z_index,
            pattern: pattern.data,
            flags_transform: transform_handle.to_u32() | instance_flags.bits(),
            mask: 0,
        });
    }

    pub fn prepare(&mut self, _pass: &BuiltRenderPass, shaders: &mut PrepareRenderPipelines) {
        for (_, surface, batch) in self.batches.iter_mut() {
            let pipeline = self.pipelines.get(batch.opaque, batch.edge_aa);
            let idx = shaders.prepare(RenderPipelineKey::new(
                pipeline,
                batch.pattern.shader,
                batch.pattern.blend_mode.with_alpha(batch.edge_aa),
                surface.draw_config(true, None),
            ));
            batch.pipeline_idx = Some(idx);
        }
    }

    pub fn upload(
        &mut self,
        resources: &mut GpuResources,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        for (items, _, batch) in self.batches.iter_mut() {
            batch.vbo_range = resources.common
                .vertices
                .upload(device, bytemuck::cast_slice(&items[..]));
        }
    }
}

impl core::Renderer for RectangleRenderer {
    fn prepare(&mut self, ctx: &mut PrepareContext) {
        self.prepare(ctx.pass, ctx.pipelines);
    }

    fn upload(&mut self, ctx: &mut UploadContext) {
        self.upload(ctx.resources, ctx.wgpu.device, ctx.wgpu.queue);
    }

    fn render<'pass, 'resources: 'pass>(
        &self,
        batches: &[BatchId],
        _surface_info: &SurfacePassConfig,
        ctx: core::RenderContext<'resources>,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        let common_resources = &ctx.resources.common;

        let mut helper = DrawHelper::new();
        render_pass.set_index_buffer(
            common_resources.quad_ibo.slice(..),
            wgpu::IndexFormat::Uint16,
        );

        for batch_id in batches {
            let (instances, _, batch) = self.batches.get(batch_id.index);

            let pipeline = ctx
                .render_pipelines
                .get(batch.pipeline_idx.unwrap())
                .unwrap();

            helper.resolve_and_bind(1, batch.pattern.bindings, ctx.bindings, render_pass);

            render_pass.set_pipeline(pipeline);
            render_pass.set_vertex_buffer(
                0,
                common_resources
                    .vertices
                    .get_buffer_slice(batch.vbo_range.as_ref().unwrap()),
            );
            render_pass.draw_indexed(0..6, 0, 0..(instances.len() as u32));
        }
    }
}
