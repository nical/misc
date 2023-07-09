use core::{
    canvas::{RendererId, Canvas, CanvasRenderer, RenderPassState, DrawHelper, SurfaceState, SubPass},
    resources::{ResourcesHandle, GpuResources, CommonGpuResources},
    gpu::{
        DynBufferRange, Shaders, GpuStoreHandle
    },
    pattern::{BuiltPattern},
    bytemuck,
    wgpu, BindingResolver, units::LocalRect, batching::{BatchList, BatchFlags},
};
use crate::{resources::Instance, InstanceFlags};

use super::RectangleGpuResources;

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
    pattern: BuiltPattern,
    surface: SurfaceState,
    vbo_range: Option<DynBufferRange>,
    opaque: bool,
    edge_aa: bool,
}

pub struct RectangleRenderer {
    common_resources: ResourcesHandle<CommonGpuResources>,
    resources: ResourcesHandle<RectangleGpuResources>,
    batches: BatchList<Instance, Batch>,
}

impl RectangleRenderer {
    pub fn new(renderer_id: RendererId, common_resources: ResourcesHandle<CommonGpuResources>, resources: ResourcesHandle<RectangleGpuResources>) -> Self {
        RectangleRenderer {
            common_resources,
            resources,

            batches: BatchList::new(renderer_id),
        }
    }

    pub fn begin_frame(&mut self, _canvas: &Canvas) {
        self.batches.clear();
    }

    pub fn supports_surface(&self, _surface: SurfaceState) -> bool {
        true
    }

    pub fn fill_rect(
        &mut self,
        canvas: &mut Canvas,
        local_rect: &LocalRect,
        mut aa: Aa,
        pattern: BuiltPattern,
        transform_handle: GpuStoreHandle,
    ) {
        let z_index = canvas.z_indices.push();
        let aabb = canvas.transforms.get_current().matrix().outer_transformed_box(&local_rect.cast_unit());

        let instance_flags = InstanceFlags::from_bits(aa.bits()).unwrap();
        let surface = canvas.surface.current_state();

        if surface.msaa {
            aa = Aa::NONE;
        }

        // Inner rect.
        // This one is pushed as an optimization to avoid rendering large opaque patterns
        // with blending enabled when we only require blendin near the edges for anti-aliasing.
        // We push a rect to render in the opaque pass with AaCenter bit set so that the shader
        // deflate the rect to only the opaque portion. The depth test will prevent the subsequent
        // blended rect from evaluating and blending fragments in that area.
        if pattern.is_opaque
            && aa != Aa::NONE
            && surface.depth
            && aabb.width() > 50.0
            && aabb.height() > 50.0 {
            let (commands, _) = self.batches.find_or_add_batch(
                &mut canvas.batcher,
                &pattern.batch_key(),
                &aabb,
                BatchFlags::ORDER_INDEPENDENT,
                &mut|| Batch {
                    pattern,
                    surface,
                    opaque: true,
                    edge_aa: true,
                    vbo_range: None,
                },
            );

            commands.push(Instance {
                local_rect: *local_rect,
                z_index,
                pattern: pattern.data,
                flags_transform: transform_handle.to_u32()
                | (instance_flags | InstanceFlags::AaCenter).bits(),
                mask: 0,
            });
        }

        // Main rect
        let use_opaque_pass = pattern.is_opaque
            && (aa == Aa::NONE || surface.msaa)
            && surface.depth;

        let batch_flags = if use_opaque_pass {
            BatchFlags::ORDER_INDEPENDENT
        } else {
            BatchFlags::empty()
        };

        let (commands, batch) = self.batches.find_or_add_batch(
            &mut canvas.batcher,
            &pattern.batch_key(),
            &aabb,
            batch_flags,
            &mut|| Batch {
                pattern,
                surface,
                opaque: use_opaque_pass,
                edge_aa: false,
                vbo_range: None,
            },
        );

        batch.edge_aa |= aa != Aa::NONE;

        commands.push(Instance {
            local_rect: *local_rect,
            z_index,
            pattern: pattern.data,
            flags_transform: transform_handle.to_u32() | instance_flags.bits(),
            mask: 0,
        });
    }

    pub fn prepare(&mut self, _canvas: &Canvas) {
        if self.batches.is_empty() {
            return;
        }
    }

    pub fn upload(&mut self,
        resources: &mut GpuResources,
        shaders: &mut Shaders,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        let res = &resources[self.resources];
        let pipelines = res.pipelines.clone();
        let common = &mut resources[self.common_resources];
        for (items, batch) in self.batches.iter_mut() {
            batch.vbo_range = common.vertices.upload(device, bytemuck::cast_slice(&items[..]));

            let pipeline = pipelines.get(batch.opaque, batch.edge_aa);
            shaders.prepare_pipeline(device, pipeline, batch.pattern.shader, batch.surface.surface_config(true, None));
        }
    }
}

impl CanvasRenderer for RectangleRenderer {
    fn render<'pass, 'resources: 'pass>(
        &self,
        sub_passes: &[SubPass],
        surface_info: &RenderPassState,
        shaders: &'resources Shaders,
        resources: &'resources GpuResources,
        bindings: &'resources dyn BindingResolver,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        let common_resources = &resources[self.common_resources];
        let rect_resources = &resources[self.resources];

        let surface = surface_info.surface_config(true, None);
        let mut helper = DrawHelper::new();
        render_pass.set_index_buffer(common_resources.quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.set_bind_group(0, &common_resources.main_target_and_gpu_store_bind_group, &[]);

        for sub_pass in sub_passes {
            let (instances, batch) = self.batches.get(sub_pass.internal_index);

            let pipleine_id = rect_resources.pipelines.get(batch.opaque, batch.edge_aa);
            let pipeline = shaders.try_get(pipleine_id, batch.pattern.shader, surface).unwrap();

            helper.resolve_and_bind(1, batch.pattern.bindings, bindings, render_pass);

            render_pass.set_pipeline(pipeline);
            render_pass.set_vertex_buffer(0, common_resources.vertices.get_buffer_slice(batch.vbo_range.as_ref().unwrap()));
            render_pass.draw_indexed(0..6, 0, 0..(instances.len() as u32));
        }
    }
}

