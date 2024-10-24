use core::{
    batching::{BatchFlags, BatchId, BatchList}, bytemuck, context::{
        BuiltRenderPass, DrawHelper, RenderPassContext, RendererId, SurfacePassConfig, ZIndex
    }, gpu::{shader::{BaseShaderId, BlendMode, PrepareRenderPipelines, RenderPipelineIndex, RenderPipelineKey}, DynBufferRange}, pattern::BuiltPattern, resources::GpuResources, shape::FilledPath, transform::{TransformId, Transforms}, units::{point, LocalRect, SurfaceIntRect, SurfaceRect}, wgpu, PrepareContext, UploadContext
};
use std::ops::Range;

use crate::{tiler::{EncodedPathInfo, Tiler, TilerOutput}, FillOptions, Occlusion, RendererOptions, TileGpuResources};

pub(crate) struct BatchInfo {
    pattern: BuiltPattern,
    opaque_draw: Option<Draw>,
    masked_draw: Option<Draw>,
    blend_mode: BlendMode,
}

struct Draw {
    tiles: Range<u32>,
    pipeline: RenderPipelineIndex,
}

enum Shape {
    Path(FilledPath),
    Surface,
}

impl Shape {
    pub fn aabb(&self) -> Option<LocalRect> {
        match self {
            // TODO: return the correct aabb for inverted shapes.
            Shape::Path(shape) => Some(*shape.path.aabb()),
            Shape::Surface => None,
        }
    }
}

pub(crate) struct Fill {
    shape: Shape,
    pattern: BuiltPattern,
    transform: TransformId,
    z_index: ZIndex,
}

pub struct TileRenderer {
    pub(crate) renderer_id: RendererId,
    pub tolerance: f32,
    pub(crate) occlusion: Occlusion,
    pub(crate) no_opaque_batches: bool,
    pub(crate) back_to_front: bool,
    pub(crate) tiler: Tiler,
    pub(crate) tiles: TilerOutput,

    pub(crate) batches: BatchList<Fill, BatchInfo>,
    pub(crate) base_shader: BaseShaderId,
    pub(crate) instances: Option<DynBufferRange>,

    pub(crate) resources: TileGpuResources,
}

impl TileRenderer {
    pub fn supports_surface(&self, surface: SurfacePassConfig) -> bool {
        if self.occlusion.gpu && !surface.depth {
            return false;
        }

        true
    }

    pub fn begin_frame(&mut self) {
        self.batches.clear();
        self.instances = None;
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

    pub fn fill_surface(&mut self, ctx: &mut RenderPassContext, transforms: &Transforms, pattern: BuiltPattern) {
        self.fill_shape(ctx, transforms, Shape::Surface, pattern);
    }

    fn fill_shape(&mut self, ctx: &mut RenderPassContext, transforms: &Transforms, shape: Shape, mut pattern: BuiltPattern) {
        let transform = transforms.current_id();
        let z_index = ctx.z_indices.push();

        let aabb = if let Some(aabb) = shape.aabb() {
            transforms
                .get_current()
                .matrix()
                .outer_transformed_box(&aabb)
        } else {
            use std::f32::{MIN, MAX};
            SurfaceRect { min: point(MIN, MIN), max: point(MAX, MAX) }
        };

        let batch_flags = if self.back_to_front || self.no_opaque_batches {
            BatchFlags::empty()
        } else {
            // Each shape has potententially a draw call for the masked tiles and
            // one for the opaque interriors, so letting them overlap would break
            // ordering.
            BatchFlags::NO_OVERLAP | BatchFlags::EARLIEST_CANDIDATE
        };

        if self.no_opaque_batches {
            pattern.is_opaque = false;
        }

        self.batches.find_or_add_batch(
            ctx,
            &pattern.batch_key(),
            &aabb,
            batch_flags,
            &mut || BatchInfo {
                pattern,
                opaque_draw: None,
                masked_draw: None,
                blend_mode: pattern.blend_mode,
            },
        ).push(Fill {
            shape,
            pattern,
            transform,
            z_index,
        });
    }

    pub fn set_options(&mut self, options: &RendererOptions) {
        self.tolerance = options.tolerance;
        self.occlusion = options.occlusion;
        self.no_opaque_batches = options.no_opaque_batches;
    }

    pub fn prepare(&mut self, pass: &BuiltRenderPass, transforms: &Transforms, shaders: &mut PrepareRenderPipelines) {
        if self.batches.is_empty() {
            return;
        }

        if self.occlusion.gpu {
            debug_assert!(pass.surface().depth);
        }

        let size = pass.surface_size();
        self.tiler.begin_target(SurfaceIntRect::from_size(size));
        if self.occlusion.cpu {
            self.tiles.occlusion.init(size.width as u32, size.height as u32);
        } else {
            self.tiles.occlusion.disable();
        }
        self.tiles.clear();

        let id = self.renderer_id;
        let mut batches = self.batches.take();
        if self.back_to_front {
            for batch_id in pass
                .batches()
                .iter()
                .rev()
                .filter(|batch| batch.renderer == id)
            {
                self.prepare_batch(*batch_id, transforms, shaders, &mut batches);
            }
        } else {
            for batch_id in pass
                .batches()
                .iter()
                .filter(|batch| batch.renderer == id)
            {
                self.prepare_batch(*batch_id, transforms, shaders, &mut batches);
            }
        }

        self.batches = batches;
    }

    fn prepare_batch(
        &mut self,
        batch_id: BatchId,
        transforms: &Transforms,
        shaders: &mut PrepareRenderPipelines,
        batches: &mut BatchList<Fill, BatchInfo>
    ) {
        let (commands, surface, info) = &mut batches.get_mut(batch_id.index);

        let opaque_tiles_start = self.tiles.opaque_tiles.len() as u32;
        let mask_tiles_start = self.tiles.mask_tiles.len() as u32;

        if self.back_to_front {
            for fill in commands.iter().rev() {
                self.prepare_fill(fill, transforms);
            }
        } else {
            for fill in commands.iter() {
                self.prepare_fill(fill, transforms);
            }
        }

        let opaque_tiles_end = self.tiles.opaque_tiles.len() as u32;
        let mask_tiles_end = self.tiles.mask_tiles.len() as u32;

        let draw_config = surface.draw_config(true, None);
        if opaque_tiles_end > opaque_tiles_start {
            info.opaque_draw = Some(Draw {
                tiles: opaque_tiles_start..opaque_tiles_end,
                pipeline: shaders.prepare(RenderPipelineKey::new(
                    self.base_shader,
                    info.pattern.shader,
                    BlendMode::None,
                    draw_config,
                )),
            })
        }

        if mask_tiles_end > mask_tiles_start {
            if self.back_to_front {
                self.tiles.mask_tiles[mask_tiles_start as usize .. mask_tiles_end as usize].reverse();
            }
            info.masked_draw = Some(Draw {
                tiles: mask_tiles_start..mask_tiles_end,
                pipeline: shaders.prepare(RenderPipelineKey::new(
                    self.base_shader,
                    info.pattern.shader,
                    info.blend_mode.with_alpha(true),
                    draw_config,
                )),
            })
        }
    }

    fn prepare_fill(&mut self, fill: &Fill, transforms: &Transforms) {
        let transform = transforms.get(fill.transform).matrix();

        match &fill.shape {
            Shape::Path(shape) => {
                let options = FillOptions::new()
                    .with_transform(Some(transform))
                    .with_fill_rule(shape.fill_rule)
                    .with_z_index(fill.z_index)
                    .with_tolerance(self.tolerance)
                    .with_inverted(shape.inverted);
                self.tiler.fill_path(
                    shape.path.iter(),
                    &options,
                    &fill.pattern,
                    &mut self.tiles,
                );
            }
            Shape::Surface => {
                let opacity = 1.0; // TODO
                self.tiler.fill_surface(&fill.pattern, opacity, fill.z_index, &mut self.tiles);
            }
        }
    }

    pub fn upload(
        &mut self,
        resources: &mut GpuResources,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        if self.tiles.is_empty() {
            return;
        }

        //println!("tiling2 stats:\n\tedges: {}({}kb)\n\t{} opaque tiles\n\t{} alpha tiles\n\t",
        //    self.tiles.edges.len(),
        //    (self.tiles.edges.len() * 4) as f32 / 1000.0,
        //    self.tiles.opaque_tiles.len(),
        //    self.tiles.mask_tiles.len(),
        //);

        self.instances = resources.common.vertices.upload_multiple(device,
            &[
                bytemuck::cast_slice(&self.tiles.mask_tiles),
                bytemuck::cast_slice(&self.tiles.opaque_tiles)
            ]
        );

        let edges_per_row = 1024;
        let rem = edges_per_row - self.tiles.edges.len() % edges_per_row;
        if rem != edges_per_row {
            self.tiles.edges.reserve(rem);
            for _ in 0..rem {
                self.tiles.edges.push(0);
            }
        }

        let rows = (self.tiles.edges.len() / edges_per_row) as u32;
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.resources.edge_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&self.tiles.edges),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(edges_per_row as u32 * 4),
                rows_per_image: Some(rows)
            },
            wgpu::Extent3d {
                width: edges_per_row as u32,
                height: rows,
                depth_or_array_layers: 1,
            },
        );


        let paths_per_row = 256; // 512 texels with 2 texels per path.
        let rem = paths_per_row - self.tiles.paths.len() % paths_per_row;
        if rem != paths_per_row {
            self.tiles.paths.reserve(rem);
            for _ in 0..rem {
                self.tiles.paths.push([0; 8]);
            }
        }

        let rows = (self.tiles.paths.len() / paths_per_row) as u32;
        let size_of_pathinfo = std::mem::size_of::<EncodedPathInfo>() as u32;
        let size_of_texel = 16;
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.resources.path_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&self.tiles.paths),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(paths_per_row as u32 * size_of_pathinfo),
                rows_per_image: Some(rows)
            },
            wgpu::Extent3d {
                width: paths_per_row as u32 * (size_of_pathinfo / size_of_texel),
                height: rows,
                depth_or_array_layers: 1,
            },
        );
    }
}

impl core::Renderer for TileRenderer {
    fn prepare(&mut self, ctx: &mut PrepareContext) {
        self.prepare(ctx.pass, ctx.transforms, ctx.pipelines);
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
        if self.instances.is_none() {
            return;
        }

        let common_resources = &ctx.resources.common;

        render_pass.set_index_buffer(
            common_resources.quad_ibo.slice(..),
            wgpu::IndexFormat::Uint16,
        );

        render_pass.set_vertex_buffer(
            0,
            common_resources
                .vertices
                .get_buffer_slice(self.instances.as_ref().unwrap()),
        );

        render_pass.set_bind_group(
            1,
            &self.resources.bind_group,
            &[],
        );

        let mut helper = DrawHelper::new();

        for batch_id in batches {
            let (_, _, batch) = self.batches.get(batch_id.index);
            helper.resolve_and_bind(2, batch.pattern.bindings, ctx.bindings, render_pass);

            if let Some(opaque) = &batch.opaque_draw {
                let pipeline = ctx.render_pipelines.get(opaque.pipeline).unwrap();
                let mut instances = opaque.tiles.clone();
                let offset = self.tiles.mask_tiles.len() as u32;
                instances.start += offset;
                instances.end += offset;

                //println!("draw opaque instances {instances:?}");
                render_pass.set_pipeline(pipeline);
                render_pass.draw_indexed(0..6, 0, instances);
            }
            if let Some(masked) = &batch.masked_draw {
                let pipeline = ctx.render_pipelines.get(masked.pipeline).unwrap();
                let instances = masked.tiles.clone();

                //println!("draw mask instances {instances:?}");
                render_pass.set_pipeline(pipeline);
                render_pass.draw_indexed(0..6, 0, instances);
            }
        }
    }
}

impl core::FillPath for TileRenderer {
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
