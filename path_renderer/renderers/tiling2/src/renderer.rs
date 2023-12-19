use core::{
    batching::{BatchFlags, BatchList},
    canvas::{
        CanvasRenderer, Context, DrawHelper, RenderContext, RenderPassState, RendererId, SubPass,
        SurfacePassConfig, ZIndex, FillPath,
    },
    pattern::BuiltPattern,
    resources::{CommonGpuResources, GpuResources, ResourcesHandle},
    shape::FilledPath,
    transform::TransformId,
    units::LocalRect,
    wgpu, gpu::{shader::{RenderPipelineIndex, PrepareRenderPipelines, RenderPipelineKey, GeneratedPipelineId}, DynBufferRange}, bytemuck,
};
use std::ops::Range;

use lyon::geom::Box2D;

use crate::{TileGpuResources, tiler::{Tiler, FillOptions, TilerOutput, PathInfo}};

struct BatchInfo {
    surface: SurfacePassConfig,
    pattern: BuiltPattern,
    opaque_draw: Option<Draw>,
    masked_draw: Option<Draw>,
}

struct Draw {
    tiles: Range<u32>,
    pipeline: RenderPipelineIndex,
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

pub struct TileRenderer {
    renderer_id: RendererId,
    common_resources: ResourcesHandle<CommonGpuResources>,
    resources: ResourcesHandle<TileGpuResources>,
    tolerance: f32,
    tiler: Tiler,
    tiles: TilerOutput,

    batches: BatchList<Fill, BatchInfo>,
    opaque_pipeline: GeneratedPipelineId,
    masked_pipeline: GeneratedPipelineId,
    instances: Option<DynBufferRange>,
}

impl TileRenderer {
    pub fn new(
        renderer_id: RendererId,
        common_resources: ResourcesHandle<CommonGpuResources>,
        resources: ResourcesHandle<TileGpuResources>,
        res: &TileGpuResources,
    ) -> Self {
        TileRenderer {
            renderer_id,
            common_resources,
            resources: resources,
            tolerance: 0.25,
            tiler: Tiler::new(),
            tiles: TilerOutput {
                paths: Vec::new(),
                edges: Vec::new(),
                mask_tiles: Vec::new(),
                opaque_tiles: Vec::new(),
            },

            batches: BatchList::new(renderer_id),
            opaque_pipeline: res.opaque_pipeline,
            masked_pipeline: res.masked_pipeline,
            instances: None,
        }
    }

    pub fn supports_surface(&self, _surface: SurfacePassConfig) -> bool {
        true
    }

    pub fn begin_frame(&mut self, ctx: &Context) {
        self.batches.clear();
        self.tiler.begin_frame(Box2D::from_size(ctx.surface.size().cast_unit()).cast());
        self.tiles.clear();
        self.instances = None;
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

        let batch_key = pattern.batch_key();
        let (commands, info) = self.batches.find_or_add_batch(
            &mut ctx.batcher,
            &batch_key,
            &aabb,
            BatchFlags::empty(),
            &mut || BatchInfo {
                surface: ctx.surface.current_config(),
                pattern,
                opaque_draw: None,
                masked_draw: None,
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
            .rev()
            .filter(|batch| batch.renderer == id)
        {
            let (commands, info) = &mut batches.get_mut(batch_id.index);

            let surface = info.surface;

            let opaque_tiles_start = self.tiles.opaque_tiles.len() as u32;
            let mask_tiles_start = self.tiles.mask_tiles.len() as u32;

            for fill in commands.iter().rev() {
                self.prepare_fill(fill, ctx);
            }

            let opaque_tiles_end = self.tiles.opaque_tiles.len() as u32;
            let mask_tiles_end = self.tiles.mask_tiles.len() as u32;

            let draw_config = surface.draw_config(true, None);
            if opaque_tiles_end > opaque_tiles_start {
                info.opaque_draw = Some(Draw {
                    tiles: opaque_tiles_start..opaque_tiles_end,
                    pipeline: shaders.prepare(RenderPipelineKey::new(
                        self.opaque_pipeline,
                        info.pattern.shader,
                        draw_config,
                    )),
                })
            }

            if mask_tiles_end > mask_tiles_start {
                self.tiles.mask_tiles[mask_tiles_start as usize .. mask_tiles_end as usize].reverse();
                info.masked_draw = Some(Draw {
                    tiles: mask_tiles_start..mask_tiles_end,
                    pipeline: shaders.prepare(RenderPipelineKey::new(
                        self.masked_pipeline,
                        info.pattern.shader,
                        draw_config,
                    )),
                })
            }
        }

        self.batches = batches;
    }

    fn prepare_fill(&mut self, fill: &Fill, ctx: &Context) {
        let transform = ctx.transforms.get(fill.transform).matrix();

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
        }
    }

    pub fn upload(
        &mut self,
        resources: &mut GpuResources,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        if self.tiles.edges.is_empty() {
            return;
        }

        println!("tiling2 stats:\n\tedges: {}({}kb)\n\t{} opaque tiles\n\t{} alpha tiles\n\t",
            self.tiles.edges.len(),
            (self.tiles.edges.len() * 4) as f32 / 1000.0,
            self.tiles.opaque_tiles.len(),
            self.tiles.mask_tiles.len(),
        );

        let res = &mut resources[self.common_resources];
        self.instances = res.vertices.upload_multiple(device,
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

        let res = &mut resources[self.resources];

        let rows = (self.tiles.edges.len() / edges_per_row) as u32;
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &res.edge_texture,
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


        let paths_per_row = 512;
        let rem = paths_per_row - self.tiles.paths.len() % paths_per_row;
        if rem != paths_per_row {
            self.tiles.paths.reserve(rem);
            for _ in 0..rem {
                self.tiles.paths.push([0, 0, 0, 0]);
            }
        }

        let rows = (self.tiles.paths.len() / paths_per_row) as u32;
        let size_of_pathinfo = std::mem::size_of::<PathInfo>() as u32;
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &res.path_texture,
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
                width: paths_per_row as u32,
                height: rows,
                depth_or_array_layers: 1,
            },
        );
    }
}

impl CanvasRenderer for TileRenderer {
    fn render<'pass, 'resources: 'pass>(
        &self,
        sub_passes: &[SubPass],
        _surface_info: &RenderPassState,
        ctx: RenderContext<'resources>,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        if self.instances.is_none() {
            return;
        }

        let common_resources = &ctx.resources[self.common_resources];
        let resources = &ctx.resources[self.resources];

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
            0,
            &common_resources.main_target_and_gpu_store_bind_group,
            &[],
        );

        render_pass.set_bind_group(
            1,
            &resources.bind_group,
            &[],
        );

        let mut helper = DrawHelper::new();

        for sub_pass in sub_passes {
            let (_, batch) = self.batches.get(sub_pass.internal_index);
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

impl FillPath for TileRenderer {
    fn fill_path(
        &mut self,
        ctx: &mut Context,
        path: FilledPath,
        pattern: BuiltPattern,
    ) {
        self.fill_shape(ctx, Shape::Path(path), pattern);
    }
}
