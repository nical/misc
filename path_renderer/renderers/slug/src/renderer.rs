//! Slug renderer implementation: path decomposition, batching, and Renderer trait.

use lyon::geom::{CubicBezierSegment, QuadraticBezierSegment};
use lyon::path::PathEvent;

use core::bytemuck;
use core::gpu::{StreamId, UploadStats};
use core::pattern::BuiltPattern;
use core::batching::{BatchFlags, BatchList};
use core::render_pass::{
    BuiltRenderPass, RenderCommandId, RenderPassConfig, RenderPassContext, RendererId,
    ZIndex,
};
use core::shading::{
    BindGroupLayoutId, GeometryId, RenderPipelineIndex, RenderPipelineKey,
};
use core::shape::FilledPath;
use core::transform::{Transform, TransformId};
use core::units::LocalRect;
use core::{wgpu, PrepareContext, UploadContext};
use core::utils::DrawHelper;
use std::ops::Range;


#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ShapeInstance {
    pub local_rect: [f32; 4],
    pub band_params: [f32; 4],
    pub band_loc: u32,
    pub band_max: u32,
    pub z_index: u32,
    pub pattern: u32,
    pub render_task: u32,
    pub flags_transform: u32,
}

unsafe impl bytemuck::Zeroable for ShapeInstance {}
unsafe impl bytemuck::Pod for ShapeInstance {}

struct Fill {
    path: FilledPath,
    pattern: BuiltPattern,
    transform: TransformId,
    z_index: ZIndex,
    render_task: u32,
}

struct BatchInfo {
    instances: Range<u32>,
    pattern: BuiltPattern,
    pipeline_idx: Option<RenderPipelineIndex>,
}

struct GpuResources {
    curve_tex: wgpu::Texture,
    band_tex: wgpu::Texture,
    bind_group: wgpu::BindGroup,
}

pub struct SlugRenderer {
    renderer_id: RendererId,
    geometry: GeometryId,
    bind_group_layout: BindGroupLayoutId,

    batches: BatchList<Fill, BatchInfo>,

    // CPU-side data rebuilt each frame.
    builder: ShapeBuilder,
    instances: Vec<ShapeInstance>,

    resources: Option<GpuResources>,
    instances_stream: Option<StreamId>,
    resources_dirty: bool,
}

impl SlugRenderer {
    pub(crate) fn new(
        renderer_id: RendererId,
        geometry: GeometryId,
        bind_group_layout: BindGroupLayoutId,
    ) -> Self {
        SlugRenderer {
            renderer_id,
            geometry,
            bind_group_layout,
            batches: BatchList::new(renderer_id),
            builder: ShapeBuilder::new(),
            instances: Vec::new(),
            resources: None,
            instances_stream: None,
            resources_dirty: false,
        }
    }

    pub fn begin_frame(&mut self) {
        self.batches.clear();
        self.builder.clear();
        self.instances.clear();
        self.instances_stream = None;
        self.resources_dirty = false;
    }

    pub fn fill_path(
        &mut self,
        ctx: &mut RenderPassContext,
        transform: &Transform,
        path: impl Into<FilledPath>,
        pattern: BuiltPattern,
    ) {
        let path = path.into();
        let z_index = ctx.z_indices.push();
        let transform_id = transform.id();

        let aabb = transform
            .matrix()
            .outer_transformed_box(&path.path.aabb());

        self.batches.add(
            ctx,
            &pattern.batch_key(),
            &aabb,
            BatchFlags::empty(),
            &mut || BatchInfo {
                instances: 0..0,
                pattern,
                pipeline_idx: None,
            },
            &mut |mut batch, task| {
                batch.push(Fill {
                    path: path.clone(),
                    transform: transform_id,
                    z_index,
                    pattern,
                    render_task: task.gpu_address.to_u32(),
                });
            },
        );
    }
}

impl core::Renderer for SlugRenderer {
    fn name(&self) -> &'static str { "slug" }

    fn prepare(&mut self, ctx: &mut PrepareContext, _passes: &[BuiltRenderPass]) {
        if self.batches.is_empty() {
            return;
        }

        let worker_data = &mut ctx.workers.data();
        let shaders = &mut worker_data.pipelines;
        let mut f32_buffer = worker_data.f32_buffer.write();
        let stream = worker_data.instances.next_stream_id();
        let mut gpu_instances = worker_data.instances.write(stream, 0);
        self.instances_stream = Some(stream);

        let mut batches = self.batches.take();
        for (items, surface, batch_info) in batches.iter_mut() {
            let start = gpu_instances.pushed_bytes()
                / std::mem::size_of::<ShapeInstance>() as u32;

            for fill in items.iter() {
                let transform = ctx.transforms.get(fill.transform);
                let transform_handle = transform.request_gpu_handle(&mut f32_buffer);
                let aabb = *fill.path.path.aabb();

                let shape = match self.builder.add_path(fill.path.path.iter(), &aabb) {
                    Some(s) => s,
                    None => continue,
                };

                let instance = ShapeInstance {
                    local_rect: [aabb.min.x, aabb.min.y, aabb.max.x, aabb.max.y],
                    band_params: [
                        shape.band_scale_x,
                        shape.band_scale_y,
                        shape.band_offset_x,
                        shape.band_offset_y,
                    ],
                    band_loc: (shape.band_tex_x as u32 & 0xFFFF)
                        | ((shape.band_tex_y as u32 & 0xFFFF) << 16),
                    band_max: (shape.band_max_x as u32 & 0xFFFF)
                        | ((shape.band_max_y as u32 & 0xFFFF) << 16),
                    z_index: fill.z_index,
                    pattern: fill.pattern.data,
                    render_task: fill.render_task,
                    flags_transform: transform_handle.to_u32(),
                };
                gpu_instances.push(instance);
            }

            let end = gpu_instances.pushed_bytes()
                / std::mem::size_of::<ShapeInstance>() as u32;
            batch_info.instances = start..end;

            let idx = shaders.prepare(RenderPipelineKey::new(
                self.geometry,
                batch_info.pattern.shader,
                batch_info.pattern.blend_mode.with_alpha(true),
                surface.draw_config(true, None),
            ));
            batch_info.pipeline_idx = Some(idx);
        }
        self.batches = batches;
        self.resources_dirty = true;
    }

    fn upload(&mut self, ctx: &mut UploadContext) -> UploadStats {
        if !self.resources_dirty {
            return UploadStats::default();
        }

        let device = ctx.wgpu.device;
        let queue = ctx.wgpu.queue;

        if self.resources.is_none() {
            self.resources = Some(create_gpu_resources(
                device, ctx.shaders, self.bind_group_layout,
            ));
        }

        let gpu = self.resources.as_ref().unwrap();
        write_gpu_resources(queue, gpu, &self.builder);

        UploadStats::default()
    }

    fn render<'pass, 'resources: 'pass, 'tmp>(
        &self,
        commands: &[RenderCommandId],
        _surface_info: &RenderPassConfig,
        ctx: core::RenderContext<'resources, 'tmp>,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        let common = &ctx.resources.common;
        let Some(instance_buffer) = common.instances.resolve_buffer_slice(self.instances_stream)
            else { return; };
        let Some(resources) = &self.resources else { return; };

        let mut helper = DrawHelper::new();
        render_pass.set_index_buffer(
            common.quad_ibo.slice(..),
            wgpu::IndexFormat::Uint16,
        );
        render_pass.set_vertex_buffer(0, instance_buffer);

        render_pass.set_bind_group(1, &resources.bind_group, &[]);

        for batch_id in commands {
            let (_, _, batch_info) = self.batches.get(batch_id.index);
            if batch_info.instances.is_empty() {
                continue;
            }

            let query = ctx.gpu_profiler.begin_query("slug batch", render_pass);

            let pipeline = ctx.render_pipelines
                .get(batch_info.pipeline_idx.unwrap())
                .unwrap();

            helper.resolve_and_bind(2, batch_info.pattern.bindings, ctx.bindings, render_pass);

            render_pass.set_pipeline(pipeline);
            render_pass.draw_indexed(0..6, 0, batch_info.instances.clone());
            ctx.stats.draw_calls += 1;

            ctx.gpu_profiler.end_query(render_pass, query);
        }
    }
}

fn create_gpu_resources(
    device: &wgpu::Device,
    shaders: &core::shading::Shaders,
    bind_group_layout_id: BindGroupLayoutId,
) -> GpuResources {
    let curve_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("slug::curves"),
        size: wgpu::Extent3d {
            width: TEX_WIDTH as u32,
            height: TEX_HEIGHT as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let band_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("slug::bands"),
        size: wgpu::Extent3d {
            width: TEX_WIDTH as u32,
            height: TEX_HEIGHT as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Uint,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let curve_view = curve_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let band_view = band_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let bgl = shaders.get_bind_group_layout(bind_group_layout_id);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("slug"),
        layout: &bgl.handle,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&curve_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&band_view),
            },
        ],
    });

    GpuResources {
        curve_tex,
        band_tex,
        bind_group,
    }
}

fn write_gpu_resources(queue: &wgpu::Queue, gpu: &GpuResources, atlas: &ShapeBuilder) {
    let row_bytes = TEX_WIDTH * 4 * 4; // 4 components * 4 bytes per component

    let curve_rows = atlas.curve_rows_used();
    if curve_rows > 0 {
        let data_len = curve_rows as usize * TEX_WIDTH * 4;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &gpu.curve_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&atlas.curve_tex[..data_len]),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes as u32),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: TEX_WIDTH as u32,
                height: curve_rows,
                depth_or_array_layers: 1,
            },
        );
    }

    let band_rows = atlas.band_rows_used();
    if band_rows > 0 {
        let data_len = band_rows as usize * TEX_WIDTH * 4;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &gpu.band_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&atlas.band_tex[..data_len]),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes as u32),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: TEX_WIDTH as u32,
                height: band_rows,
                depth_or_array_layers: 1,
            },
        );
    }
}

pub const TEX_WIDTH: usize = 4096;
// TODO: adaptatively size the data textures.
pub const TEX_HEIGHT: usize = 512;

/// Per-shape metadata produced during atlas building.
#[derive(Clone, Copy, Default, Debug)]
pub struct SlugShape {
    pub local_aabb: LocalRect,
    pub band_tex_x: i32,
    pub band_tex_y: i32,
    pub band_max_x: i32,
    pub band_max_y: i32,
    pub band_scale_x: f32,
    pub band_scale_y: f32,
    pub band_offset_x: f32,
    pub band_offset_y: f32,
}

/// CPU-side atlas backing the curve and band GPU textures.
///
/// - `curve_tex`: RGBA32Float texture storing control points (2 texels per curve).
/// - `band_tex`: RGBA32Uint texture storing band headers and curve index lists.
pub struct ShapeBuilder {
    pub curve_tex: Vec<f32>,
    pub band_tex: Vec<u32>,
    pub curve_cx: i32,
    pub curve_cy: i32,
    pub band_cx: i32,
    pub band_cy: i32,
    /// Reusable scratch buffer for path decomposition, avoids per-shape allocation.
    curves: Vec<QuadraticBezierSegment<f32>>,
}

impl ShapeBuilder {
    pub fn new() -> Self {
        Self {
            curve_tex: vec![0.0f32; TEX_HEIGHT * TEX_WIDTH * 4],
            band_tex: vec![0u32; TEX_HEIGHT * TEX_WIDTH * 4],
            curve_cx: 0,
            curve_cy: 0,
            band_cx: 0,
            band_cy: 0,
            curves: Vec::new(),
        }
    }

    /// Number of rows used in the curve texture (0 if empty).
    pub fn curve_rows_used(&self) -> u32 {
        if self.curve_cx == 0 && self.curve_cy == 0 {
            0
        } else {
            self.curve_cy as u32 + 1
        }
    }

    /// Number of rows used in the band texture (0 if empty).
    pub fn band_rows_used(&self) -> u32 {
        if self.band_cx == 0 && self.band_cy == 0 {
            0
        } else {
            self.band_cy as u32 + 1
        }
    }

    pub fn clear(&mut self) {
        //self.curve_tex.fill(0.0);
        //self.band_tex.fill(0);
        self.curve_cx = 0;
        self.curve_cy = 0;
        self.band_cx = 0;
        self.band_cy = 0;
    }

    fn band_advance(&mut self, n: i32) {
        self.band_cx += n;
        while self.band_cx >= TEX_WIDTH as i32 {
            self.band_cx -= TEX_WIDTH as i32;
            self.band_cy += 1;
        }
    }

    fn write_curve(&mut self, curve: &QuadraticBezierSegment<f32>) -> Option<i32> {
        if self.curve_cy as usize >= TEX_HEIGHT {
            return None;
        }

        if self.curve_cx + 2 > TEX_WIDTH as i32 {
            self.curve_cx = 0;
            self.curve_cy += 1;
        }

        let x = self.curve_cx;
        let y = self.curve_cy;

        // First texel: from.xy, ctrl.xy
        let base_addr = tex_idx(y, x);
        self.curve_tex[base_addr] = curve.from.x;
        self.curve_tex[base_addr + 1] = curve.from.y;
        self.curve_tex[base_addr + 2] = curve.ctrl.x;
        self.curve_tex[base_addr + 3] = curve.ctrl.y;

        // Second texel: to.xy, 0, 0
        self.curve_tex[base_addr + 4] = curve.to.x;
        self.curve_tex[base_addr + 5] = curve.to.y;
        self.curve_tex[base_addr + 6] = 0.0;
        self.curve_tex[base_addr + 7] = 0.0;

        self.curve_cx += 2;
        if self.curve_cx >= TEX_WIDTH as i32 {
            self.curve_cx = 0;
            self.curve_cy += 1;
        }

        Some(x | (y << 16))
    }
}

fn tex_idx(y: i32, x: i32) -> usize {
    (y as usize * TEX_WIDTH + x as usize) * 4
}

struct BandEntry {
    loc: i32,
    sort_key: f32,
}

/// Build horizontal and vertical band lookup structures for a shape's curves.
/// Reads curves from `builder.curves` (populated by `add_path`).
fn build_bands(
    builder: &mut ShapeBuilder,
    shape: &mut SlugShape,
    nband: i32,
) -> Result<(), ()> {
    let num_curves = builder.curves.len();
    if num_curves == 0 {
        return Ok(());
    }

    let nbx = nband;
    let nby = nband;
    shape.band_max_x = nbx - 1;
    shape.band_max_y = nby - 1;

    let w = shape.local_aabb.width();
    let h = shape.local_aabb.height();
    let inv_w = if w > 0.0 { nbx as f32 / w } else { 1.0 };
    let inv_h = if h > 0.0 { nby as f32 / h } else { 1.0 };
    shape.band_scale_x = inv_w;
    shape.band_scale_y = inv_h;
    shape.band_offset_x = -shape.local_aabb.min.x * inv_w;
    shape.band_offset_y = -shape.local_aabb.min.y * inv_h;

    // Write all curves to the curve texture and record their locations.
    let mut curve_locs = Vec::with_capacity(num_curves);
    for i in 0..num_curves {
        let curve = builder.curves[i];
        match builder.write_curve(&curve) {
            Some(loc) => curve_locs.push(loc),
            None => return Err(()),
        }
    }

    shape.band_tex_x = builder.band_cx;
    shape.band_tex_y = builder.band_cy;

    let hstart_x = builder.band_cx;
    let hstart_y = builder.band_cy;
    // Reserve space for band headers (nby horizontal + nbx vertical).
    builder.band_advance(nby + nbx);

    let mut tmp = Vec::with_capacity(num_curves);

    // axis 0 = horizontal bands (y-axis slicing), axis 1 = vertical bands (x-axis slicing).
    for axis in 0..2 {
        let nb = if axis == 0 { nby } else { nbx };
        let step = if axis == 0 {
            h / nby as f32
        } else {
            w / nbx as f32
        };
        let min = if axis == 0 {
            shape.local_aabb.min.y
        } else {
            shape.local_aabb.min.x
        };

        for b in 0..nb {
            let b0 = b as f32;
            let b1 = (b + 1) as f32;
            let band_lo = min + b0 * step;
            let band_hi = min + b1 * step;

            tmp.clear();
            for (i, c) in builder.curves.iter().enumerate() {
                let (v0, v1, v2, sort_v) = if axis == 0 {
                    (
                        c.from.y, c.ctrl.y, c.to.y,
                        c.from.x.max(c.ctrl.x).max(c.to.x),
                    )
                } else {
                    (
                        c.from.x, c.ctrl.x, c.to.x,
                        c.from.y.max(c.ctrl.y).max(c.to.y),
                    )
                };
                let lo = v0.min(v1).min(v2);
                let hi = v0.max(v1).max(v2);
                if hi >= band_lo && lo <= band_hi {
                    tmp.push(BandEntry {
                        loc: curve_locs[i],
                        sort_key: sort_v,
                    });
                }
            }
            // Sort descending by sort_key for early-out in the shader.
            tmp.sort_by(|a, b| {
                b.sort_key
                    .partial_cmp(&a.sort_key)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let cur_lin = builder.band_cy as i32 * TEX_WIDTH as i32 + builder.band_cx;
            let base_lin = hstart_y * TEX_WIDTH as i32 + hstart_x;
            let offset = (cur_lin - base_lin) as u32;

            // Write band header at the reserved slot.
            let header_idx = if axis == 0 { b } else { nby + b };
            let hx = (hstart_x + header_idx) % TEX_WIDTH as i32;
            let hy = hstart_y + (hstart_x + header_idx) / TEX_WIDTH as i32;
            let idx = tex_idx(hy, hx);
            builder.band_tex[idx] = tmp.len() as u32;
            builder.band_tex[idx + 1] = offset;
            builder.band_tex[idx + 2] = 0;
            builder.band_tex[idx + 3] = 0;

            // Write curve index list for this band.
            for entry in &tmp {
                assert!((builder.band_cy as usize) < TEX_HEIGHT);
                let idx = tex_idx(builder.band_cy, builder.band_cx);
                builder.band_tex[idx] = (entry.loc & 0xFFFF) as u32;
                builder.band_tex[idx + 1] = (entry.loc >> 16) as u32;
                builder.band_tex[idx + 2] = 0;
                builder.band_tex[idx + 3] = 0;
                builder.band_advance(1);
            }
        }
    }

    Ok(())
}

impl ShapeBuilder {
    /// Add a path to the builder by iterating its events and converting each
    /// segment to quadratic Bézier curves.
    ///
    /// `aabb` is the path's axis-aligned bounding box in local space.
    ///
    /// Returns the SlugShape metadata on success, or None if the builder is full.
    pub fn add_path(
        &mut self,
        path: impl IntoIterator<Item = PathEvent>,
        aabb: &core::units::LocalRect,
    ) -> Option<SlugShape> {
        self.curves.clear();

        for event in path {
            match event {
                PathEvent::Begin { .. } => {}
                PathEvent::Line { from, to } => {
                    // Represent the line as a quadratic with ctrl == from.
                    // using ctrl=midpoint(from,to) causes a.y ~= 0 in the quadratic
                    // equation solver in the shader.
                    // With ctrl=from, a.y = from.y - 2*from.y + to.y = to.y - from.y,
                    // which is non-zero for non-horizontal lines, ensuring
                    // exactly one root contributes (no cancellation).
                    self.curves.push(QuadraticBezierSegment { from, ctrl: from, to });
                }
                PathEvent::Quadratic { from, ctrl, to } => {
                    self.curves.push(QuadraticBezierSegment { from, ctrl, to });
                }
                PathEvent::Cubic { from, ctrl1, ctrl2, to } => {
                    let cubic = CubicBezierSegment { from, ctrl1, ctrl2, to };
                    cubic.for_each_quadratic_bezier(0.001, &mut |quad| {
                        self.curves.push(*quad);
                    });
                }
                PathEvent::End { last, first, .. } => {
                    if last != first {
                        self.curves.push(QuadraticBezierSegment { from: last, ctrl: last, to: first });
                    }
                }
            }
        }

        if self.curves.is_empty() {
            return None;
        }

        let band_count = {
            let bc = (self.curves.len() as f32).sqrt().ceil() as i32;
            bc.clamp(4, 64)
        };

        let mut shape = SlugShape {
            local_aabb: *aabb,
            ..Default::default()
        };

        if build_bands(self, &mut shape, band_count).is_err() {
            return None;
        }

        Some(shape)
    }
}
