use std::ops::Range;

use lyon::{path::{PathSlice, PathEvent, traits::PathIterator}, math::{Box2D, Point, Transform, point}, geom::{QuadraticBezierSegment, CubicBezierSegment}, lyon_tessellation::VertexBuffers};

use crate::{canvas::{Canvas, Shape, Pattern, RendererCommandIndex, Fill, GpuResources, RendererId, ResourcesHandle, CommonGpuResources, RecordedShape, RecordedPattern, CanvasRenderer, SubPass, ZIndex, RenderPasses}, tiling::resources, gpu::DynBufferRange};

use super::StencilAndCoverResources;



#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct StencilVertex {
    pub x: f32, pub y: f32 
}

impl StencilVertex {
    pub fn from_point(p: Point) -> Self {
        StencilVertex { x: p.x, y: p.y }
    }
}

unsafe impl bytemuck::Pod for StencilVertex {}
unsafe impl bytemuck::Zeroable for StencilVertex {}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CoverVertex {
    pub x: f32,
    pub y: f32,
    pub z_index: u32,
    pub pattern: u32,
}

unsafe impl bytemuck::Pod for CoverVertex {}
unsafe impl bytemuck::Zeroable for CoverVertex {}

struct Draw {
    stencil_indices: Range<u32>,
    cover_indices: Range<u32>,
    pattern: u32,
    z_index: ZIndex,
}

struct Pass {
    draws: Range<usize>,
    z_index: ZIndex,
}

pub struct StencilAndCoverRenderer {
    commands: Vec<Fill>,
    renderer_id: RendererId,
    resources: ResourcesHandle<StencilAndCoverResources>,
    common_resources: ResourcesHandle<CommonGpuResources>,
    stencil_geometry: VertexBuffers<StencilVertex, u32>,
    cover_geometry: VertexBuffers<CoverVertex, u32>,
    draws: Vec<Draw>,
    passes: Vec<Pass>,
    vbo_range: Option<DynBufferRange>,
    ibo_range: Option<DynBufferRange>,
    cover_vbo_range: Option<DynBufferRange>,
    cover_ibo_range: Option<DynBufferRange>,
    enable_msaa: bool,
}

impl StencilAndCoverRenderer {
    pub fn new(
        renderer_id: RendererId,
        common_resources: ResourcesHandle<CommonGpuResources>,    
        resources: ResourcesHandle<StencilAndCoverResources>,
    ) -> Self {
        StencilAndCoverRenderer {
            commands: Vec::new(),
            renderer_id,
            resources,
            common_resources,
            stencil_geometry: VertexBuffers::new(),
            cover_geometry: VertexBuffers::new(),
            draws: Vec::new(),
            passes: Vec::new(),
            vbo_range: None,
            ibo_range: None,
            cover_vbo_range: None,
            cover_ibo_range: None,
            enable_msaa: false,
        }
    }

    pub fn begin_frame(&mut self, canvas: &Canvas) {
        self.commands.clear();
        self.draws.clear();
        self.passes.clear();
        self.stencil_geometry.vertices.clear();
        self.stencil_geometry.indices.clear();
        self.cover_geometry.vertices.clear();
        self.cover_geometry.indices.clear();
        self.vbo_range = None;
        self.ibo_range = None;
        self.cover_vbo_range = None;
        self.cover_ibo_range = None;
        self.enable_msaa = canvas.surface.msaa();
    }

    pub fn fill<S: Shape, P: Pattern>(&mut self, canvas: &mut Canvas, shape: S, pattern: P) {
        let transform = canvas.transforms.current();
        let z_index = canvas.z_indices.push();
        let index = self.commands.len() as RendererCommandIndex;
        self.commands.push(Fill {
            shape: shape.to_command(),
            pattern: pattern.to_command(),
            transform,
            z_index,
        });
        canvas.commands.push(self.renderer_id, index);
    }

    pub fn prepare(&mut self, canvas: &Canvas) {
        let commands = std::mem::take(&mut self.commands);
        for range in canvas.commands.with_renderer(self.renderer_id) {
            let range = range.start as usize .. range.end as usize;

            let draws_start = self.draws.len();
            let pass_z_index = commands[range.start].z_index;

            for fill in commands[range].iter() {
                let transform = canvas.transforms.get(fill.transform);

                let pattern = match fill.pattern {
                    RecordedPattern::Color(color) => {
                        color.to_u32()
                    }
                    _ => {
                        todo!();
                    }
                };

                match &fill.shape {
                    RecordedShape::Path(shape) => {
                        let aabb = lyon::algorithms::aabb::fast_bounding_box(&shape.path.as_slice());

                        let idx_start = self.stencil_geometry.indices.len() as u32;
                        generate_geometry(
                            shape.path.as_slice(),
                            transform,
                            canvas.params.tolerance,
                            &aabb,
                            &mut self.stencil_geometry,
                        );
                        let idx_end = self.stencil_geometry.indices.len() as u32;

                        let a = transform.transform_point(aabb.min);
                        let b = transform.transform_point(point(aabb.max.x, aabb.min.y));
                        let c = transform.transform_point(aabb.max);
                        let d = transform.transform_point(point(aabb.min.x, aabb.max.y));

                        let z_index = fill.z_index;

                        let cov_start = self.cover_geometry.indices.len() as u32;
                        let offset = self.cover_geometry.vertices.len() as u32;
                        self.cover_geometry.vertices.push(CoverVertex { x: a.x, y: a.y, z_index, pattern });
                        self.cover_geometry.vertices.push(CoverVertex { x: b.x, y: b.y, z_index, pattern });
                        self.cover_geometry.vertices.push(CoverVertex { x: c.x, y: c.y, z_index, pattern });
                        self.cover_geometry.vertices.push(CoverVertex { x: d.x, y: d.y, z_index, pattern });
                        self.cover_geometry.indices.push(offset);
                        self.cover_geometry.indices.push(offset + 1);
                        self.cover_geometry.indices.push(offset + 2);
                        self.cover_geometry.indices.push(offset);
                        self.cover_geometry.indices.push(offset + 2);
                        self.cover_geometry.indices.push(offset + 3);
                        let cov_end = self.cover_geometry.indices.len() as u32;

                        self.draws.push(Draw {
                            stencil_indices: idx_start..idx_end,
                            cover_indices: cov_start..cov_end,
                            pattern,
                            z_index,
                        });
                    }
                    _ => {
                        todo!()
                    }
                }
            }

            let draws_end = self.draws.len();

            if draws_start < draws_end {
                self.passes.push(Pass {
                    draws: draws_start..draws_end,
                    z_index: pass_z_index,
                });
            }
        }
    }

    pub fn upload(&mut self, resources: &mut GpuResources, device: &wgpu::Device) {
        let res = &mut resources[self.common_resources];
        self.vbo_range = res.vertices.upload(device, bytemuck::cast_slice(&self.stencil_geometry.vertices));
        self.ibo_range = res.indices.upload(device, bytemuck::cast_slice(&self.stencil_geometry.indices));
        self.cover_vbo_range = res.vertices.upload(device, bytemuck::cast_slice(&self.cover_geometry.vertices));
        self.cover_ibo_range = res.indices.upload(device, bytemuck::cast_slice(&self.cover_geometry.indices));
    }
}

pub fn generate_geometry(
    path: PathSlice,
    transform: &Transform,
    tolerance: f32,
    aabb: &Box2D,
    stencil_geometry: &mut VertexBuffers<StencilVertex, u32>,
) {
    let clip_rect = transform.outer_transformed_box(
        &aabb
    );

    let vertices = &mut stencil_geometry.vertices;
    let indices = &mut stencil_geometry.indices;

    fn vertex(vertices: &mut Vec<StencilVertex>, p: Point) -> u32 {
        let idx = vertices.len() as u32;
        vertices.push(StencilVertex::from_point(p));
        idx
    }

    fn triangle(indices: &mut Vec<u32>, a: u32, b: u32, c: u32) {
        indices.push(a);
        indices.push(b);
        indices.push(c)
    }

    let pivot = vertex(vertices, clip_rect.min);

    for evt in path.iter().transformed(transform) {
        match evt {
            PathEvent::Begin { at } => {

            }
            PathEvent::End { last, first, .. } => {
                let f = clip_rect.contains(last);
                let t = clip_rect.contains(first);
                if !f && !t {
                    continue;
                }

                let a = vertex(vertices, last);
                let b = vertex(vertices, first);
                triangle(indices, pivot, a, b);
            }
            PathEvent::Line { from, to } => {
                let f = clip_rect.contains(from);
                let t = clip_rect.contains(to);
                if !f && !t {
                    continue;
                }

                let a = vertex(vertices, from);
                let b = vertex(vertices, to);
                triangle(indices, pivot, a, b);
            }
            PathEvent::Quadratic { from, ctrl, to } => {
                let f = clip_rect.contains(from);
                let c = clip_rect.contains(ctrl);
                let t = clip_rect.contains(to);
                if !f && !c && !t {
                    continue;
                }

                let a = vertex(vertices, from);
                let b = vertex(vertices, to);

                triangle(indices, pivot, a, b);
                let mut prev = a;
                QuadraticBezierSegment { from, ctrl, to }.for_each_flattened(tolerance, &mut |seg| {
                    let next = vertex(vertices, seg.to);
                    if prev != a {
                        triangle(indices, a, prev, next);
                    }
                    prev = next;
                });
            }
            PathEvent::Cubic { from, ctrl1, ctrl2, to } => {
                let f = clip_rect.contains(from);
                let c1 = clip_rect.contains(ctrl1);
                let c2 = clip_rect.contains(ctrl1);
                let t = clip_rect.contains(to);
                if !f && !c1 && !c2 && !t {
                    continue;
                }

                CubicBezierSegment { from, ctrl1, ctrl2, to}.for_each_quadratic_bezier(tolerance, &mut |quad| {
                    let a = vertex(vertices, quad.from);
                    let b = vertex(vertices, quad.to);
    
                    triangle(indices, pivot, a, b);
                    let mut prev = a;
                    quad.for_each_flattened(tolerance, &mut |seg| {
                        let next = vertex(vertices, seg.to);
                        if prev != a {
                            triangle(indices, a, prev, next);    
                        }
                        prev = next;
                    });
                });
            }
        }
    }
}

impl CanvasRenderer for StencilAndCoverRenderer {
    fn add_render_passes(&mut self, render_passes: &mut RenderPasses) {
        for (idx, pass) in self.passes.iter().enumerate() {
            render_passes.push(SubPass {
                renderer_id: self.renderer_id,
                internal_index: idx as u32,
                require_pre_pass: false,
                z_index: pass.z_index,
                use_depth: false,
                use_msaa: self.enable_msaa,
                use_stencil: true,
            });
        }
    }

    fn render<'pass, 'resources: 'pass>(
        &self,
        index: u32,
        resources: &'resources GpuResources,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {

        let common_resources = &resources[self.common_resources];
        let stencil_resources = &resources[self.resources];

        let pass = &self.passes[index as usize];

        render_pass.set_bind_group(0, &common_resources.main_target_and_gpu_store_bind_group, &[]);

        render_pass.set_stencil_reference(128);

        for draw in &self.draws[pass.draws.clone()] {
            // Stencil
            let pipeline = if self.enable_msaa {
                &stencil_resources.msaa_stencil_pipeline
            } else {
                &stencil_resources.stencil_pipeline
            };
            render_pass.set_index_buffer(common_resources.indices.get_buffer_slice(self.ibo_range.as_ref().unwrap()), wgpu::IndexFormat::Uint32);
            render_pass.set_vertex_buffer(0, common_resources.vertices.get_buffer_slice(self.vbo_range.as_ref().unwrap()));
            render_pass.set_pipeline(pipeline);
            render_pass.draw_indexed(draw.stencil_indices.clone(), 0, 0..1);

            // Cover
            let pipeline = if self.enable_msaa {
                &stencil_resources.msaa_evenodd_color_pipeline
            } else {
                &stencil_resources.evenodd_color_pipeline
            };

            render_pass.set_index_buffer(common_resources.indices.get_buffer_slice(self.cover_ibo_range.as_ref().unwrap()), wgpu::IndexFormat::Uint32);
            render_pass.set_vertex_buffer(0, common_resources.vertices.get_buffer_slice(self.cover_vbo_range.as_ref().unwrap()));
            render_pass.set_pipeline(pipeline);
            render_pass.draw_indexed(draw.cover_indices.clone(), 0, 0..1);
        }
    }
}