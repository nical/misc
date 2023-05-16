use lyon::{tessellation::{FillTessellator, FillOptions}, lyon_tessellation::{VertexBuffers, FillVertexConstructor, BuffersBuilder}, path::traits::PathIterator};
use crate::{canvas::{ResourcesHandle, RendererId, Fill, Shape, RendererCommandIndex, Canvas, Pattern, RecordedShape, RecordedPattern, GpuResources, CanvasRenderer, RenderPasses, SubPass, ZIndex, CommonGpuResources}, gpu::{gpu_store::GpuStore, DynBufferRange}};
use std::ops::Range;

use super::MeshGpuResources;

pub const PATTERN_KIND_COLOR: u32 = 0;
pub const PATTERN_KIND_SIMPLE_LINEAR_GRADIENT: u32 = 1;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vertex {
    pub x: f32,
    pub y: f32,
    pub z_index: u32,
    pub pattern: u32,
}

unsafe impl bytemuck::Zeroable for Vertex {}
unsafe impl bytemuck::Pod for Vertex {}

struct VertexCtor {
    pattern: u32, z_index: u32,
}

impl FillVertexConstructor<Vertex> for VertexCtor {
    fn new_vertex(&mut self, vertex: lyon::lyon_tessellation::FillVertex) -> Vertex {
        let (x, y) = vertex.position().to_tuple();
        Vertex { x, y, z_index: self.z_index, pattern: self.pattern }
    }
}

struct MeshSubPass {
    opaque_geometry: Range<u32>,
    blended_geometry: Range<u32>,
    z_index: ZIndex,
}

pub struct MeshRenderer {
    renderer_id: RendererId,
    common_resources: ResourcesHandle<CommonGpuResources>,
    resources: ResourcesHandle<MeshGpuResources>,
    tessellator: FillTessellator,
    geometry: VertexBuffers<Vertex, u32>,
    tolerenace: f32,
    enable_opaque_pass: bool,
    enable_msaa: bool,

    commands: Vec<Fill>,
    render_passes: Vec<MeshSubPass>,
    vbo_range: Option<DynBufferRange>,
    ibo_range: Option<DynBufferRange>,
}

impl MeshRenderer {
    pub fn new(renderer_id: RendererId, common_resources: ResourcesHandle<CommonGpuResources>, resources: ResourcesHandle<MeshGpuResources>) -> Self {
        MeshRenderer {
            renderer_id,
            common_resources,
            resources,
            tessellator: FillTessellator::new(),
            geometry: VertexBuffers::new(),
            tolerenace: 0.25,
            enable_opaque_pass: false,
            enable_msaa: false,

            commands: Vec::new(),
            render_passes: Vec::new(),
            vbo_range: None,
            ibo_range: None,
        }
    }

    pub fn begin_frame(&mut self, canvas: &Canvas) {
        self.commands.clear();
        self.render_passes.clear();
        self.geometry.vertices.clear();
        self.geometry.indices.clear();
        self.enable_opaque_pass = canvas.surface.opaque_pass();
        self.enable_msaa = canvas.surface.msaa();
        self.vbo_range = None;
        self.ibo_range = None;
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

    pub fn prepare(&mut self, canvas: &Canvas, gpu_store: &mut GpuStore) {
        let commands = std::mem::take(&mut self.commands);
        for range in canvas.commands.with_renderer(self.renderer_id) {
            let range = range.start as usize .. range.end as usize;
            let z_index = commands[range.start].z_index;

            // Opaque pass.
            let opaque_start = self.geometry.indices.len() as u32;
            if self.enable_opaque_pass {
                for fill in commands[range.clone()].iter().rev() {
                    if fill.pattern.is_opaque() {
                        self.prepare_fill(fill, canvas, gpu_store);
                    }
                }
            }
            let opaque_geometry = opaque_start..self.geometry.indices.len() as u32;

            // Blended pass.
            for fill in commands[range.clone()].iter() {
                if !self.enable_opaque_pass || !fill.pattern.is_opaque() {
                    self.prepare_fill(fill, canvas, gpu_store);
                }
            }
            let blended_geometry = opaque_geometry.end..self.geometry.indices.len() as u32;

            if !opaque_geometry.is_empty() || !blended_geometry.is_empty() {
                self.render_passes.push(MeshSubPass {
                    opaque_geometry,
                    blended_geometry,
                    z_index,
                })
            }
        }

        self.commands = commands;
    }

    fn prepare_fill(&mut self, fill: &Fill, canvas: &Canvas, _gpu_store: &mut GpuStore) {
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
                let options = FillOptions::tolerance(self.tolerenace)
                    .with_fill_rule(shape.fill_rule);

                // TODO: some way to simplify/discard offscreen geometry, would probably be best
                // done in the tessellator itself.
                self.tessellator.tessellate(
                    shape.path.iter().transformed(transform),
                    &options,
                    &mut BuffersBuilder::new(
                        &mut self.geometry,
                        VertexCtor {
                            z_index: fill.z_index,
                            pattern,
                        },
                    )
                ).unwrap();
            }
            _ => {
                todo!()
            }
        }
    }

    pub fn upload(&mut self,
        resources: &mut GpuResources,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        let res = &mut resources[self.common_resources];
        self.vbo_range = res.vertices.upload(device, bytemuck::cast_slice(&self.geometry.vertices));
        self.ibo_range = res.indices.upload(device, bytemuck::cast_slice(&self.geometry.indices));
    }
}

impl CanvasRenderer for MeshRenderer {
    fn add_render_passes(&mut self, render_passes: &mut RenderPasses) {
        for (idx, pass) in self.render_passes.iter().enumerate() {
            render_passes.push(SubPass {
                renderer_id: self.renderer_id,
                internal_index: idx as u32,
                require_pre_pass: false,
                z_index: pass.z_index,
                use_depth: self.enable_opaque_pass,
                use_stencil: false,
                use_msaa: self.enable_msaa,
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
        let mesh_resources = &resources[self.resources];

        let pass = &self.render_passes[index as usize];

        render_pass.set_bind_group(0, &common_resources.main_target_and_gpu_store_bind_group, &[]);

        render_pass.set_index_buffer(common_resources.indices.get_buffer_slice(self.ibo_range.as_ref().unwrap()), wgpu::IndexFormat::Uint32);
        render_pass.set_vertex_buffer(0, common_resources.vertices.get_buffer_slice(self.vbo_range.as_ref().unwrap()));

        let pipelines = mesh_resources.pipelines(self.enable_opaque_pass, self.enable_msaa);

        // TODO: loop over batches.

        if !pass.opaque_geometry.is_empty() {
            render_pass.set_pipeline(&pipelines.opaque_color);
            render_pass.draw_indexed(pass.opaque_geometry.clone(), 0, 0..1);
        }

        if !pass.blended_geometry.is_empty() {
            render_pass.set_pipeline(&pipelines.alpha_color);
            render_pass.draw_indexed(pass.blended_geometry.clone(), 0, 0..1);
        }
    }
}

