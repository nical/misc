use bytemuck::Pod;

use crate::TILED_IMAGE_PATTERN;
use crate::gpu::ShaderSources;
use crate::tile_encoder::{BufferRange, TileAllocator};
use crate::tile_renderer::{PatternRenderer, BufferBumpAllocator};

use std::ops::Range;

pub trait CustomPattern: Sized {
    type Instance: Pod;
    type RenderData;

    fn is_opaque(&self) -> bool;

    fn new_tile(&self, tile_id: u32, x: f32, y: f32) -> Self::Instance;

    fn new_renderer(
        device: &wgpu::Device,
        shaders: &mut ShaderSources,
        tile_atlas_desc_layout: &wgpu::BindGroupLayout,
    ) -> CustomPatternRenderer<Self>;

    fn set_render_pass_state<'a, 'b: 'a>(data: &'a Self::RenderData, pass: &mut wgpu::RenderPass<'b>);
}

pub struct CustomPatternBuilder<Parameters: CustomPattern> {
    parameters: Parameters,
    tiles: Vec<Parameters::Instance>,
    vbo_range: BufferRange,
    render_passes: Vec<Range<u32>>,
    current_texture: u32,
    render_pass_start: u32,
    x: u32,
    y: u32,
    tile_size: f32,
}

impl<Parameters: CustomPattern> CustomPatternBuilder<Parameters> {
    pub fn new(tile_size: f32, parameters: Parameters) -> Self {
        CustomPatternBuilder {
            parameters,
            tiles: Vec::new(),
            vbo_range: BufferRange(0, 0),
            render_passes: Vec::new(),
            current_texture: 0,
            render_pass_start: 0,
            x: 0,
            y: 0,
            tile_size,
        }
    }

    pub fn set(&mut self, parameters: Parameters) {
        self.parameters = parameters;
    }

    pub fn reset(&mut self) {
        self.tiles.clear();
        self.render_pass_start = 0;
        self.current_texture = 0;
        self.render_passes.clear();
    }

    pub fn allocate_buffer_ranges(&mut self, renderer: &mut CustomPatternRenderer<Parameters>) {
        self.vbo_range = renderer.vbo_alloc.push(self.tiles.len());
    }

    pub fn upload(&self, renderer: &mut CustomPatternRenderer<Parameters>, queue: &wgpu::Queue) {
        if self.tiles.is_empty() {
            return;
        }
        queue.write_buffer(
            &renderer.vbo,
            self.vbo_range.byte_offset::<Parameters::Instance>(),
            bytemuck::cast_slice(&self.tiles),
        );
        renderer.render_passes.clear();
        renderer.render_passes.extend_from_slice(&self.render_passes);
}

    pub fn end_render_pass(&mut self) {
        while self.render_passes.len() as u32 <= self.current_texture {
            self.render_passes.push(0..0);
        }
        let render_pass_end = self.tiles.len() as u32;
        self.render_passes[self.current_texture as usize] = self.render_pass_start..render_pass_end;
        self.render_pass_start = render_pass_end;
    }
}

impl<Parameters: CustomPattern> crate::tiler::TilerPattern for CustomPatternBuilder<Parameters> {
    fn pattern_kind(&self) -> u32 { TILED_IMAGE_PATTERN }
    fn tile_is_opaque(&self) -> bool { self.parameters.is_opaque() }
    fn set_tile(&mut self, x: u32, y: u32) {
        self.x = x;
        self.y = y;
    }

    fn request_tile(&mut self, atlas: &mut TileAllocator) -> u32 {
        let (tile_id, texture) = atlas.allocate();

        if texture != self.current_texture {
            self.end_render_pass();
            self.current_texture = texture;
        }

        let instance = self.parameters.new_tile(
            tile_id,
            self.x as f32 * self.tile_size,
            self.y as f32 * self.tile_size,
        );

        self.tiles.push(instance);

        tile_id
    }

    fn set_render_pass(&mut self, pass_index: u32) {
        if pass_index == self.current_texture {
            return;
        }

        self.end_render_pass();

        self.current_texture = pass_index;
    }
}

pub struct CustomPatternRenderer<Pattern: CustomPattern> {
    pipeline: wgpu::RenderPipeline,
    // TODO: should be in a different struct.
    vbo: wgpu::Buffer,
    vbo_alloc: BufferBumpAllocator,
    render_passes: Vec<Range<u32>>,
    data: Pattern::RenderData,
}

impl<Pattern: CustomPattern> CustomPatternRenderer<Pattern> {
    pub fn new(
        label: &str,
        device: &wgpu::Device,
        pipeline: wgpu::RenderPipeline,
        data: Pattern::RenderData,
    ) -> Self {

        let vbo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: 4096 * 64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        CustomPatternRenderer {
            pipeline,
            vbo,
            vbo_alloc: BufferBumpAllocator::new(),
            render_passes: Vec::new(),
            data,
        }
    }

    fn get_tiles(&self, pass_idx: u32) -> Option<Range<u32>> {
        let pass_idx = pass_idx as usize;
        if pass_idx >= self.render_passes.len() {
            return None;
        }

        let range = self.render_passes[pass_idx].clone();

        if range.is_empty() {
            None
        } else {
            Some(range)
        }
    }
}

impl<Pattern: CustomPattern> PatternRenderer for CustomPatternRenderer<Pattern> {
    fn begin_frame(&mut self) {
        self.vbo_alloc.clear();
    }

    fn has_content(&self, pass_idx: u32) -> bool {
        self.get_tiles(pass_idx).is_some()
    }

    fn render<'a, 'b: 'a>(&'b self, pass_idx: u32, pass: &mut wgpu::RenderPass<'a>) {
        if let Some(tile_range) = self.get_tiles(pass_idx) {
            pass.set_pipeline(&self.pipeline);
            pass.set_vertex_buffer(0, self.vbo.slice(..));
            Pattern::set_render_pass_state(&self.data, pass);
            pass.draw_indexed(0..6, 0, tile_range);
        }
    }
}
