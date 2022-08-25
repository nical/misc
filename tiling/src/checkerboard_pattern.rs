use crate::{Color, TILED_IMAGE_PATTERN};
use crate::gpu::{ShaderSources, VertexBuilder, PipelineDefaults};
use crate::tile_encoder::BufferRange;
use crate::tile_renderer::BufferBumpAllocator;
use crate::tiler::TileIdAllcator;

pub struct CheckerboardPatternBuilder {
    colors: [Color; 2],
    tiles: Vec<CheckerboardPatternTile>,
    ids: TileIdAllcator,
    is_opaque: bool,
    tile_size: f32,
    scale: f32,
    vbo_range: BufferRange,
    x: u32,
    y: u32,
}

impl CheckerboardPatternBuilder {
    pub fn new(color1: Color, color2: Color, scale: f32, ids: TileIdAllcator, tile_size: f32) -> Self {
        CheckerboardPatternBuilder {
            colors: [color1, color2],
            tiles: Vec::new(),
            ids,
            is_opaque: color1.is_opaque() && color2.is_opaque(),
            tile_size,
            scale,
            vbo_range: BufferRange(0, 0),
            x: 0,
            y: 0,
        }
    }

    pub fn set(&mut self, color1: Color, color2: Color, scale: f32) {
        self.colors = [color1, color2];
        self.is_opaque = color1.is_opaque() && color2.is_opaque();
        self.scale = scale;
    }

    pub fn tiles(&self) -> &[CheckerboardPatternTile] {
        &self.tiles
    }

    pub fn reset(&mut self) {
        self.tiles.clear();
    }

    pub fn allocate_buffer_ranges(&mut self, renderer: &mut CheckerboardRenderer) {
        self.vbo_range = renderer.vbo_alloc.push(self.tiles.len());
    }

    pub fn upload(&self, renderer: &mut CheckerboardRenderer, queue: &wgpu::Queue) {
        if !self.tiles.is_empty() {
            queue.write_buffer(
                &renderer.vbo,
                self.vbo_range.byte_offset::<CheckerboardPatternTile>(),
                bytemuck::cast_slice(self.tiles()),
            );
        }
    }
}

impl crate::tiler::TilerPattern for CheckerboardPatternBuilder {
    fn pattern_kind(&self) -> u32 { TILED_IMAGE_PATTERN }
    fn tile_is_opaque(&self) -> bool { self.is_opaque }
    fn set_tile(&mut self, x: u32, y: u32) {
        self.x = x;
        self.y = y;
    }
    fn request_tile(&mut self) -> u32 {
        let tile_id = self.ids.allocate();

        self.tiles.push(CheckerboardPatternTile {
            tile_id,
            offset: [
                self.x as f32 * self.tile_size,
                self.y as f32 * self.tile_size,
            ],
            scale: self.scale,
            colors: [
                self.colors[0].to_u32(),
                self.colors[1].to_u32(),
            ],
        });

        tile_id
    }
}


#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CheckerboardPatternTile {
    pub tile_id: u32,
    pub scale: f32,
    pub colors: [u32; 2],
    pub offset: [f32; 2],
}

unsafe impl bytemuck::Pod for CheckerboardPatternTile {}
unsafe impl bytemuck::Zeroable for CheckerboardPatternTile {}

pub struct CheckerboardRenderer {
    pub pipeline: wgpu::RenderPipeline,
    // TODO: should be in a different struct.
    pub vbo: wgpu::Buffer,
    pub vbo_alloc: BufferBumpAllocator,
}

impl CheckerboardRenderer {
    pub fn new(
        device: &wgpu::Device,
        shaders: &mut ShaderSources,
        tile_atlas_desc_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let src = include_str!("../shaders/checkerboard_pattern.wgsl");
        let module = shaders.create_shader_module(device, "checkerboard_pattern", src, &[]);

        let defaults = PipelineDefaults::new();
        let attributes = VertexBuilder::from_slice(&[
            wgpu::VertexFormat::Uint32,
            wgpu::VertexFormat::Float32,
            wgpu::VertexFormat::Uint32x2,
            wgpu::VertexFormat::Float32x2,
        ]);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Opaque solid tiles"),
            bind_group_layouts: &[&tile_atlas_desc_layout],
            push_constant_ranges: &[],
        });
        let pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Checkerboard pattern tiles"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: "vs_main",
                buffers: &[attributes.buffer_layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "fs_main",
                targets: defaults.color_target_state_no_blend(),
            }),
            primitive: defaults.primitive_state(),
            depth_stencil: None,
            multiview: None,
            multisample: wgpu::MultisampleState::default(),
        };

        let vbo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Checkerboard tiles"),
            size: 4096 * 64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        CheckerboardRenderer {
            pipeline: device.create_render_pipeline(&pipeline_descriptor),
            vbo,
            vbo_alloc: BufferBumpAllocator::new(),
        }
    }

    pub fn begin_frame(&mut self) {
        self.vbo_alloc.clear();
    }
}
