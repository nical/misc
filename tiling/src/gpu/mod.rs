pub mod mask_uploader;
//pub mod render_target;

pub use wgslp::preprocessor::{Preprocessor, Source, SourceError};
use std::{collections::HashMap};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct GpuTileAtlasDescriptor {
    pub inv_width: f32,
    pub inv_height: f32,
    pub tiles_per_row: u32,
    pub tiles_per_atlas: u32,
}

impl GpuTileAtlasDescriptor {
    pub fn new(w: u32, h: u32, tile_size: u32) -> Self {
        let inv_width = 1.0 / (w as f32);
        let inv_height = 1.0 / (h as f32);
        let tiles_per_row = (w + tile_size - 1) / tile_size;
        let tiles_per_atlas = tiles_per_row * ((h + tile_size - 1) / tile_size);
        GpuTileAtlasDescriptor { inv_width, inv_height, tiles_per_row, tiles_per_atlas }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct GpuGlobals {
    pub target_tiles: GpuTileAtlasDescriptor,
    pub src_masks: GpuTileAtlasDescriptor,
    pub src_color: GpuTileAtlasDescriptor,
}

unsafe impl bytemuck::Pod for GpuGlobals {}
unsafe impl bytemuck::Zeroable for GpuGlobals {}
unsafe impl bytemuck::Pod for GpuTileAtlasDescriptor {}
unsafe impl bytemuck::Zeroable for GpuTileAtlasDescriptor {}

impl GpuGlobals {
    pub fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Globals"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<GpuGlobals>() as u64),
                    },
                    count: None,
                },
            ],
        })
    }
}

impl GpuTileAtlasDescriptor {
    pub fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mask atlas descriptor"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<GpuTileAtlasDescriptor>() as u64),
                    },
                    count: None,
                },
            ],
        })
    }
}

pub struct ShaderSources {
    pub source_library: HashMap<String, Source>,
    pub preprocessor: Preprocessor,
}

impl ShaderSources {
    pub fn new() -> Self {
        let mut library = HashMap::default();

        library.insert("rect".into(), include_str!("../../shaders/rect.wgsl").into());
        library.insert("tiling".into(), include_str!("../../shaders/tiling.wgsl").into());
        library.insert("render_target".into(), include_str!("../../shaders/render_target.wgsl").into());
        library.insert("raster::fill".into(), include_str!("../../shaders/raster/fill.wgsl").into());
        library.insert("pattern::color".into(), include_str!("../../shaders/pattern/color.wgsl").into());

        ShaderSources {
            source_library: library,
            preprocessor: Preprocessor::new()
        }
    }

    pub fn preprocess(&mut self, name: &str, src: &str, defines: &[&str]) -> Result<String, SourceError> {
        self.preprocessor.reset_defines();
        for define in defines {
            self.preprocessor.define(define);
        }
        self.preprocessor.preprocess(name, src, &mut self.source_library)
    }

    pub fn create_shader_module(
        &mut self,
        device: &wgpu::Device,
        name: &str,
        src: &str,
        defines: &[&str]
    ) -> wgpu::ShaderModule {
        let src = self.preprocess(name, src, defines).unwrap();

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });

        module
    }
}

pub struct VertexBuilder {
    location: u32,
    offset: u64,
    attributes: Vec<wgpu::VertexAttribute>,
}

impl VertexBuilder {
    pub fn new() -> Self {
        VertexBuilder { location: 0, offset: 0, attributes: Vec::with_capacity(16) }
    }

    pub fn from_slice(formats: &[wgpu::VertexFormat]) -> Self {
        let mut attributes = VertexBuilder::new();
        for format in formats {
            attributes.push(*format);
        }

        attributes
    }

    pub fn push(&mut self, format: wgpu::VertexFormat) {
        self.attributes.push(wgpu::VertexAttribute {
            format,
            offset: self.offset,
            shader_location: self.location
        });
        self.offset += format.size();
        self.location += 1;
    }

    pub fn get(&self) -> &[wgpu::VertexAttribute] {
        &self.attributes
    }

    pub fn clear(&mut self) {
        self.location = 0;
        self.offset = 0;
        self.attributes.clear();
    }
}

pub struct PipelineHelpers {
}

impl PipelineHelpers {
    pub fn new() -> Self {
        PipelineHelpers {  }
    }
    pub fn default_primitive_state(&self) -> wgpu::PrimitiveState {
        wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            polygon_mode: wgpu::PolygonMode::Fill,
            front_face: wgpu::FrontFace::Ccw,
            strip_index_format: None,
            cull_mode: None,
            unclipped_depth: false,
            conservative: false,
        }
    }
}