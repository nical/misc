use lyon::geom::Vector;
pub mod advanced_tiles;
pub mod mask_uploader;
pub mod render_target;

pub use wgslp::preprocessor::{Preprocessor, Source, SourceError};
use std::{collections::HashMap};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct GpuGlobals {
    pub resolution: Vector<f32>,
    pub tile_size: u32,
    pub tile_atlas_size: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct GpuTileAtlasDescriptor {
    pub tile_size: f32,
    pub inv_atlas_size: f32,
    pub masks_per_row: u32,
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

        library.insert("quad".into(), include_str!("../../shaders/quad.wgsl").into());
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
