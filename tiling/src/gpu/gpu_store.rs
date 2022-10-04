use core::num::NonZeroU32;

const GPU_STORE_WIDTH: u32 = 2048;
const FLOATS_PER_ROW: usize = GPU_STORE_WIDTH as usize * 4;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct GpuStoreHandle(u32);

unsafe impl bytemuck::Pod for GpuStoreHandle {}
unsafe impl bytemuck::Zeroable for GpuStoreHandle {}

impl GpuStoreHandle {
    pub const INVALID: Self = GpuStoreHandle(std::u32::MAX);

    pub fn to_u32(self) -> u32 { self.0 }
}

pub struct GpuStore {
    data: Vec<f32>,

    offset: usize,
    height: usize,

    texture: wgpu::Texture,
}

impl GpuStore {
    pub fn new(h: u32, device: &wgpu::Device) -> Self {
        let size = FLOATS_PER_ROW * h as usize;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gpu store"),
            size: wgpu::Extent3d {
                width: GPU_STORE_WIDTH,
                height: h,
                depth_or_array_layers: 1,
            },
            sample_count: 1,
            mip_level_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        GpuStore {
            offset: 0,
            data: vec![0.0; size],
            height: h as usize,

            texture,
        }
    }

    pub fn push(&mut self, data: &[f32]) -> GpuStoreHandle {
        let size = (data.len() + 3) & !3;
        if self.data.len() < self.offset + size {
            self.data.resize(self.data.len() * 2, 0.0);
        }

        self.data[self.offset .. self.offset + data.len()].copy_from_slice(data);

        let handle = GpuStoreHandle(self.offset as u32 / 4);
        self.offset += size;

        return handle;
    }

    pub fn clear(&mut self) {
        self.offset = 0;
    }

    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.offset == 0 {
            return;
        }

        let w = 4 * GPU_STORE_WIDTH as usize;
        let rows = self.offset / w + if self.offset % w == 0 { 0 } else { 1 };

        if rows > self.height {
            self.height = self.data.len() / FLOATS_PER_ROW;
            self.texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gpu store"),
                size: wgpu::Extent3d {
                    width: GPU_STORE_WIDTH,
                    height: self.height as u32,
                    depth_or_array_layers: 1,
                },
                sample_count: 1,
                mip_level_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            });
    
        }

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&self.data[..(rows * w)]),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(GPU_STORE_WIDTH * 16),
                rows_per_image: NonZeroU32::new(rows as u32),
            },
            wgpu::Extent3d {
                width: GPU_STORE_WIDTH,
                height: rows as u32,
                depth_or_array_layers: 1,
            }
        );
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }
}
