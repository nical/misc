use core::num::NonZeroU32;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct GpuStoreHandle(u32);

unsafe impl bytemuck::Pod for GpuStoreHandle {}
unsafe impl bytemuck::Zeroable for GpuStoreHandle {}

impl GpuStoreHandle {
    pub const INVALID: Self = GpuStoreHandle(std::u32::MAX);

    fn new(x: u32, y: u32) -> Self {
        debug_assert!(x < 1 << 16);
        debug_assert!(y < 1 << 16);
        GpuStoreHandle((x << 16) | y)
    }

    pub fn to_u32(self) -> u32 { self.0 }
}

pub struct GpuStore {
    data: Vec<f32>,

    current_row: usize,
    current_row_offset: usize,
    width: usize,
    height: u16,

    texture: wgpu::Texture,
}

impl GpuStore {
    pub fn new(w: u16, h: u16, device: &wgpu::Device) -> Self {
        let w = w as usize;
        let size = w * h as usize;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gpu store"),
            size: wgpu::Extent3d {
                width: w as u32,
                height: h as u32,
                depth_or_array_layers: 1,
            },
            sample_count: 1,
            mip_level_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        GpuStore {
            current_row: 0,
            current_row_offset: 0,
            data: vec![0.0; size],
            width: w * 4,
            height: h,

            texture,
        }
    }

    pub fn push(&mut self, data: &[f32]) -> GpuStoreHandle {
        assert!(data.len() <= self.width);
        let size = (data.len() + 3) & !3;

        let rem = self.width - self.current_row_offset;
        if rem < size {
            self.current_row += 1;
            self.current_row_offset = 0;
        }

        let offset = self.current_row * self.width + self.current_row_offset;
        self.data[offset .. offset + data.len()].copy_from_slice(data);

        let handle = GpuStoreHandle::new(
            (self.current_row_offset / 4) as u32,
            self.current_row as u32
        );
        self.current_row_offset += size;

        assert!(self.current_row_offset < self.height as usize);

        return handle;
    }

    pub fn clear(&mut self) {
        self.current_row = 0;
        self.current_row_offset = 0;
    }

    pub fn upload(&self, queue: &wgpu::Queue) {
        if self.current_row == 0 && self.current_row_offset == 0 {
            return;
        }

        let rows = self.current_row + if self.current_row_offset == 0 { 0 } else { 1 };
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&self.data[..(rows * self.width)]),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(self.width as u32 * 4),
                rows_per_image: NonZeroU32::new(rows as u32),
            },
            wgpu::Extent3d {
                width: self.width as u32 / 4,
                height: rows as u32,
                depth_or_array_layers: 1,
            }
        );
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }
}
