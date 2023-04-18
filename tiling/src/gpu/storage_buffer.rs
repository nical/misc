use std::num::NonZeroU32;
use std::ops::Range;

const FLOAT_TEXTURE_WIDTH: u32 = 1024;
const BYTES_PER_ROW: u32 = FLOAT_TEXTURE_WIDTH * 16;

pub enum StorageKind {
    Buffer,
    Texture,
}

enum Storage {
    Buffer { buffer: wgpu::Buffer },
    Texture { texture: wgpu::Texture, view: wgpu::TextureView },
}

pub struct StorageBuffer {
    handle: Storage,
    label: &'static str,
    allocator: BufferBumpAllocator,
    allocated_size: u32,
    size_per_element: u32,
    epoch: u32,
}

impl StorageBuffer {
    pub fn new<T>(device: &wgpu::Device, label: &'static str, size: u32, kind: StorageKind) -> Self {
        let size_per_element = std::mem::size_of::<T>() as u32;
        let byte_size = size * 4;
        let handle = match kind {
            StorageKind::Buffer => {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: byte_size as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });

                Storage::Buffer { buffer }
            }
            StorageKind::Texture => {
                let h = byte_size / BYTES_PER_ROW;
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    dimension: wgpu::TextureDimension::D2,
                    size: wgpu::Extent3d {
                        width: FLOAT_TEXTURE_WIDTH,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    format: wgpu::TextureFormat::Rgba32Float,
                    usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                    mip_level_count: 1,
                    sample_count: 1,
                    view_formats: &[wgpu::TextureFormat::Rgba32Float],
                });

                let view = texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(label),
                    format: Some(wgpu::TextureFormat::Rgba32Float),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    base_array_layer: 0,
                    mip_level_count: NonZeroU32::new(1),
                    array_layer_count: NonZeroU32::new(1),
                });
        
                Storage::Texture { texture, view }
            }
        };

        StorageBuffer {
            handle,
            label,
            allocator: BufferBumpAllocator::new(),
            allocated_size: byte_size,
            size_per_element,
            epoch: 0,
        }
    }

    pub fn begin_frame(&mut self) {
        self.allocator.clear();
    }

    pub fn ensure_allocated(&mut self, device: &wgpu::Device) -> bool {
        if self.allocator.len() * self.size_per_element <= self.allocated_size {
            return false;
        }

        let p = self.allocated_size;
        let s = self.allocator.len() * self.size_per_element;
        let multiple = BYTES_PER_ROW * 4;
        self.allocated_size = (s + (multiple - 1)) & !(multiple - 1);
        println!("reallocate {:?} from {} to {} ({})", self.label, p, self.allocated_size, s);

        match &mut self.handle {
            Storage::Buffer { buffer } => {
                buffer.destroy();
                *buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(self.label),
                    size: self.allocated_size as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
            }
            Storage::Texture { texture, .. } => {
                texture.destroy();
                let h = self.allocated_size / BYTES_PER_ROW;
                *texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(self.label),
                    dimension: wgpu::TextureDimension::D2,
                    size: wgpu::Extent3d {
                        width: FLOAT_TEXTURE_WIDTH,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    format: wgpu::TextureFormat::Rgba32Float,
                    usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                    mip_level_count: 1,
                    sample_count: 1,
                    view_formats: &[wgpu::TextureFormat::Rgba32Float],
                });
            }
        }

        self.epoch = self.epoch.wrapping_add(1);

        true
    }

    pub fn upload(&mut self, offset: u32, data: &[f32], queue: &wgpu::Queue) {
        self.upload_bytes(offset, bytemuck::cast_slice(data), queue)
    }

    pub fn upload_bytes(&mut self, offset: u32, data: &[u8], queue: &wgpu::Queue) {
        match &self.handle {
            Storage::Buffer { buffer } => {
                queue.write_buffer(
                    buffer,
                    offset as u64,
                    bytemuck::cast_slice(data),
                );
            }
            Storage::Texture { texture, .. } => {
                let w = BYTES_PER_ROW / self.size_per_element;
                let offset = self.allocator.len();
                let rows_per_image = self.allocated_size / BYTES_PER_ROW;
                let full_rows = offset / w;
                let split = (full_rows * BYTES_PER_ROW) as usize;

                if full_rows > 0 {
                    queue.write_texture(
                        wgpu::ImageCopyTexture {
                            texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        bytemuck::cast_slice(&data[..split]),
                        wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: NonZeroU32::new(BYTES_PER_ROW),
                            rows_per_image: NonZeroU32::new(rows_per_image),
                        },
                        wgpu::Extent3d {
                            width: FLOAT_TEXTURE_WIDTH,
                            height: full_rows,
                            depth_or_array_layers: 1,
                        }
                    );    
                }
                if offset % w != 0 {
                    let rem = data.len() - split;
                    queue.write_texture(
                        wgpu::ImageCopyTexture {
                            texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d {
                                x: 0,
                                y: full_rows,
                                z: 0,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        bytemuck::cast_slice(&data[split..]),
                        wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: NonZeroU32::new(BYTES_PER_ROW),
                            rows_per_image: NonZeroU32::new(rows_per_image),
                        },
                        wgpu::Extent3d {
                            width: rem as u32 / self.size_per_element,
                            height: 1,
                            depth_or_array_layers: 1,
                        }
                    );
                }
            }
        }
    }

    pub fn bump_allocator(&mut self) -> &mut BufferBumpAllocator {
        &mut self.allocator
    }

    pub fn buffer(&self) -> Option<&wgpu::Buffer> {
        match &self.handle {
            Storage::Buffer { buffer, .. } => Some(buffer),
            _ => None,
        }
    }

    pub fn texture(&self) -> Option<&wgpu::Texture> {
        match &self.handle {
            Storage::Texture{ texture, .. } => Some(texture),
            _ => None,
        }
    }

    pub fn binding_type(&self) -> wgpu::BindingType {
        match &self.handle {
            Storage::Buffer { .. } => {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(self.size_per_element as u64),
                }        
            }
            Storage::Texture { .. } => {
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                }
            }
        }
    }

    pub fn binding_resource(&self) -> wgpu::BindingResource {
        match &self.handle {
            Storage::Buffer { buffer } => {
                wgpu::BindingResource::Buffer(buffer.as_entire_buffer_binding())
            }
            Storage::Texture { view, .. } => {
                wgpu::BindingResource::TextureView(view)
            }
        }
    }
}

pub struct BufferBumpAllocator {
    cursor: u32,
}

impl BufferBumpAllocator {
    pub fn new() -> Self {
        BufferBumpAllocator { cursor: 0 }
    }

    pub fn push(&mut self, n: usize) -> BufferRange {
        let range = BufferRange(self.cursor, self.cursor + n as u32);
        self.cursor = range.1;

        range
    }

    pub fn len(&self) -> u32 {
        self.cursor
    }

    pub fn clear(&mut self) {
        self.cursor = 0;
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct BufferRange(pub u32, pub u32);
impl BufferRange {
    pub fn start(&self) -> u32 { self.0 }
    pub fn is_empty(&self) -> bool { self.0 >= self.1 }
    pub fn to_u32(&self) -> Range<u32> { self.0 .. self.1 }
    pub fn byte_range<Ty>(&self) -> Range<u64> {
        let s = std::mem::size_of::<Ty>() as u64;
        self.0 as u64 * s .. self.1 as u64 * s
    }
    pub fn byte_offset<Ty>(&self) -> u64 {
        self.0 as u64 * std::mem::size_of::<Ty>() as u64
    }
}
