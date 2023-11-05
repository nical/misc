use std::ops::Range;
use wgpu::BufferAddress;

const GPU_STORE_WIDTH: u32 = 2048;
const FLOATS_PER_ROW: usize = GPU_STORE_WIDTH as usize * 4;

// Packed into 20 bits, leaving 12 bits unused so that it can be packed
// with other data in GPU instances.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct GpuStoreHandle(u32);

unsafe impl bytemuck::Pod for GpuStoreHandle {}
unsafe impl bytemuck::Zeroable for GpuStoreHandle {}

impl GpuStoreHandle {
    pub const MASK: u32 = 0xFFFFF;
    pub const INVALID: Self = GpuStoreHandle(Self::MASK);

    pub fn to_u32(self) -> u32 {
        self.0
    }
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
            view_formats: &[wgpu::TextureFormat::Rgba32Float],
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

        self.data[self.offset..self.offset + data.len()].copy_from_slice(data);

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
                view_formats: &[wgpu::TextureFormat::Rgba32Float],
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
                bytes_per_row: Some(GPU_STORE_WIDTH * 16),
                rows_per_image: Some(rows as u32),
            },
            wgpu::Extent3d {
                width: GPU_STORE_WIDTH,
                height: rows as u32,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    pub fn bind_group_layout_entry(
        &self,
        binding: u32,
        visibility: wgpu::ShaderStages,
    ) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        }
    }

    pub fn create_texture_view(&self) -> wgpu::TextureView {
        self.texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("gpu store"),
            format: Some(wgpu::TextureFormat::Rgba32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            base_array_layer: 0,
            mip_level_count: Some(1),
            array_layer_count: Some(1),
        })
    }
}

struct StagingBuffer {
    handle: wgpu::Buffer,
    size: u32,
    offset: u32,
}

struct GpuBuffer {
    handle: wgpu::Buffer,
    size: u32,
}

#[derive(Clone, Debug)]
pub struct DynBufferRange {
    pub buffer_index: u32,
    pub range: Range<u32>, // In bytes.
}

impl DynBufferRange {
    pub fn address_range(&self) -> Range<BufferAddress> {
        self.range.start as BufferAddress..self.range.end as BufferAddress
    }
}

// TODO: Dynamic store uses a few large-ish fixed size vertex buffers. Another way would be to use a single very large
// one, however we'd more easily hit limits. Or we could use some hybrid of the two: a single vertex buffer of the
// required size up to a limit and spill into another variably sized one.
// TODO: This manages its own staging buffer but we could probably centralize staging buffer management into a separate
// thing.
// TODO: recycle the staging buffers!

/// GPU buffer memory that is re-built every frame.
///
/// Typically useful for dynamic vertices.
pub struct DynamicStore {
    vbos: Vec<GpuBuffer>,
    staging: Vec<StagingBuffer>,
    buffer_size: u32,
    usage: wgpu::BufferUsages,
    label: &'static str,
}

impl DynamicStore {
    pub fn new(buffer_size: u32, usage: wgpu::BufferUsages, label: &'static str) -> Self {
        DynamicStore {
            vbos: Vec::new(),
            staging: Vec::new(),
            buffer_size,
            usage,
            label,
        }
    }

    pub fn new_vertices(buffer_size: u32) -> Self {
        Self::new(buffer_size, wgpu::BufferUsages::VERTEX, "Dynamic vertices")
    }

    // Note: Using GPU-store for float-only data is better since it can fall back to textures.
    pub fn new_storage(buffer_size: u32) -> Self {
        Self::new(
            buffer_size,
            wgpu::BufferUsages::STORAGE,
            "Dynamic storage buffer",
        )
    }

    pub fn upload(&mut self, device: &wgpu::Device, data: &[u8]) -> Option<DynBufferRange> {
        if data.is_empty() {
            return None;
        }

        let len = data.len() as u32;
        let mut selected_buffer = None;
        for (idx, buffer) in self.staging.iter().enumerate() {
            if buffer.size >= buffer.offset + len {
                selected_buffer = Some(idx);
                break;
            }
        }

        let selected_buffer = match selected_buffer {
            Some(idx) => idx,
            None => self.add_staging_buffer(device, len as BufferAddress),
        };

        let staging = &mut self.staging[selected_buffer];

        let start = staging.offset as BufferAddress;
        let end = start + data.len() as BufferAddress;
        staging
            .handle
            .slice(start..end)
            .get_mapped_range_mut()
            .copy_from_slice(data);

        let aligned_end = align(end, 64);

        // Fill the hole with zeroes in case it is write-combined memory.
        if end > aligned_end {
            staging
                .handle
                .slice(end..aligned_end)
                .get_mapped_range_mut()
                .fill(0);
        }

        staging.offset = aligned_end as u32;

        Some(DynBufferRange {
            buffer_index: selected_buffer as u32,
            range: (start as u32)..(end as u32),
        })
    }

    fn add_staging_buffer(
        &mut self,
        device: &wgpu::Device,
        requested_size: BufferAddress,
    ) -> usize {
        // If we are requesting a buffer larger than the default size, dump vbos to force re-creating one.
        // TODO: that's obviously not great, but is intended to be rare.
        if requested_size > self.buffer_size as BufferAddress
            && self.vbos.len() > self.staging.len()
        {
            self.vbos.truncate(self.staging.len());
        }

        let actual_size = align(requested_size.max(self.buffer_size as BufferAddress), 64);

        let idx = self.staging.len();

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size: actual_size,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });

        self.staging.push(StagingBuffer {
            handle: staging,
            size: self.buffer_size,
            offset: 0,
        });

        if self.vbos.len() <= idx {
            self.vbos.push(GpuBuffer {
                handle: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(self.label),
                    size: actual_size,
                    usage: self.usage | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                size: actual_size as u32,
            });
        }

        idx
    }

    pub fn unmap(&mut self, encoder: &mut wgpu::CommandEncoder) {
        for buffer in &mut self.staging {
            buffer.handle.unmap();
        }

        for (src, dst) in self.staging.iter().zip(self.vbos.iter()) {
            encoder.copy_buffer_to_buffer(
                &src.handle,
                0,
                &dst.handle,
                0,
                src.offset as BufferAddress,
            );
        }
    }

    pub fn end_frame(&mut self) {
        // TODO: recycle the staging buffers.
        self.staging.clear();
        // Throw away buffers with a special size.
        self.vbos.retain(|buffer| buffer.size == self.buffer_size);
    }

    pub fn get_buffer(&self, index: u32) -> &wgpu::Buffer {
        &self.vbos[index as usize].handle
    }

    pub fn get_buffer_slice(&self, range: &DynBufferRange) -> wgpu::BufferSlice {
        self.get_buffer(range.buffer_index)
            .slice(range.address_range())
    }
}

fn align(size: BufferAddress, alignment: BufferAddress) -> BufferAddress {
    let rem = size % alignment;
    if rem == 0 {
        size
    } else {
        size + alignment - rem
    }
}
