use std::{ops::Range, sync::{atomic::{AtomicU32, Ordering}, Arc, Mutex}, u32};
use wgpu::BufferAddress;

use crate::units::SurfaceIntSize;

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
    // This vector is allocated and potentially grown each frame.
    // It would be worth switching to a list of fixed-size chunks that are
    // copied separately.
    data: Vec<f32>,
    offset: usize,
}

impl GpuStore {
    pub fn new(h: u32) -> Self {
        let size = FLOATS_PER_ROW * h as usize;

        GpuStore {
            offset: 0,
            data: vec![0.0; size],
        }
    }

    pub fn push(&mut self, data: &[f32]) -> GpuStoreHandle {
        let size = (data.len() + 3) & !3;
        if self.data.len() < self.offset + size {
            self.data.resize(self.data.len() * 2, 0.0);
            println!("Growing the gpu store CPU buffer to {:?}", self.data.len())
        }

        self.data[self.offset..self.offset + data.len()].copy_from_slice(data);

        let handle = GpuStoreHandle(self.offset as u32 / 4);
        self.offset += size;

        return handle;
    }

    pub fn clear(&mut self) {
        self.offset = 0;
    }

    pub fn upload(&self, device: &wgpu::Device, queue: &wgpu::Queue, resources: &mut GpuStoreResources) {
        if self.offset == 0 {
            return;
        }

        let w = 4 * GPU_STORE_WIDTH as usize;
        let rows = self.offset / w + if self.offset % w == 0 { 0 } else { 1 };

        if rows > resources.rows as usize {
            resources.epoch += 1;
            let height = self.data.len() / FLOATS_PER_ROW + 1;
            resources.texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gpu store"),
                size: wgpu::Extent3d {
                    width: GPU_STORE_WIDTH,
                    height: height as u32,
                    depth_or_array_layers: 1,
                },
                sample_count: 1,
                mip_level_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[wgpu::TextureFormat::Rgba32Float],
            });

            resources.view = resources.texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("gpu store"),
                format: Some(wgpu::TextureFormat::Rgba32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                base_array_layer: 0,
                mip_level_count: Some(1),
                array_layer_count: Some(1),
                usage: None,
            });
        }

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &resources.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&self.data[..(rows * w)]),
            wgpu::TexelCopyBufferLayout {
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

    pub fn texture_size(&self) -> SurfaceIntSize {
        let height = self.data.len() / FLOATS_PER_ROW + 1;
        SurfaceIntSize::new(GPU_STORE_WIDTH as i32, height as i32)
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
}

pub struct GpuStoreResources {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub rows: i32,
    epoch: u32,
}

impl GpuStoreResources {
    pub fn new(device: &wgpu::Device) -> Self {
        let width = GPU_STORE_WIDTH;
        let height = 512;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gpu store"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            sample_count: 1,
            mip_level_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[wgpu::TextureFormat::Rgba32Float],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("gpu store"),
            format: Some(wgpu::TextureFormat::Rgba32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            base_array_layer: 0,
            mip_level_count: Some(1),
            array_layer_count: Some(1),
            usage: None,
        });

        GpuStoreResources {
            texture,
            view,
            rows: height as i32,
            epoch: 0,
        }
    }

    /// The epoch changes when the texture is reallocated.
    pub fn epoch(&self) -> u32 {
        self.epoch
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
        self.upload_multiple(device, &[data])
    }

    pub fn upload_multiple(&mut self, device: &wgpu::Device, data: &[&[u8]]) -> Option<DynBufferRange> {
        let len: u32 = data.iter().map(|data| data.len() as u32).sum();
        if len == 0 {
            return None;
        }

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
        let end = start + len as BufferAddress;
        let mut mapped = staging
            .handle
            .slice(start..end)
            .get_mapped_range_mut();

        let mut offset = 0;
        for chunk in data {
            if chunk.is_empty() {
                continue;
            }
            let len = chunk.len();
            let end = offset + len;
            mapped[offset..end].copy_from_slice(chunk);
            offset = end;
        }

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



// Offset in number of items (not necessarily bytes) into a buffer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GpuOffset(pub u32);
// Offset in bytes into a buffer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GpuByteOffset(pub u32);
/// Offset in bytes into a staging buffer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct StagingOffset(pub u32);
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct StagingBufferId(pub u32);

struct TransferOp {
    staging_id: StagingBufferId,
    // In bytes.
    staging_offset: StagingOffset,
    // In bytes.
    size: u32,
    // in items.
    gpu_offset: GpuOffset,
}


pub struct GpuStore2 {
    align_mask: u32,
    offset_shift: u32,
    chunk_size: u32,

    resource: GpuStoreResource,

    next_gpu_offset: AtomicU32,

    label: &'static str,
}

pub enum GpuStoreResource {
    Texture {
        handle: Option<wgpu::Texture>,
        view: Option<wgpu::TextureView>,
        rows: u32,
        width: u32,
        bytes_per_px: u32,
        format: wgpu::TextureFormat
    }
}

impl GpuStore2 {
    pub fn new_texture(
        item_size: u32,
        format: wgpu::TextureFormat,
        items_per_row: u32,
        initial_row_count: u32,
        label: &'static str,
        device: Option<&wgpu::Device>,
    ) -> Self {
        assert!(item_size.is_power_of_two());
        let item_alignment = item_size;
        let align_mask = item_alignment - 1;
        let offset_shift = item_size.trailing_zeros();

        let bytes_per_px = match format {
            wgpu::TextureFormat::Rgba32Float => 16,
            wgpu::TextureFormat::Rgba8Uint
            | wgpu::TextureFormat::Rgba8Unorm
            | wgpu::TextureFormat::Rgba8UnormSrgb
            | wgpu::TextureFormat::Rgba8Sint
            | wgpu::TextureFormat::R32Float => 4,
            _ => unimplemented!(),
        };

        let width = (items_per_row * item_size) / bytes_per_px;
        let height = initial_row_count;

        let (handle, view) = if let Some(device) = device {
            let handle = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                sample_count: 1,
                mip_level_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

            let view = handle.create_view(&wgpu::TextureViewDescriptor {
                label: Some(label),
                format: Some(format),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                base_array_layer: 0,
                mip_level_count: Some(1),
                array_layer_count: Some(1),
                usage: None,
            });

            (Some(handle), Some(view))
        } else {
            (None, None)
        };

        GpuStore2 {
            align_mask,
            offset_shift,
            chunk_size: items_per_row * item_size,
            next_gpu_offset: AtomicU32::new(0),
            resource: GpuStoreResource::Texture {
                handle,
                view,
                rows: 0,
                width,
                bytes_per_px,
                format,
            },
            label,
        }
    }

    pub fn worker(&self) -> GpuStoreWorker {
        let ptr = std::ptr::null_mut();
        GpuStoreWorker {
            chunk_start: ptr,
            chunk_gpu_offset: GpuOffset(0),
            cursor_gpu_offset: GpuOffset(0),
            // Initialze to chunk_size so that it triggers flush
            // at the first push.
            staging_local_offset: self.chunk_size,
            chunk_size: self.chunk_size,
            staging_buffer_id: StagingBufferId(0),
            staging_buffer_offset: StagingOffset(0),
            align_mask: self.align_mask as usize,
            offset_shift: self.offset_shift,
            transfer_ops: Vec::new(),
            next_gpu_offset: &self.next_gpu_offset,
        }
    }

    pub fn upload(
        &mut self,
        workers: &[GpuStoreWorker],
        staging_buffers: &StagingBufferPool,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let total_size_bytes = self.next_gpu_offset.load(Ordering::Acquire);

        match &mut self.resource {
            GpuStoreResource::Texture { handle, view, rows, width, bytes_per_px, format } => {
                let bytes_per_row = *width * *bytes_per_px;
                let cap = *rows * bytes_per_row;
                if cap < total_size_bytes {
                    let height = total_size_bytes / bytes_per_row + 1;
                    let texture = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some(self.label),
                        size: wgpu::Extent3d {
                            width: *width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        sample_count: 1,
                        mip_level_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: *format,
                        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    });

                    *view = Some(texture.create_view(&wgpu::TextureViewDescriptor {
                        label: Some(self.label),
                        format: Some(*format),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        aspect: wgpu::TextureAspect::All,
                        base_mip_level: 0,
                        base_array_layer: 0,
                        mip_level_count: Some(1),
                        array_layer_count: Some(1),
                        usage: None,
                    }));
                    *handle = Some(texture);
                }

                let texture = handle.as_ref().unwrap();
                for worker in workers {
                    for ops in &worker.transfer_ops {
                        let staging_buffer = staging_buffers.get_handle(ops.staging_id);
                        let first_row = (ops.gpu_offset.0 << self.offset_shift) / bytes_per_row;
                        let num_rows = ops.size / bytes_per_row;
                        encoder.copy_buffer_to_texture(
                            wgpu::TexelCopyBufferInfo {
                                buffer: staging_buffer,
                                layout: wgpu::TexelCopyBufferLayout {
                                    offset: ops.staging_offset.0 as u64,
                                    bytes_per_row: Some(bytes_per_row),
                                    rows_per_image: Some(num_rows),
                                },
                            },
                            wgpu::TexelCopyTextureInfo {
                                texture,
                                mip_level: 0,
                                origin: wgpu::Origin3d {
                                    x: 0,
                                    y: first_row,
                                    z: 0,
                                },
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::Extent3d {
                                width: *width,
                                height: num_rows,
                                depth_or_array_layers: 1,
                            },
                        )
                    }
                }
            }
        }
    }
}

// Associated with a specific resource.
// Pushes data into a staging buffer chunk provided by `Uploader`
pub struct GpuStoreWorker<'l> {
    chunk_start: *mut u8,
    // Offset of the current chunk in the destination buffer (not necessarily in bytes)
    chunk_gpu_offset: GpuOffset,
    // Offset of the cursor in the destination buffer (not necessarily in bytes)
    cursor_gpu_offset: GpuOffset,
    // Offset of the cursor relative to the beginning of the current chunk.
    staging_local_offset: u32,
    // In bytes
    chunk_size: u32,
    // Offset of the current chunk in bytes.
    staging_buffer_offset: StagingOffset,
    staging_buffer_id: StagingBufferId,
    align_mask: usize,
    // Apply a right shift by this amount to a destination buffer offset in bytes
    // to obtain the final offset.
    offset_shift: u32,

    transfer_ops: Vec<TransferOp>,

    // In bytes
    next_gpu_offset: &'l AtomicU32,
}

unsafe impl<'l> Send for GpuStoreWorker<'l> {}

impl<'l> GpuStoreWorker<'l> {
    pub fn push(&mut self, allocator: &mut Uploader, data: &[u8]) -> GpuOffset {
        let aligned_size = (data.len() + self.align_mask) & !self.align_mask;
        let new_local_offset = self.staging_local_offset + aligned_size as u32;

        if new_local_offset > self.chunk_size {
            self.flush_buffer(allocator);
        }

        unsafe {
            let dst_ptr = self.chunk_start.add(self.staging_local_offset as usize);
            let dst = std::slice::from_raw_parts_mut(dst_ptr, aligned_size);
            dst.copy_from_slice(data);
        }

        let address = self.cursor_gpu_offset;
        self.cursor_gpu_offset.0 += aligned_size as u32 >> self.offset_shift;

        address
    }

    #[cold]
    fn flush_buffer(&mut self, allocator: &mut Uploader) {
        let size = (self.cursor_gpu_offset.0 - self.chunk_gpu_offset.0) << self.offset_shift;
        if size > 0 {
            self.transfer_ops.push(TransferOp {
                staging_id: self.staging_buffer_id,
                staging_offset: self.staging_buffer_offset,
                size,
                gpu_offset: self.chunk_gpu_offset,
            });
        }

        let chunk = allocator.get_chunk(self.chunk_size);

        //println!("WorkerGpuStore::flush_buffer ptr:{:?} size:{:?} id:{:?} offset:{:?}", chunk.ptr, chunk.size, chunk.id, chunk.offset);

        self.staging_buffer_id = chunk.id;
        self.staging_buffer_offset = chunk.offset;
        self.chunk_start = chunk.ptr;
        let gpu_byte_offset = self.next_gpu_offset.fetch_add(self.chunk_size, Ordering::Relaxed);
        self.chunk_gpu_offset = GpuOffset(gpu_byte_offset >> self.offset_shift);
        self.cursor_gpu_offset = self.chunk_gpu_offset;
        self.staging_local_offset = 0;
    }
}

/// A bump-allocator into staging buffers, used by a single worker thread at
/// a time.
pub struct Uploader {
    current: MappedStagingBufferChunk,
    // Relative to the beginning of the current chunk.
    local_staging_offset: u32,
    pool: Arc<Mutex<StagingBufferPool>>,
}

unsafe impl Send for Uploader {}

impl Uploader {
    pub fn new(pool: Arc<Mutex<StagingBufferPool>>) -> Uploader {
        Uploader {
            current: MappedStagingBufferChunk {
                ptr: std::ptr::null_mut(),
                id: StagingBufferId(u32::MAX),
                size: 0,
                offset: StagingOffset(u32::MAX),
            },
            local_staging_offset: 0,
            pool,
        }
    }

    pub fn get_chunk(&mut self, size: u32) -> MappedStagingBufferChunk {
        let mut local_offset = self.local_staging_offset;
        self.local_staging_offset += size;

        if self.local_staging_offset > self.current.size {
            self.replace_chunk();
            self.local_staging_offset = size;
            local_offset = 0;
        }

        //println!("get_chunk current offset: {:?} local offset {:?}", self.current.offset.0, self.local_staging_offset);

        let ptr = unsafe { self.current.ptr.add(local_offset as usize) };

        MappedStagingBufferChunk {
            ptr,
            id: self.current.id,
            offset: StagingOffset(self.current.offset.0 + local_offset),
            size,
        }
    }

    #[cold]
    fn replace_chunk(&mut self) {
        self.current = self.pool.lock().unwrap().get_mapped_chunk();
    }

    pub fn staging_buffer_pool(&self) -> &Arc<Mutex<StagingBufferPool>> {
        &self.pool
    }
}

pub struct MappedStagingBuffer {
    pub ptr: *mut u8,
    // in bytes.
    pub size: u32,
    pub id: StagingBufferId,
}

pub struct MappedStagingBufferChunk {
    pub ptr: *mut u8,
    // In bytes.
    pub size: u32,
    pub id: StagingBufferId,
    pub offset: StagingOffset,
}

pub struct StagingBufferPool {
    buffer_size: u32,

    // Mapped an ready for use.
    available: Vec<wgpu::Buffer>,
    // In use this frame.
    active: Vec<wgpu::Buffer>,
    // map_aysnc has been called, waiting for the callback to resolve.
    pending: Vec<Option<wgpu::Buffer>>,

    ready_list: Arc<Mutex<Vec<(u16, bool)>>>,

    current_chunks: MappedStagingBuffer,
    current_chunks_offset: u32,
    chunk_size: u32,

    cpu_buffers: Vec<Vec<u8>>,

    // TODO: In the next wgpu version this will be a regular cloneable
    // Device handle.
    device: Option<wgpu::Device>,
}

unsafe impl Send for StagingBufferPool {}
unsafe impl Sync for StagingBufferPool {}

impl StagingBufferPool {
    pub unsafe fn new(buffer_size: u32, chunk_size: u32, device: wgpu::Device) -> Self {
        StagingBufferPool {
            buffer_size,
            available: Vec::new(),
            active: Vec::new(),
            pending: Vec::new(),
            ready_list: Arc::new(Mutex::new(Vec::new())),
            current_chunks: MappedStagingBuffer {
                ptr: std::ptr::null_mut(),
                id: StagingBufferId(u32::MAX),
                size: 0,
            },
            cpu_buffers: Vec::new(),
            current_chunks_offset: 0,
            chunk_size,
            device: Some(device),
        }
    }

    pub fn new_for_testing(buffer_size: u32, chunk_size: u32) -> Self {
        StagingBufferPool {
            buffer_size,
            available: Vec::new(),
            active: Vec::new(),
            pending: Vec::new(),
            ready_list: Arc::new(Mutex::new(Vec::new())),
            current_chunks: MappedStagingBuffer {
                ptr: std::ptr::null_mut(),
                id: StagingBufferId(u32::MAX),
                size: 0,
            },
            cpu_buffers: Vec::new(),
            current_chunks_offset: 0,
            chunk_size,
            device: None,
        }
    }

    fn device(&self) -> &wgpu::Device {
        self.device.as_ref().unwrap()
    }

    pub fn get_mapped_staging_buffer(&mut self) -> MappedStagingBuffer {
        if self.device.is_none() {
            return self.create_cpu_buffer();
        }

        let id = StagingBufferId(self.active.len() as u32);
        let buffer = if let Some(buffer) = self.available.pop() {
            buffer
        } else {
            Self::create_buffer(self.device(), self.buffer_size)
        };

        let ptr = buffer.slice(..)
            .get_mapped_range_mut()
            .as_mut_ptr();

        self.active.push(buffer);

        MappedStagingBuffer {
            ptr,
            id,
            size: self.buffer_size,
        }
    }

    fn create_cpu_buffer(&mut self) -> MappedStagingBuffer {
        let mut buffer = vec![0; self.buffer_size as usize];

        let id = StagingBufferId(self.cpu_buffers.len() as u32);
        let ptr = buffer.as_mut_ptr();
        self.cpu_buffers.push(buffer);

        MappedStagingBuffer {
            ptr,
            id,
            size: self.buffer_size,
        }
    }

    pub fn get_mapped_chunk(&mut self) -> MappedStagingBufferChunk {
        let mut offset = self.current_chunks_offset;
        self.current_chunks_offset = offset + self.chunk_size;

        if self.current_chunks_offset >= self.current_chunks.size {
            self.current_chunks = self.get_mapped_staging_buffer();
            self.current_chunks_offset = self.chunk_size;
            offset = 0;
        }

        //println!("Pool::get_mapped_chunk -> id {:?} offset {:?}", self.current_chunks.id, offset);

        let ptr = unsafe { self.current_chunks.ptr.add(offset as usize) };
        MappedStagingBufferChunk {
            ptr,
            size: self.chunk_size,
            offset: StagingOffset(offset),
            id: self.current_chunks.id,
        }
    }

    pub fn get_handle(&self, id: StagingBufferId) -> &wgpu::Buffer {
        &self.active[id.0 as usize]
    }

    fn create_buffer(device: &wgpu::Device, size: u32) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: size as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
            mapped_at_creation: true,
        })
    }

    pub fn triage_available_buffers(&mut self) {
        // Pull pending buffers that are ready and move them into the available list.
        let mut list = self.ready_list.lock().unwrap();
        for (index, success) in list.drain(..) {
            if let Some(buffer) = self.pending[index as usize].take() {
                if success {
                    self.available.push(buffer);
                }
            }
        }
    }

    /// # Safety
    ///
    /// No writes to this frame's MappedStagingBuffer can be done
    /// after this call.
    pub unsafe fn unmap_active_buffers(&mut self) {
        for buffer in &mut self.active {
            buffer.unmap();
        }
    }

    // unmap_active_buffers must be called before doing this.
    pub fn recycle_active_buffers(&mut self) {
        // Move active buffers into the pending list.
        let mut idx = 0;
        for buffer in self.active.drain(..) {
            // Find an index in the pending array.
            loop {
                if idx == self.pending.len() || self.pending[idx].is_none() {
                    break;
                }
                idx += 1;
            }
            if idx == self.pending.len() {
                self.pending.push(None);
            }

            let list = Arc::clone(&self.ready_list);
            buffer.slice(..).map_async(wgpu::MapMode::Write, move|result| {
                let mut ready = list.lock().unwrap();
                ready.push((idx as u16, result.is_ok()));
            });
        }
    }
}

#[test]
fn gpu_store_simple() {
    let staging_buffers = Arc::new(Mutex::new(
        StagingBufferPool::new_for_testing(1024 * 32, 1024 * 8)
    ));

    let store = GpuStore2::new_texture(
        8,
        wgpu::TextureFormat::Rgba8Unorm,
        128,
        128,
        "gpu store",
        None
    );

    let mut workers = Vec::new();
    let mut uploaders = Vec::new();
    let mut expected = Vec::new();
    for _ in 0..4 {
        workers.push(store.worker());
        uploaders.push(Uploader::new(staging_buffers.clone()));
        expected.push((0u8, 0u8));
    }

    for c in 0..16 {
        for idx in 0..4 {
            //println!("\nwrite worker:{idx:?} run {c:?}");
            let store = &mut workers[idx];
            let ctx = &mut uploaders[idx];
            let a = idx as u8;

            for b in 0..=255 {
                store.push(ctx, &[a, b, c, 0, 1, 2, 3, 4]);
            }
        }
    }

    let pool = staging_buffers.lock().unwrap();
    for idx in 0..4 {
        let worker = &workers[idx];
        let expected = &mut expected[idx];
        //println!("worker {idx:?}:");
        for op in &worker.transfer_ops {
            let buf = &pool.cpu_buffers[op.staging_id.0 as usize];
            let start = op.staging_offset.0 as usize;
            let chunk = &buf[start .. start + op.size as usize];
            //println!(" id: {:?} chunk {:?}..{:?}", op.staging_id, start, start + op.size as usize);
            //let mut count = 0;
            for item in chunk.chunks(8) {
                assert_eq!(item[0], idx as u8);
                assert_eq!(item[1], expected.0);
                assert_eq!(item[2], expected.1);
                assert_eq!(item[3], 0);
                assert_eq!(item[4], 1);
                assert_eq!(item[5], 2);
                assert_eq!(item[6], 3);
                assert_eq!(item[7], 4);
                if expected.0 == 255 {
                    expected.0 = 0;
                    expected.1 += 1;
                } else {
                    expected.0 += 1;
                }
                //count += 1;
            }
            //println!(" read {count} items");
        }
    }
}
