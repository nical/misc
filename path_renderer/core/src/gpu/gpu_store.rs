use std::{ops::Range, sync::{atomic::{AtomicU32, Ordering}, Arc, Mutex}, u32};
use wgpu::BufferAddress;

const GPU_STORE_WIDTH: u32 = 2048;

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
// TODO: this is redundant with GpuStoreHandle
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

#[derive(Debug)]
struct TransferOp {
    staging_id: StagingBufferId,
    // In bytes.
    staging_offset: StagingOffset,
    // In bytes.
    size: u32,
    // in items.
    gpu_offset: GpuOffset,
}

pub struct TransferOps {
    ops: Vec<TransferOp>,
}

pub enum GpuStoreDescriptor {
    Texture {
        format: wgpu::TextureFormat,
        width: u32,
        label: Option<&'static str>,
    }
}

impl GpuStoreDescriptor {
    pub fn rgba32_float_texture() -> Self {
        GpuStoreDescriptor::Texture {
            format: wgpu::TextureFormat::Rgba32Float,
            width: GPU_STORE_WIDTH,
            label: Some(&"GPU store"),
        }
    }

    pub fn rgba8_unorm_texture() -> Self {
        GpuStoreDescriptor::Texture {
            format: wgpu::TextureFormat::Rgba8Uint,
            width: GPU_STORE_WIDTH,
            label: Some(&"GPU store"),
        }
    }
}

impl Default for GpuStoreDescriptor {
    fn default() -> Self {
        GpuStoreDescriptor::rgba32_float_texture()
    }
}


// Lives in the instance.
pub struct GpuStoreResource {
    storage: Storage,
    align_mask: u32,
    offset_shift: u32,
    chunk_size: u32,
    epoch: u32,
    label: Option<&'static str>,

    next_gpu_offset: Arc<AtomicU32>,
}

enum Storage {
    Texture {
        handle: Option<wgpu::Texture>,
        view: Option<wgpu::TextureView>,
        rows: u32,
        width: u32,
        bytes_per_px: u32,
        format: wgpu::TextureFormat
    }
}

impl GpuStoreResource {
    pub fn new(desc: &GpuStoreDescriptor) -> Self {
        match desc {
            GpuStoreDescriptor::Texture { format, width, label } => {
                Self::new_texture(*format, *width, *label)
            }
        }
    }

    fn new_texture(
        format: wgpu::TextureFormat,
        width: u32,
        label: Option<&'static str>,
    ) -> Self {
        let bytes_per_px: u32 = match format {
            wgpu::TextureFormat::Rgba32Float => 16,
            wgpu::TextureFormat::Rgba8Uint
            | wgpu::TextureFormat::Rgba8Unorm
            | wgpu::TextureFormat::Rgba8UnormSrgb
            | wgpu::TextureFormat::Rgba8Sint
            | wgpu::TextureFormat::R32Float => 4,
            _ => unimplemented!(),
        };

        let item_alignment = bytes_per_px;
        let align_mask = item_alignment - 1;
        let offset_shift = bytes_per_px.trailing_zeros();


        GpuStoreResource {
            storage: Storage::Texture {
                handle: None,
                view: None,
                rows: 0,
                width,
                bytes_per_px,
                format,
            },
            align_mask,
            offset_shift,
            chunk_size: width * bytes_per_px,
            epoch: 0,
            label,
            next_gpu_offset: Arc::new(AtomicU32::new(0)),
        }
    }

    pub fn allocate(&mut self, row_count: u32, device: &wgpu::Device) {
        let height = row_count;

        match &mut self.storage {
            Storage::Texture { handle, view, rows, width, format, .. } => {
                let tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: self.label,
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

                *view = Some(tex.create_view(&wgpu::TextureViewDescriptor {
                    label: self.label,
                    format: None,
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    base_array_layer: 0,
                    mip_level_count: Some(1),
                    array_layer_count: Some(1),
                    usage: None,
                }));
                self.epoch += 1;
                *handle = Some(tex);
                *rows = row_count;
            }
        }
    }

    pub fn begin_frame(&self, staging_buffers: Arc<Mutex<StagingBufferPool>>) -> GpuStoreWriter {

        debug_assert!(Arc::strong_count(&self.next_gpu_offset) == 1);
        self.next_gpu_offset.store(0, Ordering::Release);

        GpuStoreWriter {
            chunk_start: std::ptr::null_mut(),
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
            next_gpu_offset: self.next_gpu_offset.clone(),
            staging_buffers,
        }
    }

    pub fn upload(
        &mut self,
        transfer_ops: &[TransferOps],
        staging_buffers: &StagingBufferPool,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let total_size_bytes = self.next_gpu_offset.load(Ordering::Acquire);

        match &mut self.storage {
            Storage::Texture { handle, view, rows, width, bytes_per_px, format } => {
                let bytes_per_row = *width * *bytes_per_px;
                let cap = *rows * bytes_per_row;
                if cap < total_size_bytes {
                    let height = total_size_bytes / bytes_per_row + 1;
                    let texture = device.create_texture(&wgpu::TextureDescriptor {
                        label: self.label,
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
                        label: self.label,
                        format: Some(*format),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        aspect: wgpu::TextureAspect::All,
                        base_mip_level: 0,
                        base_array_layer: 0,
                        mip_level_count: Some(1),
                        array_layer_count: Some(1),
                        usage: None,
                    }));
                    *rows = height;
                    *handle = Some(texture);
                    self.epoch += 1;
                }

                let texture = handle.as_ref().unwrap();
                for ops in transfer_ops {
                    for op in &ops.ops {
                        let staging_buffer = staging_buffers.get_handle(op.staging_id);
                        let first_row = (op.gpu_offset.0 << self.offset_shift) / bytes_per_row;
                        let num_rows = op.size / bytes_per_row + (op.size % bytes_per_row).min(1);
                        encoder.copy_buffer_to_texture(
                            wgpu::TexelCopyBufferInfo {
                                buffer: staging_buffer,
                                layout: wgpu::TexelCopyBufferLayout {
                                    offset: op.staging_offset.0 as u64,
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

    /// The epoch changes when the texture is reallocated.
    pub fn epoch(&self) -> u32 {
        self.epoch
    }

    pub fn as_texture_view(&self) -> Option<&wgpu::TextureView> {
        match &self.storage {
            Storage::Texture { view, .. } => view.as_ref()
        }
    }

    pub fn as_bind_group_entry(&self, binding: u32) -> wgpu::BindGroupEntry {
        wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::TextureView(
                self.as_texture_view().unwrap()
            ),
        }
    }
}

// Associated with a specific resource.
// Pushes data into a staging buffer chunk provided by `Uploader`
pub struct GpuStoreWriter {
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
    next_gpu_offset: Arc<AtomicU32>,
    staging_buffers: Arc<Mutex<StagingBufferPool>>,
}

unsafe impl<'l> Send for GpuStoreWriter {}

impl GpuStoreWriter {
    pub fn push_bytes(&mut self, data: &[u8]) -> GpuStoreHandle {
        let aligned_size = (data.len() + self.align_mask) & !self.align_mask;
        let mut new_local_offset = self.staging_local_offset + aligned_size as u32;

        if new_local_offset > self.chunk_size {
            self.flush_buffer();
            new_local_offset = self.staging_local_offset + aligned_size as u32;
        }

        unsafe {
            let dst_ptr = self.chunk_start.add(self.staging_local_offset as usize);
            let dst = std::slice::from_raw_parts_mut(dst_ptr, data.len());
            dst.copy_from_slice(data);
        }

        let address = self.cursor_gpu_offset;
        self.cursor_gpu_offset.0 += aligned_size as u32 >> self.offset_shift;
        self.staging_local_offset = new_local_offset;

        GpuStoreHandle(address.0)
    }

    pub fn push_f32(&mut self, data: &[f32]) -> GpuStoreHandle {
        self.push_bytes(bytemuck::cast_slice(data))
    }

    #[cold]
    fn flush_buffer(&mut self) {
        let size = (self.cursor_gpu_offset.0 - self.chunk_gpu_offset.0) << self.offset_shift;
        if size > 0 {
            self.transfer_ops.push(TransferOp {
                staging_id: self.staging_buffer_id,
                staging_offset: self.staging_buffer_offset,
                size,
                gpu_offset: self.chunk_gpu_offset,
            });
        }

        let chunk = self.staging_buffers
            .lock()
            .unwrap()
            .get_mapped_chunk(self.chunk_size);

        //println!("GpuStoreWriter::flush_buffer ptr:{:?} size:{:?} id:{:?} offset:{:?}", chunk.ptr, chunk.size, chunk.id, chunk.offset);

        self.staging_buffer_id = chunk.id;
        self.staging_buffer_offset = chunk.offset;
        self.chunk_start = chunk.ptr;
        let gpu_byte_offset = self.next_gpu_offset.fetch_add(self.chunk_size, Ordering::Relaxed);
        self.chunk_gpu_offset = GpuOffset(gpu_byte_offset >> self.offset_shift);
        self.cursor_gpu_offset = self.chunk_gpu_offset;
        self.staging_local_offset = 0;
    }

    pub(crate) fn finish(&mut self) -> TransferOps {
        self.flush_buffer();
        TransferOps {
            ops: std::mem::take(&mut self.transfer_ops)
        }
    }
}

impl Clone for GpuStoreWriter {
    fn clone(&self) -> GpuStoreWriter {
        let ptr = std::ptr::null_mut();
        GpuStoreWriter {
            chunk_start: ptr,
            chunk_gpu_offset: GpuOffset(0),
            cursor_gpu_offset: GpuOffset(0),
            // Initialze to chunk_size so that it triggers flush
            // at the first push.
            staging_local_offset: self.chunk_size,
            chunk_size: self.chunk_size,
            staging_buffer_id: StagingBufferId(0),
            staging_buffer_offset: StagingOffset(0),
            align_mask: self.align_mask,
            offset_shift: self.offset_shift,
            transfer_ops: Vec::new(),
            next_gpu_offset: self.next_gpu_offset.clone(),
            staging_buffers: self.staging_buffers.clone(),
        }
    }
}

// TODO: remove the need to pass an Uploader in GpuStoreWriter::push.

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
        self.current = self.pool.lock().unwrap().get_mapped_chunk(4096);
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

    cpu_buffers: Vec<Vec<u8>>,

    device: Option<wgpu::Device>,
}

unsafe impl Send for StagingBufferPool {}
unsafe impl Sync for StagingBufferPool {}

impl StagingBufferPool {
    pub unsafe fn new(buffer_size: u32, device: wgpu::Device) -> Self {
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
            current_chunks_offset: 0,
            cpu_buffers: Vec::new(),
            device: Some(device),
        }
    }

    #[allow(unused)]
    fn new_for_testing(buffer_size: u32) -> Self {
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

    pub fn get_mapped_chunk(&mut self, size: u32) -> MappedStagingBufferChunk {
        let mut offset = self.current_chunks_offset;
        self.current_chunks_offset = offset + size;

        if self.current_chunks_offset >= self.current_chunks.size {
            self.current_chunks = self.get_mapped_staging_buffer();
            self.current_chunks_offset = size;
            offset = 0;
        }

        //println!("Pool::get_mapped_chunk -> id {:?} offset {:?}", self.current_chunks.id, offset);

        let ptr = unsafe { self.current_chunks.ptr.add(offset as usize) };
        MappedStagingBufferChunk {
            ptr,
            size,
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
        self.current_chunks = MappedStagingBuffer {
            ptr: std::ptr::null_mut(),
            id: StagingBufferId(u32::MAX),
            size: 0,
        };
        self.current_chunks_offset = 0;

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
        StagingBufferPool::new_for_testing(1024 * 32)
    ));

    let resource = GpuStoreResource::new(&GpuStoreDescriptor::Texture {
        format: wgpu::TextureFormat::Rgba8Unorm,
        width: 128,
        label: Some("gpu store"),
    });

    let store = resource.begin_frame(staging_buffers.clone());

    let mut writers = Vec::new();
    let mut expected = Vec::new();
    for _ in 0..4 {
        writers.push(store.clone());
        expected.push((0u8, 0u8));
    }

    for c in 0..16 {
        for idx in 0..4 {
            //println!("\nwrite worker:{idx:?} run {c:?}");
            let store = &mut writers[idx];
            let a = idx as u8;

            for b in 0..=255 {
                store.push_bytes(&[a, b, c, 0, 1, 2, 3, 4]);
            }
        }
    }

    let pool = staging_buffers.lock().unwrap();
    for idx in 0..4 {
        let writer = &writers[idx];
        let expected = &mut expected[idx];
        //println!("worker {idx:?}:");
        for op in &writer.transfer_ops {
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
