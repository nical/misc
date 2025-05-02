use std::{
    cell::RefCell,
    ops::Range,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc, Mutex,
    },
    u32,
};
use wgpu::BufferAddress;

use crate::gpu::staging_buffers::{StagingBufferId, StagingOffset};

use super::staging_buffers::StagingBufferPool;

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

    pub fn upload_multiple(
        &mut self,
        device: &wgpu::Device,
        data: &[&[u8]],
    ) -> Option<DynBufferRange> {
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
        let mut mapped = staging.handle.slice(start..end).get_mapped_range_mut();

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

fn round_up_to_power_of_two(val: u32) -> u32 {
    if val.is_power_of_two() {
        val
    } else {
        val.next_power_of_two()
    }
}

// Offset in number of items (not necessarily bytes) into a buffer.
// TODO: this is redundant with GpuStoreHandle
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GpuOffset(pub u32);
// Offset in bytes into a buffer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GpuByteOffset(pub u32);

#[derive(Debug)]
struct TransferOp {
    staging_id: StagingBufferId,
    // In bytes.
    staging_offset: StagingOffset,
    // In bytes.
    size: u32,
    // In bytes.
    dst_offset: u32,
}

pub struct TransferOps {
    ops: Vec<TransferOp>,
}

pub enum GpuStoreDescriptor {
    Texture {
        format: wgpu::TextureFormat,
        width: u32,
        label: Option<&'static str>,
    },
    Buffers {
        usages: wgpu::BufferUsages,
        default_alignment: u32,
        min_size: u32,
        max_size: u32,
        label: Option<&'static str>,
    },
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
pub struct GpuStoreResources {
    storage: Storage,
    default_alignment: u32,
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
        format: wgpu::TextureFormat,
    },
    Buffer {
        handle: Option<wgpu::Buffer>,
        allocated_size: u32,
        min_size: u32,
        max_size: u32,
        usage: wgpu::BufferUsages,
    },
}

impl GpuStoreResources {
    pub fn new(desc: &GpuStoreDescriptor) -> Self {
        match desc {
            GpuStoreDescriptor::Texture {
                format,
                width,
                label,
            } => Self::new_texture(*format, *width, *label),
            GpuStoreDescriptor::Buffers {
                usages,
                default_alignment,
                min_size,
                max_size,
                label,
            } => Self::new_buffers(*usages, *default_alignment, *min_size, *max_size, *label),
        }
    }

    fn new_texture(format: wgpu::TextureFormat, width: u32, label: Option<&'static str>) -> Self {
        let bytes_per_px: u32 = match format {
            wgpu::TextureFormat::Rgba32Float => 16,
            wgpu::TextureFormat::Rgba8Uint
            | wgpu::TextureFormat::Rgba8Unorm
            | wgpu::TextureFormat::Rgba8UnormSrgb
            | wgpu::TextureFormat::Rgba8Sint
            | wgpu::TextureFormat::R32Float => 4,
            _ => unimplemented!(),
        };

        GpuStoreResources {
            storage: Storage::Texture {
                handle: None,
                view: None,
                rows: 0,
                width,
                bytes_per_px,
                format,
            },
            chunk_size: width * bytes_per_px,
            default_alignment: bytes_per_px,
            epoch: 0,
            label,
            next_gpu_offset: Arc::new(AtomicU32::new(0)),
        }
    }

    fn new_buffers(
        usage: wgpu::BufferUsages,
        default_alignment: u32,
        min_size: u32,
        max_size: u32,
        label: Option<&'static str>,
    ) -> Self {
        GpuStoreResources {
            storage: Storage::Buffer {
                handle: None,
                allocated_size: 0,
                min_size,
                max_size,
                usage: usage | wgpu::BufferUsages::COPY_DST,
            },
            default_alignment,
            chunk_size: 8192,
            epoch: 0,
            label,
            next_gpu_offset: Arc::new(AtomicU32::new(0)),
        }
    }

    pub fn allocate(&mut self, size: u32, device: &wgpu::Device) {
        match &mut self.storage {
            Storage::Texture {
                handle,
                view,
                rows,
                width,
                format,
                bytes_per_px,
                ..
            } => {
                let height = size / (*width * *bytes_per_px);

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
                *rows = height;
            }
            Storage::Buffer {
                handle,
                allocated_size,
                min_size,
                usage,
                ..
            } => {
                let usage = *usage;
                let size = size.min(*min_size);
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: self.label,
                    size: size as u64,
                    usage,
                    mapped_at_creation: false,
                });
                *handle = Some(buffer);
                *allocated_size = size;
            }
        }
    }

    pub fn begin_frame(&self, staging_buffers: Arc<Mutex<StagingBufferPool>>) -> GpuStore {
        debug_assert!(Arc::strong_count(&self.next_gpu_offset) == 1);
        self.next_gpu_offset.store(0, Ordering::Release);

        GpuStore {
            chunk_size: self.chunk_size,
            default_alignment: self.default_alignment,
            ops: RefCell::new(Vec::new()),
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
        if total_size_bytes == 0 {
            return;
        }

        match &mut self.storage {
            Storage::Texture {
                handle,
                view,
                rows,
                width,
                bytes_per_px,
                format,
            } => {
                let bytes_per_row = *width * *bytes_per_px;
                let cap = *rows * bytes_per_row;
                if cap < total_size_bytes {
                    let height = round_up_to_power_of_two(total_size_bytes / bytes_per_row + 1);
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
                        let first_row = op.dst_offset / bytes_per_row;
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
            Storage::Buffer {
                handle,
                allocated_size,
                min_size,
                usage,
                ..
            } => {
                if *allocated_size < total_size_bytes {
                    let size = total_size_bytes.max(*min_size);
                    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: self.label,
                        size: size as u64,
                        usage: *usage,
                        mapped_at_creation: false,
                    });
                    *handle = Some(buffer);
                    *allocated_size = size;
                }

                let dst = handle.as_ref().unwrap();
                for ops in transfer_ops {
                    for op in &ops.ops {
                        let staging_buffer = staging_buffers.get_handle(op.staging_id);
                        encoder.copy_buffer_to_buffer(
                            staging_buffer,
                            op.staging_offset.0 as u64,
                            dst,
                            op.dst_offset as u64,
                            op.size as u64,
                        );
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
            Storage::Texture { view, .. } => view.as_ref(),
            _ => None,
        }
    }

    pub fn as_buffer(&self) -> Option<&wgpu::Buffer> {
        match &self.storage {
            Storage::Buffer { handle, .. } => handle.as_ref(),
            _ => None,
        }
    }

    pub fn as_bind_group_entry(&self, binding: u32) -> wgpu::BindGroupEntry {
        wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::TextureView(self.as_texture_view().unwrap()),
        }
    }
}

// TODO: GpuStore isn't a very good name for what this does.

// TODO: Add a reserve(size) API to GpuStoreWriter and GpuStreamWriter that
// asks for a larger staging buffer (and ensure in the case of GpuStore that
// the push within the reseved space will be contiguous).

// There can be multiple writers for a single store (for example to write multiple
// vertex streams into the same resource). Each writer will have its own staging
// buffer chunk.

// TODO: if each writer manages its own staging buffer chunks, then a lot of
// space will be wasted as they are rather short lived. So there needs to be
// a way to recycle chunks that have enough free space when a writer is dropped.
// It could be done in by the staging buffer pool, or using a worker-local
// pool of available chunks.

// Right now GpuStore is clonable and there is one per worker. This allows the
// TransferOp verctor to be accessible to writers without a mutex and maybe avoids
// lifetimes in the worker data. would it be better to have a single non-clonable one
// with a mutex aorund the transfer ops?

// TODO: either:
// - A GpuStore could map to a single resource and offsets are relative to the start of
//   the resource. It's the simplest approach, but it means that unrelated users must
//   either chose to be in different resources, or fit in the single common resource.
// - A GpuStore could map to multiple resources but just like with GpuStreams, an ID
//   could be provided to ensure that all pushes for the same ID go to the same resource.
//   The offsets would then be relative to a per-ID base offset which could be either
//   passed to the shader or in the case of vertex buffers, the buffers could be bound at
//   an offset. It looks like doing it at draw/binding time would not work for textures and
//   non-vertex buffers, but is the only way to make it work for vertex buffers.
//   This looks like the powerful but complicated approach.
// For now I'm going with the first (simpler) approach.

/// Used to push data to the GPU at known offsets in the destination resource.
/// For example pattern parameters or vertices.
///
/// GpuStore does not guarantee that all consecutive pushes are contiguous, but
/// consecutive pushes on the same writer tend to be contiguous because each writer
/// manages its own staging buffer chunk.
pub struct GpuStore {
    ops: RefCell<Vec<TransferOp>>,
    // In bytes.
    chunk_size: u32,
    // In bytes.
    default_alignment: u32,
    // In bytes.
    next_gpu_offset: Arc<AtomicU32>,
    staging_buffers: Arc<Mutex<StagingBufferPool>>,
}

impl GpuStore {
    pub fn write(&self) -> GpuStoreWriter {
        self.write_with_alignment(self.default_alignment)
    }

    pub fn write_items<T>(&self) -> GpuStoreWriter {
        let size = std::mem::size_of::<T>() as u32;
        debug_assert!(size.is_power_of_two());
        self.write_with_alignment(size)
    }

    pub fn write_with_alignment(&self, item_size: u32) -> GpuStoreWriter {
        debug_assert!(item_size.is_power_of_two());
        let offset_shift = item_size.trailing_zeros();
        let align_mask = (item_size - 1) as usize;

        GpuStoreWriter {
            store: self,
            chunk_start: std::ptr::null_mut(),
            chunk_gpu_offset: GpuOffset(0),
            cursor_gpu_offset: GpuOffset(0),
            // Initialze to chunk_size so that it triggers flush
            // at the first push.
            staging_local_offset: self.chunk_size,
            chunk_size: self.chunk_size,
            staging_buffer_id: StagingBufferId(0),
            staging_buffer_offset: StagingOffset(0),
            align_mask,
            offset_shift,
            pushed_bytes: 0,
            next_gpu_offset: self.next_gpu_offset.clone(),
            staging_buffers: self.staging_buffers.clone(),
        }
    }

    pub(crate) fn finish(&mut self) -> TransferOps {
        TransferOps {
            ops: std::mem::take(&mut self.ops.borrow_mut()),
        }
    }
}

unsafe impl Send for GpuStore {}

impl Clone for GpuStore {
    fn clone(&self) -> Self {
        GpuStore {
            ops: RefCell::new(Vec::new()),
            chunk_size: self.chunk_size,
            default_alignment: self.default_alignment,
            next_gpu_offset: self.next_gpu_offset.clone(),
            staging_buffers: self.staging_buffers.clone(),
        }
    }
}

// Associated with a specific resource.
pub struct GpuStoreWriter<'l> {
    store: &'l GpuStore,

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

    //
    pushed_bytes: u32,

    // In bytes
    next_gpu_offset: Arc<AtomicU32>,
    staging_buffers: Arc<Mutex<StagingBufferPool>>,
}

impl<'l> GpuStoreWriter<'l> {
    pub fn push_bytes(&mut self, data: &[u8]) -> GpuStoreHandle {
        self.pushed_bytes += data.len() as u32;
        if data.len() > self.chunk_size as usize {
            return self.push_bytes_large(data);
        }

        let aligned_size = (data.len() + self.align_mask) & !self.align_mask;
        let mut new_local_offset = self.staging_local_offset + aligned_size as u32;

        if new_local_offset > self.chunk_size {
            self.replace_staging_buffer();
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

    pub fn push_slice<T: bytemuck::Pod>(&mut self, data: &[T]) -> GpuStoreHandle {
        self.push_bytes(bytemuck::cast_slice(data))
    }

    pub fn push<T: bytemuck::Pod>(&mut self, data: T) -> GpuStoreHandle {
        self.push_bytes(bytemuck::cast_slice(&[data]))
    }

    fn push_bytes_large(&mut self, data: &[u8]) -> GpuStoreHandle {
        self.flush_staging_buffer();
        self.chunk_start = std::ptr::null_mut();
        self.staging_buffer_offset = StagingOffset(0);
        self.staging_local_offset = self.chunk_size;

        let gpu_byte_offset = self
            .next_gpu_offset
            .fetch_add(data.len() as u32, Ordering::Relaxed);

        let mut src_offset = 0;
        let mut dst_offset = gpu_byte_offset;
        loop {
            let chunk = {
                self.store
                    .staging_buffers
                    .lock()
                    .unwrap()
                    .get_mapped_chunk(self.chunk_size)
            };

            let len = (self.chunk_size as usize).min(data.len() - src_offset);
            let src = &data[src_offset..(src_offset + len)];
            unsafe {
                let dst = std::slice::from_raw_parts_mut(chunk.ptr, len);
                dst.copy_from_slice(src);
            }

            // If we are writing to the staing buffer chunk directly after the
            // previous one, grow the previus copy op instead of pushing a new
            // one. This reduces the number of buffer copy operations submitted
            // to the GPU.
            let mut ops = self.store.ops.borrow_mut();
            let mut merged = false;
            if let Some(last) = ops.last_mut() {
                if last.staging_id == chunk.id
                    && (last.staging_offset.0 + last.size) == chunk.offset.0
                {
                    merged = true;
                    last.size += len as u32;
                }
            }

            if !merged {
                ops.push(TransferOp {
                    staging_id: chunk.id,
                    staging_offset: chunk.offset,
                    size: len as u32,
                    dst_offset: dst_offset,
                });
            }

            src_offset += len;
            dst_offset += len as u32;

            if src_offset >= data.len() {
                break;
            }
        }

        GpuStoreHandle(gpu_byte_offset >> self.offset_shift)
    }

    // Note: this does not reset the offsets and pointers into
    // the staging buffer.
    fn flush_staging_buffer(&mut self) {
        let size = (self.cursor_gpu_offset.0 - self.chunk_gpu_offset.0) << self.offset_shift;
        if size > 0 {
            self.store.ops.borrow_mut().push(TransferOp {
                staging_id: self.staging_buffer_id,
                staging_offset: self.staging_buffer_offset,
                size,
                dst_offset: self.chunk_gpu_offset.0 << self.offset_shift,
            });
        }
    }

    #[cold]
    fn replace_staging_buffer(&mut self) {
        self.flush_staging_buffer();

        let chunk = self
            .store
            .staging_buffers
            .lock()
            .unwrap()
            .get_mapped_chunk(self.chunk_size);

        //println!("GpuStoreWriter::flush_buffer ptr:{:?} size:{:?} id:{:?} offset:{:?}", chunk.ptr, chunk.size, chunk.id, chunk.offset);

        self.staging_buffer_id = chunk.id;
        self.staging_buffer_offset = chunk.offset;
        self.chunk_start = chunk.ptr;
        let gpu_byte_offset = self
            .next_gpu_offset
            .fetch_add(self.chunk_size, Ordering::Relaxed);
        self.chunk_gpu_offset = GpuOffset(gpu_byte_offset >> self.offset_shift);
        self.cursor_gpu_offset = self.chunk_gpu_offset;
        self.staging_local_offset = 0;
    }

    pub fn pushed_bytes(&self) -> u32 {
        self.pushed_bytes
    }

    pub fn pushed_items(&self) -> u32 {
        self.pushed_bytes >> self.offset_shift
    }
}

impl<'l> Drop for GpuStoreWriter<'l> {
    fn drop(&mut self) {
        self.flush_staging_buffer();
    }
}

impl<'l> Clone for GpuStoreWriter<'l> {
    fn clone(&self) -> Self {
        GpuStoreWriter {
            store: self.store,
            chunk_start: std::ptr::null_mut(),
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
            next_gpu_offset: self.next_gpu_offset.clone(),
            staging_buffers: self.staging_buffers.clone(),
            pushed_bytes: 0,
        }
    }
}

#[test]
fn gpu_store_simple() {
    let staging_buffers = Arc::new(Mutex::new(StagingBufferPool::new_for_testing(1024 * 32)));

    let resource = GpuStoreResources::new(&GpuStoreDescriptor::Texture {
        format: wgpu::TextureFormat::Rgba8Unorm,
        width: 128,
        label: Some("gpu store"),
    });

    let store = resource.begin_frame(staging_buffers.clone());

    let mut stores = Vec::new();
    let mut writers = Vec::new();
    let mut expected = Vec::new();
    for _ in 0..4 {
        stores.push(store.clone());
        expected.push((0u8, 0u8));
    }
    for store in &stores {
        writers.push(store.write());
    }

    for c in 0..16 {
        for idx in 0..4 {
            //println!("\nwrite worker:{idx:?} run {c:?}");
            let writer = &mut writers[idx];
            let a = idx as u8;

            for b in 0..=255 {
                writer.push_bytes(&[a, b, c, 0, 1, 2, 3, 4]);
            }
        }
    }

    let pool = staging_buffers.lock().unwrap();
    for idx in 0..4 {
        let store = &stores[idx];
        let expected = &mut expected[idx];
        //println!("worker {idx:?}:");
        for op in &*store.ops.borrow() {
            let buf = &pool.cpu_buffers[op.staging_id.0 as usize];
            let start = op.staging_offset.0 as usize;
            let chunk = &buf[start..start + op.size as usize];
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
