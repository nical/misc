use std::{
    cell::RefCell,
    ops::Range,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc, Mutex,
    },
    u32, u64,
};

use super::{StagingBufferId, StagingBufferPool, StagingOffset};

// Associated with a specific resource.

/// Used to push data to the GPU without requiring the offset of each element
/// to be known at the time of pushing them.
///
/// Ideal for index and instance buffers.
pub struct GpuStreams {
    ops: RefCell<Vec<CopyOp>>,
    chunk_size: u32,
    next_stream_id: Arc<AtomicU32>,
    staging_buffers: Arc<Mutex<StagingBufferPool>>,
}

unsafe impl<'l> Send for GpuStreams {}

impl GpuStreams {
    pub fn next_stream_id(&self) -> StreamId {
        StreamId(self.next_stream_id.fetch_add(1, Ordering::Relaxed))
    }

    pub fn write(&self, stream: StreamId, sort_key: u32) -> GpuStreamWriter {
        GpuStreamWriter {
            chunk_start: std::ptr::null_mut(),
            // Initialze to chunk_size so that it triggers flush
            // at the first push.
            staging_local_offset: self.chunk_size,
            staging_local_start_offset: self.chunk_size,
            chunk_size: self.chunk_size,
            staging_buffer_id: StagingBufferId(0),
            staging_buffer_offset: StagingOffset(0),
            current_sort_key: u64::MAX,
            chunk_idx: 0,

            streams: self,
            key: (stream.0 as u64) << 48 | sort_key as u64,
            pushed_bytes: 0,
        }
    }

    pub(crate) fn finish(&mut self) -> StreamOps {
        StreamOps {
            ops: std::mem::take(&mut self.ops.borrow_mut()),
        }
    }
}

impl Clone for GpuStreams {
    fn clone(&self) -> GpuStreams {
        GpuStreams {
            chunk_size: self.chunk_size,
            ops: RefCell::new(Vec::new()),
            staging_buffers: self.staging_buffers.clone(),
            next_stream_id: self.next_stream_id.clone(),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct StreamId(u32);

// TODO: It would be useful to be able to resolve a specific sub-stream
// of a StreamId (the sort key would become a sub-stream id) to express
// that we want two parts of a stream to be in the same buffer and have
// access to their respective ranges.

// TODO: Would it be more convenient to make the writer generic over the
// item type? It looks like each writer always writes a single type
// of thing which makes sense since they are going to be read as indices
// or instances.

pub struct GpuStreamWriter<'l> {
    chunk_start: *mut u8,
    // Offset of the cursor relative to the beginning of the current chunk, in bytes.
    staging_local_offset: u32,
    staging_local_start_offset: u32,
    // In bytes
    chunk_size: u32,
    // Offset of the current chunk in bytes.
    staging_buffer_offset: StagingOffset,
    staging_buffer_id: StagingBufferId,

    current_sort_key: u64,
    // Index of the current chunk in the stream. It is used to ensure that multiple
    // chunks within a stream remain in the same order after sorting.
    chunk_idx: u32,

    streams: &'l GpuStreams,
    key: u64,
    pushed_bytes: u32,
}

impl<'l> GpuStreamWriter<'l> {
    #[inline]
    pub fn push_slice<T>(&mut self, data: &[T]) where T: bytemuck::Pod {
        self.push_bytes_impl(self.key, bytemuck::cast_slice(data));
    }

    //
    #[inline]
    pub fn push<T>(&mut self, data: T) where T: bytemuck::Pod {
        self.push_val_impl(self.key, data);
    }

    #[inline]
    pub fn push_bytes(&mut self, data: &[u8]) {
        self.push_bytes_impl(self.key, data);
    }

    fn push_bytes_impl(&mut self, sort_key: u64, data: &[u8]) {
        self.pushed_bytes += data.len() as u32;
        let size = data.len() as u32;
        if size > self.chunk_size {
            self.push_bytes_large(sort_key, data);
            return;
        }
        let mut new_local_offset = self.staging_local_offset + size;
        // TODO: support breaking large pushes into multiple chunks!
        if new_local_offset > self.chunk_size || sort_key != self.current_sort_key {
            self.replace_staging_buffer();
            new_local_offset = self.staging_local_offset + size;
            self.current_sort_key = sort_key;
        }

        unsafe {
            let dst_ptr = self.chunk_start.add(self.staging_local_offset as usize);
            let dst = std::slice::from_raw_parts_mut(dst_ptr, data.len());
            dst.copy_from_slice(data);
        }

        self.staging_local_offset = new_local_offset;
    }

    fn push_val_impl<T>(&mut self, sort_key: u64, data: T) {
        let size_of = std::mem::size_of::<T>();
        let size = size_of as u32;
        self.pushed_bytes += size;
        let mut new_local_offset = self.staging_local_offset + size;
        if new_local_offset > self.chunk_size || sort_key != self.current_sort_key {
            self.replace_staging_buffer();
            new_local_offset = self.staging_local_offset + size;
            self.current_sort_key = sort_key;
        }

        unsafe {
            let dst_ptr = self.chunk_start.add(self.staging_local_offset as usize) as *mut T;
            std::ptr::write(dst_ptr, data);
        }

        self.staging_local_offset = new_local_offset;
    }

    #[inline(never)]
    fn push_bytes_large(&mut self, sort_key: u64, data: &[u8]) {
        self.flush_staging_buffer();
        self.chunk_start = std::ptr::null_mut();
        self.staging_buffer_offset = StagingOffset(0);
        self.staging_local_offset = self.chunk_size;
        self.staging_local_start_offset = self.chunk_size;
        let mut can_merge = self.current_sort_key == sort_key;
        self.current_sort_key = sort_key;

        let mut src_offset = 0;
        loop {
            let chunk = {
                self.streams
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
            let mut ops = self.streams.ops.borrow_mut();
            let mut merged = false;
            if let Some(last) = ops.last_mut() {
                if can_merge
                    && last.staging_id == chunk.id
                    && (last.staging_offset.0 + last.size) == chunk.offset.0
                {
                    merged = true;
                    last.size += len as u32;
                }
            }

            if !merged {
                ops.push(CopyOp {
                    staging_id: chunk.id,
                    staging_offset: chunk.offset,
                    size: len as u32,
                    sort_key: sort_key | ((self.chunk_idx as u64) << 32),
                });
            }

            can_merge = true;
            self.chunk_idx += 1;

            src_offset += len;

            if src_offset >= data.len() {
                break;
            }
        }
    }

    // Note: this does not reset the offsets and pointers into
    // the staging buffer.
    fn flush_staging_buffer(&mut self) {
        let size = self.staging_local_offset - self.staging_local_start_offset;
        if size > 0 {
            self.streams.ops.borrow_mut().push(CopyOp {
                staging_id: self.staging_buffer_id,
                staging_offset: StagingOffset(
                    self.staging_buffer_offset.0 + self.staging_local_start_offset,
                ),
                size,
                sort_key: self.current_sort_key | ((self.chunk_idx as u64) << 32),
            });
            self.chunk_idx += 1;
            self.staging_local_start_offset = self.staging_local_offset;
        }
    }

    #[cold]
    fn replace_staging_buffer(&mut self) {
        self.flush_staging_buffer();

        let chunk = self
            .streams
            .staging_buffers
            .lock()
            .unwrap()
            .get_mapped_chunk(self.chunk_size);

        //println!("GpuStoreWriter::flush_buffer ptr:{:?} size:{:?} id:{:?} offset:{:?}", chunk.ptr, chunk.size, chunk.id, chunk.offset);

        self.staging_buffer_id = chunk.id;
        self.staging_buffer_offset = chunk.offset;
        self.chunk_start = chunk.ptr;
        self.staging_local_offset = 0;
        self.staging_local_start_offset = 0;
    }

    pub fn pushed_bytes(&self) -> u32 { self.pushed_bytes }

    pub fn pushed_items<T>(&self) -> u32 {
        self.pushed_bytes / std::mem::size_of::<T>() as u32
    }
}

impl<'l> Drop for GpuStreamWriter<'l> {
    fn drop(&mut self) {
        self.flush_staging_buffer();
    }
}

pub struct GpuStreamsDescritptor {
    pub usages: wgpu::BufferUsages,
    pub buffer_size: u32,
    pub chunk_size: u32,
    pub label: Option<&'static str>,
}

#[derive(Debug)]
struct StreamInfo {
    buffer_index: u32,
    offset: u32,
    size: u32,
    // TODO: bytes_per_item?
}

struct StreamBuffer {
    handle: wgpu::Buffer,
    size: u32,
    current_offset: u32,
}

pub struct GpuStreamsResources {
    chunk_size: u32,
    buffer_size: u32,
    usage: wgpu::BufferUsages,
    label: Option<&'static str>,

    buffers: Vec<StreamBuffer>,
    stream_info: Vec<StreamInfo>,
    next_stream_id: Arc<AtomicU32>,
}

impl GpuStreamsResources {
    pub fn new(descriptor: &GpuStreamsDescritptor) -> Self {
        GpuStreamsResources {
            buffers: Vec::new(),
            chunk_size: descriptor.chunk_size,
            buffer_size: descriptor.buffer_size,
            usage: descriptor.usages | wgpu::BufferUsages::COPY_DST,
            label: descriptor.label,

            stream_info: Vec::new(),
            next_stream_id: Arc::new(AtomicU32::new(0)),
        }
    }

    pub fn begin_frame(&self, staging_buffers: Arc<Mutex<StagingBufferPool>>) -> GpuStreams {
        self.next_stream_id.store(0, Ordering::Release);
        GpuStreams {
            chunk_size: self.chunk_size,
            ops: RefCell::new(Vec::new()),
            staging_buffers,
            next_stream_id: self.next_stream_id.clone(),
        }
    }

    pub fn upload(
        &mut self,
        stream_ops: &[StreamOps],
        staging_buffers: &StagingBufferPool,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let num_streams = self.next_stream_id.load(Ordering::Acquire) as usize;
        self.stream_info.clear();
        self.stream_info.reserve(num_streams);
        for _ in 0..num_streams {
            self.stream_info.push(StreamInfo {
                buffer_index: u32::MAX,
                offset: 0,
                size: 0,
            });
        }

        let mut num_ops = 1;
        for ops in stream_ops {
            num_ops += ops.ops.len();
        }
        let mut ops = Vec::new();
        ops.reserve(num_ops);
        for so in stream_ops {
            ops.extend_from_slice(&so.ops);
        }

        ops.sort_by_key(|op| op.sort_key);
        // Push a dummy op at the end to flush the last stream.
        ops.push(CopyOp {
            sort_key: u64::MAX,
            size: 0,
            staging_id: StagingBufferId(u32::MAX),
            staging_offset: StagingOffset(u32::MAX),
        });

        for buffer in &mut self.buffers {
            buffer.current_offset = 0;
        }

        let mut op_idx = 0;
        let mut op_start = 0;
        let mut stream_size = 0;
        let mut current_stream = std::u32::MAX;
        for op in &ops {
            let next_stream = (op.sort_key >> 48) as u32;
            if next_stream != current_stream && stream_size > 0 {
                // TODO: ideally there would be a way to have a per stream alignment
                // but for now all streams begin at a multiple of 32 bytes.
                stream_size = align(stream_size, 32);
                // Flush.
                let mut buffer_idx = 0;
                for buffer in &self.buffers {
                    if buffer.size - buffer.current_offset > stream_size {
                        break;
                    }
                    buffer_idx += 1;
                }
                if buffer_idx == self.buffers.len() {
                    let size = stream_size.max(self.buffer_size);
                    let handle = device.create_buffer(&wgpu::BufferDescriptor {
                        label: self.label,
                        size: size as u64,
                        usage: self.usage,
                        mapped_at_creation: false,
                    });

                    self.buffers.push(StreamBuffer {
                        handle,
                        size,
                        current_offset: 0,
                    });
                }
                let mut dst_offset = self.buffers[buffer_idx].current_offset;
                self.buffers[buffer_idx].current_offset += stream_size;
                debug_assert!(
                    self.buffers[buffer_idx].current_offset <= self.buffers[buffer_idx].size,
                    "{} <= {}",
                    self.buffers[buffer_idx].current_offset,
                    self.buffers[buffer_idx].size,
                );
                let dst_buffer = &self.buffers[buffer_idx].handle;

                let info = &mut self.stream_info[current_stream as usize];
                info.buffer_index = buffer_idx as u32;
                info.size = stream_size;
                info.offset = dst_offset;

                // Loop over the ops in the stream's range and issue copies from the
                // staging buffer(s).
                for op in &ops[op_start..op_idx] {
                    let staging_buffer = staging_buffers.get_handle(op.staging_id);
                    encoder.copy_buffer_to_buffer(
                        staging_buffer,
                        op.staging_offset.0 as u64,
                        dst_buffer,
                        dst_offset as u64,
                        op.size as u64,
                    );
                    dst_offset += op.size;
                }

                op_start = op_idx;
            }

            stream_size += op.size;
            current_stream = next_stream;
            op_idx += 1;
        }
    }

    /// Returns the buffer and a range in byte containing the stream.
    pub fn resolve(&self, stream: StreamId) -> Option<(&wgpu::Buffer, Range<u32>)> {
        let info = &self.stream_info[stream.0 as usize];
        let idx = info.buffer_index as usize;
        if idx >= self.buffers.len() {
            // Can happen when resolving a stream in which nothing was pushed.
            return None;
        }
        if info.size == 0 {
            return None;
        }
        Some((
            &self.buffers[idx].handle,
            info.offset..(info.offset + info.size),
        ))
    }

    pub fn resolve_buffer_slice(&self, stream: Option<StreamId>) -> Option<wgpu::BufferSlice> {
        let (buffer, range) = self.resolve(stream?)?;
        Some(buffer.slice(range.start as u64 .. range.end as u64))
    }
}

#[derive(Clone, Debug)]
struct CopyOp {
    staging_id: StagingBufferId,
    // In bytes.
    staging_offset: StagingOffset,
    // In bytes.
    size: u32,
    sort_key: u64,
}

pub struct StreamOps {
    ops: Vec<CopyOp>,
}

fn align(size: u32, alignment: u32) -> u32 {
    let rem = size % alignment;
    if rem == 0 {
        size
    } else {
        size + alignment - rem
    }
}
