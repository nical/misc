use std::{ops::Range, sync::{atomic::{AtomicU32, Ordering}, Arc, Mutex}, u32, u64};

use super::{StagingBufferId, StagingBufferPool, StagingOffset};


// Associated with a specific resource.
pub struct GpuStreamsWriter {
    chunk_start: *mut u8,
    // Offset of the cursor relative to the beginning of the current chunk, in bytes.
    staging_local_offset: u32,
    staging_local_start_offset: u32,
    // In bytes
    chunk_size: u32,
    // Offset of the current chunk in bytes.
    staging_buffer_offset: StagingOffset,
    staging_buffer_id: StagingBufferId,

    ops: Vec<StreamOp>,

    current_sort_key: u64,

    next_stream_id: Arc<AtomicU32>,
    staging_buffers: Arc<Mutex<StagingBufferPool>>,
}

unsafe impl<'l> Send for GpuStreamsWriter {}

impl GpuStreamsWriter {
    pub fn next_stream_id(&self) -> StreamId {
        StreamId(self.next_stream_id.fetch_add(1, Ordering::Relaxed))
    }

    pub fn write(&mut self, stream: StreamId, sort_key: u32) -> GpuStreamWriter {
        GpuStreamWriter {
            streams: self,
            key: (stream.0 as u64) << 32 | sort_key as u64
        }
    }

    fn push_bytes(&mut self, sort_key: u64, data: &[u8]) {
        let size = data.len() as u32;
        let mut new_local_offset = self.staging_local_offset + size;

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

    // Note: this does not reset the offsets and pointers into
    // the staging buffer.
    fn flush_staging_buffer(&mut self) {
        let size = self.staging_local_offset - self.staging_local_start_offset;
        if size > 0 {
            self.ops.push(StreamOp {
                staging_id: self.staging_buffer_id,
                staging_offset: StagingOffset(self.staging_buffer_offset.0 + self.staging_local_start_offset),
                size,
                sort_key: self.current_sort_key,
            });
        }
    }

    #[cold]
    fn replace_staging_buffer(&mut self) {
        self.flush_staging_buffer();

        let chunk = self.staging_buffers
            .lock()
            .unwrap()
            .get_mapped_chunk(self.chunk_size);

        //println!("GpuStoreWriter::flush_buffer ptr:{:?} size:{:?} id:{:?} offset:{:?}", chunk.ptr, chunk.size, chunk.id, chunk.offset);

        self.staging_buffer_id = chunk.id;
        self.staging_buffer_offset = chunk.offset;
        self.chunk_start = chunk.ptr;
        self.staging_local_offset = 0;
    }

    pub(crate) fn finish(&mut self) -> StreamOps {
        self.flush_staging_buffer();
        StreamOps {
            ops: std::mem::take(&mut self.ops)
        }
    }
}

impl Clone for GpuStreamsWriter {
    fn clone(&self) -> GpuStreamsWriter {
        let ptr = std::ptr::null_mut();
        GpuStreamsWriter {
            chunk_start: ptr,
            // Initialze to chunk_size so that it triggers flush
            // at the first push.
            staging_local_offset: self.chunk_size,
            staging_local_start_offset: self.chunk_size,
            chunk_size: self.chunk_size,
            staging_buffer_id: StagingBufferId(0),
            staging_buffer_offset: StagingOffset(0),
            ops: Vec::new(),
            staging_buffers: self.staging_buffers.clone(),
            next_stream_id: self.next_stream_id.clone(),
            current_sort_key: u64::MAX,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct StreamId(u32);

pub struct GpuStreamWriter<'l> {
    streams: &'l mut GpuStreamsWriter,
    key: u64,
}

impl<'l> GpuStreamWriter<'l> {
    #[inline]
    pub fn push_bytes(&mut self, data: &[u8]) {
        self.streams.push_bytes(self.key, data);
    }

    #[inline]
    pub fn push_f32(&mut self, data: &[f32]) {
        self.streams.push_bytes(self.key, bytemuck::cast_slice(data));
    }
}

pub struct GpuStreamsDescritptor {
    pub usage: wgpu::BufferUsages,
    pub buffer_size: u32,
    pub chunk_size: u32,
    pub label: Option<&'static str>,
}

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
            usage: descriptor.usage | wgpu::BufferUsages::COPY_DST,
            label: descriptor.label,

            stream_info: Vec::new(),
            next_stream_id: Arc::new(AtomicU32::new(0)),
        }
    }

    pub fn begin_frame(&self, staging_buffers: Arc<Mutex<StagingBufferPool>>) -> GpuStreamsWriter {
        self.next_stream_id.store(0, Ordering::Release);
        GpuStreamsWriter {
            chunk_start: std::ptr::null_mut(),
            staging_local_offset: self.chunk_size,
            staging_local_start_offset: self.chunk_size,
            chunk_size: self.chunk_size,
            staging_buffer_id: StagingBufferId(0),
            staging_buffer_offset: StagingOffset(0),
            ops: Vec::new(),
            staging_buffers,
            next_stream_id: self.next_stream_id.clone(),
            current_sort_key: 0,
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
        ops.push(StreamOp {
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
            let stream = (op.sort_key >> 32) as u32;
            if stream != current_stream && stream_size > 0 {
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
                // TODO: ideally there would be a way to have a per stream alignment
                // but for now all streams begin at a multiple of 32 bytes.
                self.buffers[buffer_idx].current_offset += align(stream_size, 32);
                let dst_buffer = &self.buffers[buffer_idx].handle;

                self.stream_info[stream as usize].buffer_index = buffer_idx as u32;

                // Loop over the ops in the stream's range and issue copies from the
                // staging buffer(s).
                for op in &ops[op_start..op_idx] {
                    let staging_buffer = staging_buffers.get_handle(op.staging_id);
                    encoder.copy_buffer_to_buffer(
                        staging_buffer,
                        op.staging_offset.0 as u64,
                        dst_buffer,
                        dst_offset as u64,
                        op.size as u64
                    );
                    dst_offset += op.size;
                }

                op_start = op_idx;
            }

            stream_size += op.size;
            current_stream = stream;
            op_idx += 1;
        }
    }

    /// Returns the buffer and a range in byte containing the stream.
    pub fn resolve(&mut self, stream: StreamId) -> (&wgpu::Buffer, Range<u32>) {
        let info = &self.stream_info[stream.0 as usize];
        let idx = info.buffer_index;
        (
            &self.buffers[idx as usize].handle,
            info.offset .. (info.offset + info.size),
        )
    }
}



#[derive(Clone, Debug)]
struct StreamOp {
    staging_id: StagingBufferId,
    // In bytes.
    staging_offset: StagingOffset,
    // In bytes.
    size: u32,
    sort_key: u64,
}

pub struct StreamOps {
    ops: Vec<StreamOp>,
}

fn align(size: u32, alignment: u32) -> u32 {
    let rem = size % alignment;
    if rem == 0 {
        size
    } else {
        size + alignment - rem
    }
}
