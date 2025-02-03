use std::sync::{Arc, Mutex};
use std::fmt;

/// Offset in bytes into a staging buffer.
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct StagingOffset(pub u32);
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct StagingBufferId(pub u32);

impl std::fmt::Debug for StagingBufferId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

impl std::fmt::Debug for StagingOffset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
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

    // Mapped and ready for use.
    available: Vec<wgpu::Buffer>,
    // In use this frame.
    active: Vec<wgpu::Buffer>,
    // map_aysnc has been called, waiting for the callback to resolve.
    pending: Vec<Option<wgpu::Buffer>>,

    ready_list: Arc<Mutex<Vec<(u16, bool)>>>,

    current_chunks: MappedStagingBuffer,
    current_chunks_offset: u32,

    pub(crate) cpu_buffers: Vec<Vec<u8>>,

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
    pub(crate) fn new_for_testing(buffer_size: u32) -> Self {
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

            self.pending[idx] = Some(buffer);
        }
    }
}
