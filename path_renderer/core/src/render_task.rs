use crate::gpu::{GpuBufferAddress, GpuBufferWriter};
use crate::units::{SurfaceIntRect, SurfaceIntSize, SurfaceRect, SurfaceVector};


#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct RenderTaskHandle(pub GpuBufferAddress);

impl RenderTaskHandle {
    pub const INVALID: Self = RenderTaskHandle(GpuBufferAddress::INVALID);

    pub fn is_valid(&self) -> bool {
        *self != RenderTaskHandle::INVALID
    }

    pub fn to_u32(&self) -> u32 {
        self.0.to_u32()
    }

    pub fn from_u32(addr: u32) -> Self {
        RenderTaskHandle(GpuBufferAddress(addr))
    }
}


#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RenderTaskInfo {
    /// Acts as a clip in surface space.
    pub bounds: SurfaceIntRect,
    /// An offset to apply after clipping.
    pub offset: SurfaceVector,
    ///
    pub handle: RenderTaskHandle,
}

/// The data copied into to the gpu store.
///
/// Must match the layout of `RenderTask` in render_task.wgsl
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct RenderTaskData {
    /// Acts as a clip.
    pub rect: SurfaceRect,
    /// Optional offset.
    pub content_offset: SurfaceVector,
    /// 1.0 / target.width
    pub rcp_target_width: f32,
    /// 1.0 / target.height
    pub rcp_target_height: f32,
}

unsafe impl bytemuck::Pod for RenderTaskData {}
unsafe impl bytemuck::Zeroable for RenderTaskData {}

pub fn add_render_task(f32_buffer: &mut GpuBufferWriter, bounds: &SurfaceIntRect, offset: SurfaceVector, target_size: SurfaceIntSize) -> RenderTaskInfo {
    let size = target_size.to_f32();
    let handle = RenderTaskHandle(f32_buffer.push(RenderTaskData {
        rect: bounds.to_f32(),
        content_offset: SurfaceVector::zero(),
        rcp_target_width: 1.0 / size.width,
        rcp_target_height: 1.0 / size.height,
    }));

    RenderTaskInfo {
        bounds: *bounds,
        offset,
        handle,
    }
}

pub struct FrameAtlasAllocator {
    allocator: guillotiere::SimpleAtlasAllocator,
    allocated_px: u32,
}

impl FrameAtlasAllocator {
    pub fn new(size: SurfaceIntSize) -> Self {
        FrameAtlasAllocator {
            allocator: guillotiere::SimpleAtlasAllocator::with_options(
                size.cast_unit(),
                &guillotiere::AllocatorOptions {
                    alignment: SurfaceIntSize::new(16, 16).cast_unit(),
                    small_size_threshold: 64,
                    large_size_threshold: 512,
                },
            ),
            allocated_px: 0,
        }
    }

    pub fn allocate(
        &mut self,
        f32_buffer: &mut GpuBufferWriter,
        bounds: &SurfaceIntRect,
    ) -> Option<RenderTaskInfo> {
        let alloc = self.allocator.allocate(bounds.size().cast_unit())?;
        let offset = alloc.min.cast_unit().to_f32().to_vector();
        self.allocated_px += alloc.area() as u32;
        Some(add_render_task(f32_buffer, bounds, offset, self.size()))
    }

    pub fn size(&self) -> SurfaceIntSize {
        self.allocator.size().cast_unit()
    }

    pub fn is_empty(&mut self) -> bool {
        self.allocator.is_empty()
    }

    pub fn occupancy(&self) -> f32 {
        let s = self.size();
        let area = s.width as f32 * s.height as f32;
        self.allocated_px as f32 / area
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct AtlasHandle(u32);

pub struct DynamicAtlasAllocator {
    allocator: etagere::AtlasAllocator,
    allocated_px: u32,
    allocations: Vec<RenderTaskInfo>
}

impl DynamicAtlasAllocator {
    pub fn new(size: SurfaceIntSize) -> Self {
        let num_columns = match size.width {
            0..1025 => 1,
            1025..4097 => 2,
            _ => 3,
        };
        DynamicAtlasAllocator {
            allocator: etagere::AtlasAllocator::with_options(
                size.cast_unit(),
                &etagere::AllocatorOptions {
                    alignment: SurfaceIntSize::new(16, 16).cast_unit(),
                    vertical_shelves: false,
                    num_columns,
                }
            ),
            allocated_px: 0,
            allocations: Vec::with_capacity(128),
        }
    }

    pub fn allocate(
        &mut self,
        bounds: &SurfaceIntRect,
    ) -> Option<AtlasHandle> {
        let alloc = self.allocator.allocate(bounds.size().cast_unit())?;
        let offset = alloc.rectangle.min.cast_unit().to_f32().to_vector();
        self.allocated_px += alloc.rectangle.area() as u32;
        self.allocations.push(RenderTaskInfo {
            bounds: *bounds,
            offset,
            handle: RenderTaskHandle::INVALID
        });

        Some(AtlasHandle(alloc.id.serialize()))
    }

    pub fn get_render_task(&mut self, f32_buffer: &mut GpuBufferWriter, handle: AtlasHandle) -> RenderTaskInfo {
        let info = &mut self.allocations[handle.0 as usize];
        if info.handle == RenderTaskHandle::INVALID {
            let size = self.allocator.size();
            let h = f32_buffer.push(RenderTaskData {
                rect: info.bounds.to_f32(),
                content_offset: SurfaceVector::zero(),
                rcp_target_width: 1.0 / size.width as f32,
                rcp_target_height: 1.0 / size.height as f32,
            });

            info.handle = RenderTaskHandle(h);
        }

        return info.clone()
    }

    pub fn size(&self) -> SurfaceIntSize {
        self.allocator.size().cast_unit()
    }

    pub fn is_empty(&mut self) -> bool {
        self.allocator.is_empty()
    }

    pub fn occupancy(&self) -> f32 {
        let s = self.size();
        let area = s.width as f32 * s.height as f32;
        self.allocated_px as f32 / area
    }
}
