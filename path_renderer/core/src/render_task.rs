use std::fmt;
use crate::gpu::{GpuBufferAddress, GpuBufferWriter};
use crate::units::{SurfaceIntRect, SurfaceIntSize, SurfaceIntVector, SurfaceRect, SurfaceVector};

/// The address of render task data in the float GpuBuffer.
///
/// This address is only valid for the duration of the frame current frame.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct RenderTaskAdress(pub GpuBufferAddress);

impl RenderTaskAdress {
    pub const NONE: Self = RenderTaskAdress(GpuBufferAddress::NONE);

    #[inline]
    pub fn is_some(self) -> bool {
        self.0.is_some()
    }

    #[inline]
    pub fn is_none(self) -> bool {
        self.0.is_none()
    }

    #[inline]
    pub fn to_buffer_address(self) -> GpuBufferAddress { self.0 }

    #[inline]
    pub fn to_u32(self) -> u32 {
        self.0.to_u32()
    }

    #[inline]
    pub fn from_u32(addr: u32) -> Self {
        RenderTaskAdress(GpuBufferAddress(addr))
    }
}

unsafe impl bytemuck::Pod for RenderTaskAdress {}
unsafe impl bytemuck::Zeroable for RenderTaskAdress {}

impl std::fmt::Debug for RenderTaskAdress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RenderTask {
    /// Acts as a clip in surface space.
    pub bounds: SurfaceRect,
    /// An offset to apply after clipping.
    pub offset: SurfaceVector,
    /// Where the task is drawn in its render target.
    pub target_rect: SurfaceIntRect,
    ///
    pub gpu_address: RenderTaskAdress,
}

impl RenderTask {
    pub fn new_sub_task(
        f32_buffer: &mut GpuBufferWriter,
        bounds: &SurfaceIntRect,
        target_offset: SurfaceVector,
        target_size: SurfaceIntSize,
    ) -> Self {
        add_render_task(f32_buffer, bounds, target_offset, target_size)
    }

    pub fn new(
        f32_buffer: &mut GpuBufferWriter,
        target_size: SurfaceIntSize,
        content_offset: SurfaceIntVector,
    ) -> Self {
        let bounds = SurfaceIntRect::from_size(target_size).translate(content_offset);
        add_render_task(f32_buffer, &bounds, SurfaceVector::zero(), target_size)
    }
}

/// The data copied into to the gpu store.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct RenderTaskGpuData {
    // The first part of this struct (8 floats) contains data that
    // is typically read by shaders when rendering into a render task.
    // It must match the layout of `RenderTask` in render_task.wgsl

    /// Acts as a clip in the surface space of the render task.
    pub clip: SurfaceRect,
    /// Optional offset.
    pub content_offset: SurfaceVector,
    /// 1.0 / target.width
    pub rcp_target_width: f32,
    /// 1.0 / target.height
    pub rcp_target_height: f32,

    // The second part, stored immediately after, is typically read by
    // shaders when reading from the output of a render task.

    /// The pixels of the task, in the coordinate space of the target.
    pub image_source: SurfaceRect
}

unsafe impl bytemuck::Pod for RenderTaskGpuData {}
unsafe impl bytemuck::Zeroable for RenderTaskGpuData {}

fn add_render_task(
    f32_buffer: &mut GpuBufferWriter,
    bounds: &SurfaceIntRect,
    target_offset: SurfaceVector,
    target_size: SurfaceIntSize
) -> RenderTask {
    let size = target_size.to_f32();
    let boundsf = bounds.to_f32();
    let content_offset = boundsf.min - target_offset.to_point();
    let image_source = SurfaceRect {
        min: target_offset.to_point(),
        max: target_offset.to_point() + boundsf.size().to_vector(),
    };

    let gpu_address = RenderTaskAdress(f32_buffer.push(RenderTaskGpuData {
        clip: boundsf,
        content_offset,
        rcp_target_width: 1.0 / size.width,
        rcp_target_height: 1.0 / size.height,
        image_source,
    }));

    RenderTask {
        bounds: boundsf,
        offset: content_offset,
        gpu_address,
        target_rect: image_source.to_i32(),
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct AtlasHandle(u32);

pub struct DynamicAtlasAllocator {
    allocator: etagere::AtlasAllocator,
    allocated_px: u32,
    allocations: Vec<RenderTask>
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
        let size = bounds.size();
        let boundsf = bounds.to_f32();
        let local_content_offset = boundsf.min.to_vector();
        let alloc = self.allocator.allocate(size.cast_unit())?;
        let target_origin = alloc.rectangle.min.cast_unit();
        let target_offset = target_origin.to_f32().to_vector();
        self.allocated_px += alloc.rectangle.area() as u32;
        self.allocations.push(RenderTask {
            bounds: boundsf,
            target_rect: SurfaceIntRect {
                min: target_origin,
                max: target_origin + size,
            },
            offset: target_offset - local_content_offset,
            gpu_address: RenderTaskAdress::NONE,
        });

        Some(AtlasHandle(alloc.id.serialize()))
    }

    pub fn get_render_task(&mut self, f32_buffer: &mut GpuBufferWriter, handle: AtlasHandle) -> RenderTask {
        let info = &mut self.allocations[handle.0 as usize];
        if info.gpu_address.is_none() {
            let size = self.allocator.size();
            let h = f32_buffer.push(RenderTaskGpuData {
                clip: info.bounds,
                content_offset: SurfaceVector::zero(),
                rcp_target_width: 1.0 / size.width as f32,
                rcp_target_height: 1.0 / size.height as f32,
                image_source: info.target_rect.to_f32(),
            });

            info.gpu_address = RenderTaskAdress(h);
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
