use crate::gpu::GpuBufferAddress;
use crate::units::{SurfaceIntRect, SurfaceRect, SurfaceVector};


#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct RenderTaskHandle(pub(crate) GpuBufferAddress);

impl RenderTaskHandle {
    pub const INVALID: Self = RenderTaskHandle(GpuBufferAddress::INVALID);

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
