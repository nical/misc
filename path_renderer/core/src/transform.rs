use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use bitflags::bitflags;
use lyon::geom::euclid::Transform2D;

use crate::{
    gpu::{GpuBufferAddress, GpuBufferWriter},
    units::{
        point, vector, LocalPoint, LocalSpace, LocalToSurfaceTransform, LocalTransform,
        SurfacePoint, SurfaceVector, Vector,
    },
};

bitflags! {
    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct TransformFlags: u16 {
        const GPU          = 1;
        const AXIS_ALIGNED = 2;
        const IDENTITY     = 4;
    }
}

/// Optional address of a transform in the float GpuBuffer.
///
/// This address is only valid for the duration of the frame current frame.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct GpuTransformAddress(pub GpuBufferAddress);

impl GpuTransformAddress {
    pub const NONE: Self = GpuTransformAddress(GpuBufferAddress::NONE);

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
    pub fn to_u32(self) -> u32 { self.0.to_u32() }
}

unsafe impl bytemuck::Pod for GpuTransformAddress {}
unsafe impl bytemuck::Zeroable for GpuTransformAddress {}

impl std::fmt::Debug for GpuTransformAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}


/// Optional index of a CPU-side transform.
///
/// Only valid durin the current frame.
/// Cannot be used on the GPU. See `GpuTransformAddress`.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TransformId(u16);

impl TransformId {
    pub const IDENTITY: Self = TransformId(0);
    pub const NONE: Self = TransformId(std::u16::MAX);

    #[inline]
    pub fn try_index(self) -> Option<usize> {
        if self == Self::NONE {
            return None;
        }

        Some(self.0 as usize)
    }

    #[inline]
    fn index(self) -> usize {
        debug_assert!(self.is_some());
        self.0 as usize
    }

    #[inline]
    pub fn from_index(idx: usize) -> Self {
        debug_assert!(idx < std::u32::MAX as usize);
        TransformId(idx as u16)
    }

    #[inline]
    pub fn is_some(self) -> bool {
        self != Self::NONE
    }

    #[inline]
    pub fn is_none(self) -> bool {
        self == Self::NONE
    }
}

// TODO: upstream into euclid.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ScaleOffset {
    pub scale: Vector,
    pub offset: SurfaceVector,
}

impl ScaleOffset {
    #[inline]
    pub fn identity() -> Self {
        ScaleOffset {
            scale: vector(1.0, 1.0),
            offset: vector(0.0, 0.0),
        }
    }

    #[inline]
    pub fn transform_point(&self, p: LocalPoint) -> SurfacePoint {
        point(p.x * self.scale.x, p.y * self.scale.y) + self.offset
    }

    #[inline]
    pub fn then(&self, other: &Self) -> Self {
        ScaleOffset {
            scale: vector(self.scale.x * other.scale.x, self.scale.y * other.scale.y),
            offset: vector(self.offset.x * self.scale.x, self.offset.y * self.scale.y)
                + other.offset,
        }
    }

    #[inline]
    pub fn to_matrix(&self) -> LocalToSurfaceTransform {
        LocalToSurfaceTransform::new(
            self.scale.x,
            0.0,
            0.0,
            self.scale.y,
            self.offset.x,
            self.offset.y,
        )
    }

    pub fn from_matrix(mat: &LocalTransform) -> Self {
        ScaleOffset {
            scale: vector(mat.m11, mat.m22),
            offset: vector(mat.m31, mat.m32),
        }
    }
}

#[derive(Debug)]
pub struct Transform {
    transform: LocalToSurfaceTransform,
    id: TransformId,
    flags: TransformFlags,
    gpu_handle: AtomicU32,
}

impl Transform {
    pub fn is_axis_aligned(&self) -> bool {
        self.flags.contains(TransformFlags::AXIS_ALIGNED)
    }

    pub fn as_scale_offset(&self) -> Option<ScaleOffset> {
        if !self.flags.contains(TransformFlags::AXIS_ALIGNED) {
            return None;
        }

        Some(ScaleOffset::from_matrix(
            &self.transform.with_destination::<LocalSpace>(),
        ))
    }

    pub fn is_identity(&self) -> bool {
        self.flags.contains(TransformFlags::IDENTITY)
    }

    pub fn matrix(&self) -> &LocalToSurfaceTransform {
        &self.transform
    }

    pub fn id(&self) -> TransformId {
        self.id
    }

    pub fn request_gpu_handle(&self, f32_buffer: &mut GpuBufferWriter) -> GpuTransformAddress {
        // Note that this is a racy operation: Multiple threads can request a GPU handle resulting
        // in the same transform being pushed multiple times to the GPU.
        // This isn't a problem, however. We only care about shaders getting correct values for
        // the transform, and avoiding pushing the same GPU transform hundreds of times.
        let handle = gpu_handle_from_u32(self.gpu_handle.load(Ordering::Relaxed));
        if handle.is_some() {
            return handle;
        }

        if self.flags.contains(TransformFlags::IDENTITY) {
            return GpuTransformAddress::NONE;
        }

        let axis_aligned = if self.flags.contains(TransformFlags::AXIS_ALIGNED) {
            1.0
        } else {
            0.0
        };
        let t = &self.transform;

        let handle = GpuTransformAddress(
            f32_buffer.push_slice(&[t.m11, t.m12, t.m21, t.m22, t.m31, t.m32, axis_aligned, 0.0])
        );

        self.gpu_handle.store(gpu_handle_to_u32(handle), Ordering::Relaxed);

        handle
    }
}

fn gpu_handle_from_u32(val: u32) -> GpuTransformAddress {
    GpuTransformAddress(GpuBufferAddress(val))
}

fn gpu_handle_to_u32(addr: GpuTransformAddress) -> u32 {
    addr.0.0
}

pub struct Transforms {
    transforms: Vec<Transform>,
}

impl Transforms {
    pub fn new() -> Self {
        let root = Transform {
            transform: LocalToSurfaceTransform::identity(),
            id: TransformId(0),
            flags: TransformFlags::AXIS_ALIGNED | TransformFlags::IDENTITY,
            gpu_handle: AtomicU32::new(gpu_handle_to_u32(GpuTransformAddress::NONE)),
        };

        Transforms {
            transforms: vec![root],
        }
    }

    pub fn add(&mut self, transform: &LocalToSurfaceTransform) -> TransformId {
        let mut flags = TransformFlags::empty();
        let is_scale_offset = is_scale_offset(transform);
        if is_scale_offset {
            flags |= TransformFlags::AXIS_ALIGNED;
        }

        let id = TransformId::from_index(self.transforms.len());

        self.transforms.push(Transform {
            id,
            transform: *transform,
            flags,
            gpu_handle: AtomicU32::new(gpu_handle_to_u32(GpuTransformAddress::NONE)),
        });

        id
    }

    pub fn identity(&self) -> &Transform { &self.transforms[0] }

    pub fn identity_mut(&mut self) -> &mut Transform { &mut self.transforms[0] }

    pub fn root_id(&self) -> TransformId { TransformId(0) }

    pub fn get(&self, id: TransformId) -> &Transform {
        &self.transforms[id.index()]
    }

    pub fn get_mut(&mut self, id: TransformId) -> &mut Transform {
        &mut self.transforms[id.index()]
    }

    pub fn clear(&mut self) {
        self.transforms.shrink_to(1);
    }
}

impl Default for Transforms {
    fn default() -> Self {
        Self::new()
    }
}

// Same as Skia's SK_ScalarNearlyZero.
const EPSILON: f32 = 1.0 / 4096.0;

pub fn is_scale_offset<S, D>(m: &Transform2D<f32, S, D>) -> bool {
    m.m12.abs() <= EPSILON && m.m21.abs() <= EPSILON
}
