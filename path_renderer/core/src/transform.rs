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
        const AXIS_ALIGNED = 1;
        const GPU          = 2;
    }
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TransformId(u16);
pub type GpuTransformId = u16;

impl TransformId {
    pub const ROOT: Self = TransformId(0);
    pub const NONE: Self = TransformId(std::u16::MAX);
    pub fn try_index(self) -> Option<usize> {
        if self == Self::NONE {
            return None;
        }

        Some(self.0 as usize)
    }

    fn index(self) -> usize {
        self.0 as usize
    }

    pub fn from_index(idx: usize) -> Self {
        debug_assert!(idx < std::u32::MAX as usize);
        TransformId(idx as u16)
    }

    pub fn is_none(self) -> bool {
        self.0 == std::u16::MAX
    }

    pub fn or(self, or: Self) -> Self {
        if self.is_none() {
            return or;
        }

        self
    }
}

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

#[derive(Copy, Clone, Debug)]
pub struct Transform {
    transform: LocalToSurfaceTransform,
    id: TransformId,
    flags: TransformFlags,
    gpu_handle: GpuBufferAddress,
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

    pub fn matrix(&self) -> &LocalToSurfaceTransform {
        &self.transform
    }

    pub fn id(&self) -> TransformId {
        self.id
    }

    pub fn get_gpu_handle(&mut self, f32_buffer: &mut GpuBufferWriter) -> GpuBufferAddress {
        if self.gpu_handle != GpuBufferAddress::INVALID {
            return self.gpu_handle;
        }

        let axis_aligned = if self.flags.contains(TransformFlags::AXIS_ALIGNED) {
            1.0
        } else {
            0.0
        };
        let t = &self.transform;

        let handle = f32_buffer.push_slice(&[t.m11, t.m12, t.m21, t.m22, t.m31, t.m32, axis_aligned, 0.0]);

        self.gpu_handle = handle;

        handle
    }
}

pub struct Transforms {
    transforms: Vec<Transform>,
}

impl Transforms {
    pub fn new() -> Self {
        let root = Transform {
            transform: LocalToSurfaceTransform::identity(),
            id: TransformId(0),
            flags: TransformFlags::AXIS_ALIGNED,
            gpu_handle: GpuBufferAddress::INVALID,
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
            gpu_handle: GpuBufferAddress::INVALID,
        });

        id
    }

    pub fn root(&self) -> &Transform { &self.transforms[0] }

    pub fn root_mut(&mut self) -> &mut Transform { &mut self.transforms[0] }

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
