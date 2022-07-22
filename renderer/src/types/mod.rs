pub mod units;

pub type PrimitiveIndex = u32;

pub type SystemId = u32;
pub type BatchIndex = u32;

#[derive(Copy, Clone, Debug)]
pub struct BatchId {
    pub system: SystemId,
    pub index: BatchIndex,
}

pub struct Primitive {
    pub system: SystemId,
    pub index: PrimitiveIndex
}
