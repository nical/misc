
pub type Size = euclid::default::Size2D<f32>;
pub type Rect = euclid::default::Box2D<f32>;
pub type Point = euclid::default::Point2D<f32>;
pub type Vector = euclid::default::Vector2D<f32>;

pub type IntSize = euclid::default::Size2D<i32>;
pub type IntRect = euclid::default::Box2D<i32>;
pub type IntPoint = euclid::default::Point2D<i32>;
pub type IntVector = euclid::default::Vector2D<i32>;

pub struct DeviceSpace;

pub type DeviceSize = euclid::Size2D<f32, DeviceSpace>;
pub type DeviceRect = euclid::Box2D<f32, DeviceSpace>;
pub type DevicePoint = euclid::Point2D<f32, DeviceSpace>;
pub type DeviceVector = euclid::Vector2D<f32, DeviceSpace>;

pub type DeviceIntSize = euclid::Size2D<i32, DeviceSpace>;
pub type DeviceIntRect = euclid::Box2D<i32, DeviceSpace>;
pub type DeviceIntPoint = euclid::Point2D<i32, DeviceSpace>;
pub type DeviceIntVector = euclid::Vector2D<i32, DeviceSpace>;

pub use euclid::size2;
pub use euclid::point2;
pub use euclid::vec2;
