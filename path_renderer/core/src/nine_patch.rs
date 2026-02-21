use crate::euclid::{Box2D, ScaleOffset2D};
use crate::units::point;
use crate::Sides;

pub type SurfaceNinePatch = NinePatch<crate::units::SurfaceSpace>;
pub type LocalNinePatch = NinePatch<crate::units::LocalSpace>;
pub type NormalizedNinePatch = NinePatch<crate::units::NormalizedSpace>;

pub struct NinePatch<U> {
    bounds: Box2D<f32, U>,
    // Absolute positions in the same coordinate space as `bounds`.
    left: f32,
    right: f32,
    top: f32,
    bottom: f32,
}

impl<U> NinePatch<U> {
    pub fn new(
        bounds: Box2D<f32, U>,
        // Relative to their respective edge.
        left: f32, top: f32, right: f32, bottom: f32,
    ) -> Self {
        NinePatch {
            bounds,
            left: bounds.min.x + left,
            top: bounds.min.y + top,
            right: bounds.max.x - right,
            bottom: bounds.max.y - bottom,
        }.canonicalized()
    }

    pub fn new_unchecked(
        bounds: Box2D<f32, U>,
        // Relative to their respective edge.
        left: f32, top: f32, right: f32, bottom: f32,
    ) -> Self {
        NinePatch {
            bounds,
            left: bounds.min.x + left,
            top: bounds.min.y + top,
            right: bounds.max.x - right,
            bottom: bounds.max.y - bottom,
        }
    }

    pub fn uniform(bounds: Box2D<f32, U>, slice: f32) -> Self {
        NinePatch::new(bounds, slice, slice, slice, slice)
    }

    pub fn canonicalized(&self) -> Self {
        let mut bounds = self.bounds;
        bounds.max.x = bounds.max.x.max(bounds.min.x);
        bounds.max.y = bounds.max.y.max(bounds.min.y);
        let mut left = self.left.max(bounds.min.x).min(bounds.max.x);
        let mut top = self.top.max(bounds.min.y).min(bounds.max.y);
        let mut right = self.right.max(bounds.min.x).min(bounds.max.x);
        let mut bottom = self.bottom.max(bounds.min.y).min(bounds.max.y);
        if left > right {
            let mid = (left + right) * 0.5;
            left = mid;
            right = mid;
        }
        if top > bottom {
            let mid = (top + bottom) * 0.5;
            top = mid;
            bottom = mid;
        }

        NinePatch { bounds, left, right, top, bottom }
    }

    pub fn transformed<Dst>(&self, transform: &ScaleOffset2D<f32, U, Dst>) -> NinePatch<Dst> {
        let mut top_left = transform.transform_point(point(self.left, self.top));
        let mut bottom_right = transform.transform_point(point(self.right, self.bottom));

        if transform.sx < 0.0 {
            std::mem::swap(&mut top_left.x, &mut bottom_right.x);
        }
        if transform.sy < 0.0 {
            std::mem::swap(&mut top_left.y, &mut bottom_right.y);
        }

        NinePatch {
            bounds: transform.transform_box(&self.bounds),
            left: top_left.x,
            top: top_left.y,
            right: bottom_right.x,
            bottom: bottom_right.y,
        }
    }

    pub fn bounds(&self) -> &Box2D<f32, U> {
        &self.bounds
    }

    pub fn left(&self) -> f32 { self.left - self.bounds.min.x }

    pub fn top(&self) -> f32 { self.top - self.bounds.min.y }

    pub fn right(&self) -> f32 { self.bounds.max.x - self.right }

    pub fn bottom(&self) -> f32 { self.bounds.max.y - self.bottom }

    pub fn set_left(&mut self, left: f32) {
        self.left = self.bounds.min.x + left;
    }

    pub fn set_top(&mut self, top: f32) {
        self.top = self.bounds.min.y + top;
    }

    pub fn set_right(&mut self, right: f32) {
        self.right = self.bounds.max.x - right;
    }

    pub fn set_bottom(&mut self, bottom: f32) {
        self.bottom = self.bounds.max.y - bottom;
    }

    pub fn for_each_segment(
        &self,
        cb: &mut impl FnMut(&Box2D<f32, U>, Sides)
    ) {
        // Top row.
        let segment = Box2D {
            min: self.bounds.min,
            max: point(self.left, self.top),
        };
        if !segment.is_empty() {
            cb(&segment, Sides::TOP | Sides::LEFT)
        }

        let segment = Box2D {
            min: point(self.left, self.bounds.min.y),
            max: point(self.right, self.top),
        };
        if !segment.is_empty() {
            cb(&segment, Sides::TOP)
        }

        let segment = Box2D {
            min: point(self.right, self.bounds.min.y),
            max: point(self.bounds.max.x, self.top),
        };
        if !segment.is_empty() {
            cb(&segment, Sides::TOP | Sides::RIGHT)
        }

        // Mid row.
        let segment = Box2D {
            min: point(self.bounds.min.x, self.top),
            max: point(self.left, self.bottom),
        };
        if !segment.is_empty() {
            cb(&segment, Sides::LEFT)
        }

        let segment = Box2D {
            min: point(self.left, self.top),
            max: point(self.right, self.bottom),
        };
        if !segment.is_empty() {
            cb(&segment, Sides::NONE)
        }

        let segment = Box2D {
            min: point(self.right, self.top),
            max: point(self.bounds.max.x, self.bottom),
        };
        if !segment.is_empty() {
            cb(&segment, Sides::RIGHT)
        }

        // Bottom row.
        let segment = Box2D {
            min: point(self.bounds.min.x, self.bottom),
            max: point(self.left, self.bounds.max.y),
        };
        if !segment.is_empty() {
            cb(&segment, Sides::BOTTOM | Sides::LEFT)
        }

        let segment = Box2D {
            min: point(self.left, self.bottom),
            max: point(self.right, self.bounds.max.y),
        };
        if !segment.is_empty() {
            cb(&segment, Sides::BOTTOM)
        }

        let segment = Box2D {
            min: point(self.right, self.bottom),
            max: self.bounds.max,
        };
        if !segment.is_empty() {
            cb(&segment, Sides::BOTTOM | Sides::RIGHT)
        }
    }

    pub fn get_segment(&self, sides: Sides) -> Box2D<f32, U> {
        match sides {
            Sides::TOP_LEFT => Box2D {
                min: self.bounds.min,
                max: point(self.left, self.top),
            },
            Sides::TOP => Box2D {
                min: point(self.left, self.bounds.min.y),
                max: point(self.right, self.top),
            },
            Sides::TOP_RIGHT => Box2D {
                min: point(self.right, self.bounds.min.y),
                max: point(self.bounds.max.x, self.top),
            },
            Sides::LEFT => Box2D {
                min: point(self.bounds.min.x, self.top),
                max: point(self.left, self.bottom),
            },
            Sides::NONE => Box2D {
                min: point(self.left, self.top),
                max: point(self.right, self.bottom),
            },
            Sides::RIGHT => Box2D {
                min: point(self.right, self.top),
                max: point(self.bounds.max.x, self.bottom),
            },
            Sides::BOTTOM_LEFT => Box2D {
                min: point(self.bounds.min.x, self.bottom),
                max: point(self.left, self.bounds.max.y),
            },
            Sides::BOTTOM => Box2D {
                min: point(self.left, self.bottom),
                max: point(self.right, self.bounds.max.y),
            },
            Sides::BOTTOM_RIGHT => Box2D {
                min: point(self.right, self.bottom),
                max: self.bounds.max,
            },
            _ => {
                panic!("Invalid ninepatch side");
            }
        }
    }

    pub fn is_valid(&self) -> bool {
        (self.bounds.max.x >= self.bounds.min.x)
            & (self.bounds.max.y >= self.bounds.min.y)
            & (self.left > self.bounds.min.x)
            & (self.left < self.bounds.max.x)
            & (self.top > self.bounds.min.y)
            & (self.top < self.bounds.max.y)
            & (self.right > self.bounds.min.x)
            & (self.right < self.bounds.max.x)
            & (self.bottom > self.bounds.min.y)
            & (self.bottom < self.bounds.max.y)
    }
}

#[test]
fn test_segments() {
    let ninepatch = LocalNinePatch::new(
        crate::units::LocalRect {
            min: point(-20.0, 20.0),
            max: point(120.0, 195.0),
        },
        10.0, 20.0, 30.0, 40.0
    );

    ninepatch.for_each_segment(&mut |rect, side| {
        assert_eq!(*rect, ninepatch.get_segment(side));
    });
}
