use std::sync::Arc;

use crate::{
    path::{Path, FillRule},
    units::{LocalRect, LocalPoint, point, vector}
};

pub struct PathShape {
    pub path: Arc<Path>,
    pub fill_rule: FillRule,
    // TODO: maybe move this out of the shape.
    pub inverted: bool,
}

impl PathShape {
    pub fn new(path: Arc<Path>) -> Self {
        PathShape { path, fill_rule: FillRule::NonZero, inverted: false }
    }
    pub fn with_fill_rule(mut self, fill_rule: FillRule) -> Self {
        self.fill_rule = fill_rule;
        self
    }
    pub fn inverted(mut self) -> Self {
        self.inverted = !self.inverted;
        self
    }
    pub fn aabb(&self) -> LocalRect {
        if self.inverted {
            use std::f32::{MIN, MAX};
            return LocalRect {
                min: point(MIN, MIN),
                max: point(MAX, MAX)
            }
        }

        *self.path.aabb()
    }
}

impl Into<PathShape> for Path {
    fn into(self) -> PathShape {
        PathShape::new(Arc::new(self))
    }
}

impl Into<PathShape> for Arc<Path> {
    fn into(self) -> PathShape {
        PathShape::new(self)
    }
}


#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Circle {
    pub center: LocalPoint,
    pub radius: f32,
    pub inverted: bool,
}

impl Circle {
    pub fn new(center: LocalPoint, radius: f32) -> Self {
        Circle { center, radius, inverted: false }
    }

    pub fn inverted(mut self) -> Self {
        self.inverted = !self.inverted;
        self
    }

    pub fn aabb(&self) -> LocalRect {
        if self.inverted {
            use std::f32::{MIN, MAX};
            return LocalRect {
                min: point(MIN, MIN),
                max: point(MAX, MAX),
            }                
        }
        LocalRect {
            min: self.center - vector(self.radius, self.radius),
            max: self.center + vector(self.radius, self.radius),
        }
    }
}
