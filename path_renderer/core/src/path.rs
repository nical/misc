use lyon::path::PathSlice;
use lyon::path::builder::WithSvg;
use lyon::path::path::Iter;
use lyon::path::traits::Build;
use lyon::path::{Path as PathInner, path::Builder as BuilderInner, builder::PathBuilder, Attributes, EndpointId};
use lyon::math::{Box2D, Point, Vector};

pub use lyon::path::FillRule;

type PathFlags = u32;
const CONVEX: PathFlags = 1;
const SINGLE_SUBPATH: PathFlags = 1 << 1;
const CUBIC_CURVES: PathFlags = 1 << 2;
const QUADRATIC_CURVES: PathFlags = 1 << 3;
const POSITIVE_TURNS: PathFlags = 1 << 4;
const NEGATIVE_TURNS: PathFlags = 1 << 5;

pub struct Path {
    path: PathInner,
    aabb: Box2D,
    flags: PathFlags,
}

impl Path {
    pub fn builder() -> Builder {
        Builder::new()
    }

    pub fn aabb(&self) -> &Box2D { &self.aabb }

    pub fn is_convex(&self) -> bool {
        self.flags | CONVEX != 0
    }

    pub fn is_concave(&self) -> bool {
        !self.is_convex()
    }

    pub fn has_multiple_subpaths(&self) -> bool {
        self.flags | SINGLE_SUBPATH == 0
    }

    pub fn as_slice(&self) -> PathSlice {
        self.path.as_slice()
    }

    pub fn iter(&self) -> Iter {
        self.path.iter()
    }
}

pub struct Builder {
    builder: BuilderInner,
    flags: u32,
    aabb: Box2D,
    v0: Vector,
    prev: Point,
    first: Point,
    num_subpaths: u16,
}

impl Builder {
    pub fn new() -> Self {
        Builder {
            builder: BuilderInner::new(),
            flags: 0,
            aabb: Box2D {
                min: Point::new(std::f32::MAX, std::f32::MAX),
                max: Point::new(std::f32::MIN, std::f32::MIN),
            },
            v0: Vector::new(0.0, 0.0),
            prev: Point::new(0.0, 0.0),
            first: Point::new(0.0, 0.0),
            num_subpaths: 0,
        }
    }

    pub fn with_svg(self) -> WithSvg<Self> {
        WithSvg::new(self)
    }

    pub fn segment_to(&mut self, p: Point) {
        self.aabb.min.x = self.aabb.min.x.min(p.x);
        self.aabb.min.y = self.aabb.min.y.min(p.y);
        self.aabb.max.x = self.aabb.max.x.max(p.x);
        self.aabb.max.y = self.aabb.max.y.max(p.y);
    
        let v = p - self.prev;
        let c = self.v0.cross(v);
        if c > 0.0 {
            self.flags |= POSITIVE_TURNS;
        } else if c < 0.0 {
            self.flags |= NEGATIVE_TURNS;
        }
        self.v0 = v;
        self.prev = p;
    }

    pub fn begin(&mut self, at: Point) -> EndpointId {
        self.segment_to(at);
        self.first = at;
        self.num_subpaths += 1;
        self.builder.begin(at)
    }

    pub fn end(&mut self, close: bool) {
        self.segment_to(self.first);
        self.builder.end(close);
    }

    pub fn line_to(&mut self, to: Point) -> EndpointId {
        self.segment_to(to);
        self.builder.line_to(to)
    }

    pub fn quadratic_bezier_to(
        &mut self,
        ctrl: Point,
        to: Point,
    ) -> EndpointId {
        self.segment_to(ctrl);
        self.segment_to(to);
        self.flags |= QUADRATIC_CURVES;
        self.builder.quadratic_bezier_to(ctrl, to)
    }

    pub fn cubic_bezier_to(
        &mut self,
        ctrl1: Point,
        ctrl2: Point,
        to: Point,
    ) -> EndpointId {
        self.segment_to(ctrl1);
        self.segment_to(ctrl2);
        self.segment_to(to);
        self.flags |= CUBIC_CURVES;
        self.builder.cubic_bezier_to(ctrl1, ctrl2, to)
    }

    pub fn build(self) -> Path {
        let mut flags = self.flags;
        if !((flags & POSITIVE_TURNS != 0) && (flags & NEGATIVE_TURNS != 0)) {
            flags |= CONVEX;
        }
        if self.num_subpaths < 2 {
            flags |= SINGLE_SUBPATH;
        }
        Path {
            path: self.builder.build(),
            aabb: self.aabb,
            flags,
        }
    }
}

impl PathBuilder for Builder {
    fn num_attributes(&self) -> usize { 0 }
    fn begin(&mut self, at: Point, _: Attributes<'_>) -> EndpointId {
        self.begin(at)
    }

    fn end(&mut self, close: bool) {
        self.end(close)
    }

    fn line_to(&mut self, to: Point, _: Attributes<'_>) -> EndpointId {
        self.line_to(to)
    }

    fn quadratic_bezier_to(
        &mut self,
        ctrl: Point,
        to: Point,
        _: Attributes<'_>
    ) -> EndpointId {
        self.quadratic_bezier_to(ctrl, to)
    }

    fn cubic_bezier_to(
        &mut self,
        ctrl1: Point,
        ctrl2: Point,
        to: Point,
        _: Attributes<'_>
    ) -> EndpointId {
        self.cubic_bezier_to(ctrl1, ctrl2, to)
    }
}

impl Build for Builder {
    type PathType = Path;
    fn build(self) -> Path {
        self.build()
    }
}
