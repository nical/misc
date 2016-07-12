#![allow(dead_code)]
#![allow(unused_variables)]

//! A partial implementation of some vector maths using the same type of tricks
//! that gecko currently uses to represent units.
//! The idea is to compare with the approach in v2.rs which is closer to what
//! euclid does.

use std::ops;
use std::marker::PhantomData;
use num::{ One, Zero };

/// A 3d vector.
pub struct Point3D<T, Unit = Untyped> {
  pub x: T,
  pub y: T,
  pub z: T,
  _unit: PhantomData<Unit>,
}

/// A 4 by 4 matrix that can represent a transformation from a space to another.
pub struct Matrix4x4<T, From = Untyped, To = From> {
  pub m11 : T,
  pub m12 : T,
  pub m13 : T,
  pub m14 : T,
  pub m21 : T,
  pub m22 : T,
  pub m23 : T,
  pub m24 : T,
  pub m31 : T,
  pub m32 : T,
  pub m33 : T,
  pub m34 : T,
  pub m41 : T,
  pub m42 : T,
  pub m43 : T,
  pub m44 : T,
  _unit: PhantomData<(From, To)>,
}

impl<T: Copy, Unit> Copy for Point3D<T, Unit> {}
impl<T: Copy, Unit> Clone for Point3D<T, Unit> { fn clone(&self) -> Point3D<T, Unit> { *self } }

impl<T, Unit> Point3D<T, Unit> {
  pub fn new(x: T, y: T, z: T) -> Point3D<T, Unit> {
    Point3D {
      x: x,
      y: y,
      z: z,
      _unit: PhantomData,
    }
  }
}

impl<T: One+Zero+Copy, From, To> Matrix4x4<T, From, To> {
  pub fn identity() -> Matrix4x4<T, From, To> {
    let one = T::one();
    let zero = T::zero();
    Matrix4x4 {
      m11: one,
      m12: zero,
      m13: zero,
      m14: zero,
      m21: zero,
      m22: one,
      m23: zero,
      m24: zero,
      m31: zero,
      m32: zero,
      m33: one,
      m34: zero,
      m41: zero,
      m42: zero,
      m43: zero,
      m44: one,
      _unit: PhantomData,
    }
  }
}

impl<T: Copy, From, To> Copy for Matrix4x4<T, From, To> {}
impl<T: Copy, From, To> Clone for Matrix4x4<T, From, To> { fn clone(&self) -> Matrix4x4<T, From, To> { *self } }

impl<T: Copy + ops::Mul<Output=T> + ops::Add<T, Output=T>, From, Inter, To>
ops::Mul<Matrix4x4<T, Inter, To>>
for Matrix4x4<T, From, Inter> {

    type Output = Matrix4x4<T, From, To>;

    #[inline]
    fn mul(self, rhs: Matrix4x4<T, Inter, To>) -> Matrix4x4<T, From, To> {
        return Matrix4x4 {
            m11: self.m11 * rhs.m11 + self.m12 * rhs.m21 + self.m13 * rhs.m31 + self.m14 * rhs.m41,
            m21: self.m21 * rhs.m11 + self.m22 * rhs.m21 + self.m23 * rhs.m31 + self.m24 * rhs.m41,
            m31: self.m31 * rhs.m11 + self.m32 * rhs.m21 + self.m33 * rhs.m31 + self.m34 * rhs.m41,
            m41: self.m41 * rhs.m11 + self.m42 * rhs.m21 + self.m43 * rhs.m31 + self.m44 * rhs.m41,
            m12: self.m11 * rhs.m12 + self.m12 * rhs.m22 + self.m13 * rhs.m32 + self.m14 * rhs.m42,
            m22: self.m21 * rhs.m12 + self.m22 * rhs.m22 + self.m23 * rhs.m32 + self.m24 * rhs.m42,
            m32: self.m31 * rhs.m12 + self.m32 * rhs.m22 + self.m33 * rhs.m32 + self.m34 * rhs.m42,
            m42: self.m41 * rhs.m12 + self.m42 * rhs.m22 + self.m43 * rhs.m32 + self.m44 * rhs.m42,
            m13: self.m11 * rhs.m13 + self.m12 * rhs.m23 + self.m13 * rhs.m33 + self.m14 * rhs.m43,
            m23: self.m21 * rhs.m13 + self.m22 * rhs.m23 + self.m23 * rhs.m33 + self.m24 * rhs.m43,
            m33: self.m31 * rhs.m13 + self.m32 * rhs.m23 + self.m33 * rhs.m33 + self.m34 * rhs.m43,
            m43: self.m41 * rhs.m13 + self.m42 * rhs.m23 + self.m43 * rhs.m33 + self.m44 * rhs.m43,
            m14: self.m11 * rhs.m14 + self.m12 * rhs.m24 + self.m13 * rhs.m34 + self.m14 * rhs.m44,
            m24: self.m21 * rhs.m14 + self.m22 * rhs.m24 + self.m23 * rhs.m34 + self.m24 * rhs.m44,
            m34: self.m31 * rhs.m14 + self.m32 * rhs.m24 + self.m33 * rhs.m34 + self.m34 * rhs.m44,
            m44: self.m41 * rhs.m14 + self.m42 * rhs.m24 + self.m43 * rhs.m34 + self.m44 * rhs.m44,
            _unit: PhantomData
        };
    }
}

impl<T: Copy + ops::Mul<Output=T> + ops::Add<T, Output=T>, From, To>
ops::Mul<Point3D<T, From>>
for Matrix4x4<T, From, To> {

    type Output = Point3D<T, To>;

    #[inline]
    fn mul(self, p: Point3D<T, From>) -> Point3D<T, To> {
        return Point3D::new(
            p.x * self.m11 + p.y * self.m21 + p.z * self.m31,
            p.x * self.m12 + p.y * self.m22 + p.z * self.m32,
            p.x * self.m13 + p.y * self.m23 + p.z * self.m33
        );
    }
}

pub struct Untyped;
pub struct World;
pub struct Screen;

type WorldPoint = Point3D<f32, World>;
type ScreenPoint = Point3D<f32, Screen>;

type WorldMat = Matrix4x4<f32, World>;
type ScreenMat = Matrix4x4<f32, Screen>;
type ProjMat = Matrix4x4<f32, World, Screen>;



// ----------------------------------------------------------------------------

// The example below shows that if you don't care about units, using this approach
// does not get in the way. In fact, it is identical to the code in v2.rs.
type Point = Point3D<f32>;

fn times_two(p: Point) -> Point {
  Point::new(p.x * 2.0, p.y * 2.0, p.z * 2.0)
}

// The function below illustrate a pretty common use case: A third party library
// provides some routines that operate on f32 values and we want to use them with our
// point struct (independently of the unit).
use third_party::f32_thing;

/// Implement a function that operate on any Point3D type using f32
/// regardless of the unit
fn generic_over_the_unit_f32<Unit>(point: Point3D<f32, Unit>) -> Point3D<f32, Unit> {
  Point3D::new(f32_thing(point.x), f32_thing(point.y), f32_thing(point.x))
}

// Some random code that uses matrices and points without units.
#[test]
fn simple_no_unit() {
  type Mat4 = Matrix4x4<f32>;
  type Point = Point3D<f32>;

  let m1 = Mat4::identity();
  let m2 = Mat4::identity();
  let m3 = Mat4::identity();

  let proj = Mat4::identity();

  let world_pos = Point::new(1.0, 2.0, 3.0);

  let m123 = m1 * m2 * m3;
  let screen_pos = (m123 * proj) * world_pos;

  let m4 = Mat4::identity();

  let screen_pos2 = m4 * screen_pos;

  // These compile, but they are clearly logic errors:
  let error1 = proj * screen_pos; // project twice !?
  let error2 = proj * proj; // project twice !?
}

// Same as above, except this time we use World and Screen spaces, and a
// projection patrix (World -> Screen) to transform between the two in a
// type-safe way.
#[test]
fn simple_with_units() {

  let m1 = WorldMat::identity();
  let m2 = WorldMat::identity();
  let m3 = WorldMat::identity();

  // Takes WorldPoints and produces ScreenPoints
  let proj = ProjMat::identity();

  // world_pos is a WorldPoint
  let world_pos = WorldPoint::new(1.0, 2.0, 3.0);

  // screen_pos is a ScreenPoint.
  let screen_pos = m1 * m2 * m3 * proj * world_pos;

  // You can transform screen_pos with ScreenMat matrices but not WorldMat ones.
  let m4 = ScreenMat::identity();
  let screen_pos2 = m4 * screen_pos;

  // These do not compile, thanks to the type system.
  // let error1 = proj * screen_pos; // project twice !?
  // let error2 = proj * proj; // project twice !?
}

// The tests below don't build. They are meant to see what kind of error message
// ones gets when messing the units up.
//
// (See the comment in v2.rs)
//

/*
#[test]
fn wrong_matrix_space() {

  let a = WorldMat::identity();
  let b = ScreenMat::identity();

  let c = a * b;

  //src/v1.rs:216:11: 216:16 error: the trait bound `v1::Matrix4x4<f32, v1::World>: std::ops::Mul<v1::Matrix4x4<f32, v1::Screen>>` is not satisfied [E0277]
  //src/v1.rs:216   let c = a * b;
  //                        ^~~~~
  //src/v1.rs:216:11: 216:16 help: run `rustc --explain E0277` to see a detailed explanation
  //src/v1.rs:216:11: 216:16 help: the following implementations were found:
  //src/v1.rs:216:11: 216:16 help:   <v1::Matrix4x4<T, From, Inter> as std::ops::Mul<v1::Matrix4x4<T, Inter, To>>>
  //src/v1.rs:216:11: 216:16 help:   <v1::Matrix4x4<T, From, To> as std::ops::Mul<v1::Point3D<T, From>>>
}
*/

/*
#[test]
fn wrong_vector_space() {

  let mat = WorldMat::identity();
  let p = ScreenPoint::new(0.0, 1.0, 2.0);

  let _ = mat * p;

  //src/v1.rs:235:11: 235:18 error: the trait bound `v1::Matrix4x4<f32, v1::World>: std::ops::Mul<v1::Point3D<f32, v1::Screen>>` is not satisfied [E0277]
  //src/v1.rs:235   let _ = mat * p;
  //                        ^~~~~~~
  //src/v1.rs:235:11: 235:18 help: run `rustc --explain E0277` to see a detailed explanation
  //src/v1.rs:235:11: 235:18 help: the following implementations were found:
  //src/v1.rs:235:11: 235:18 help:   <v1::Matrix4x4<T, From, Inter> as std::ops::Mul<v1::Matrix4x4<T, Inter, To>>>
  //src/v1.rs:235:11: 235:18 help:   <v1::Matrix4x4<T, From, To> as std::ops::Mul<v1::Point3D<T, From>>>
}
*/
