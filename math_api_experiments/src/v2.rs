#![allow(dead_code)]
#![allow(unused_variables)]

//! A partial implementation of some vector maths using the same type of tricks
//! that euclid currently uses to represent units.
//! The difference is that euclid currently can't express a matrix that transform
//! points from a certain unit into points with another units, for example a
//! projection matrix that would take WorldPoints and create ScreenPoints.

use std::ops;
use std::marker::PhantomData;
use std::convert::{ AsRef, From };
use num::{ One, Zero };

/// A 3d vector.
#[derive(Copy, Clone, Debug, PartialEq)]
struct Point3D<T> {
  pub x: T,
  pub y: T,
  pub z: T,
}

/// num::One requires Self to implement Mul<Self, Output=Self> which we can't
/// implement for TransformComponent, so we need a custom trait that provides
/// the same functionality.
pub trait CustomOne { fn new_one() -> Self; }
impl<T: One> CustomOne for T { fn new_one() -> T { T::one() } }

/// A 4 by 4 matrix.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Matrix4x4<T> {
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
}

impl<T> Point3D<T> {
  pub fn new(x: T, y: T, z: T) -> Point3D<T> {
    Point3D {
      x: x,
      y: y,
      z: z,
    }
  }
}

impl<T: CustomOne+Zero+Copy> Matrix4x4<T> {
  pub fn identity() -> Matrix4x4<T> {
    let one = T::new_one();
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
    }
  }
}

/// A scalar that is expressed in a certain space.
pub struct Typed<Unit, T>(pub T, PhantomData<Unit>);

impl<Unit, T: Copy> Copy for Typed<Unit, T> {}
impl<Unit, T: Copy> Clone for Typed<Unit, T> { fn clone(&self) -> Typed<Unit, T> { *self } }

impl<Unit, T: Copy> Typed<Unit, T> {
  pub fn new(val: T) -> Typed<Unit, T> { Typed(val, PhantomData) }
  pub fn inner(&self) -> T { self.0 }
}

impl<Unit, T: Copy> AsRef<T> for Typed<Unit, T> {
  fn as_ref(&self) -> &T { &self.0 }
}

impl<Unit, T: Copy> From<T> for Typed<Unit, T> {
  fn from(val: T) -> Typed<Unit, T> { Typed::new(val) }
}

impl<Unit, T: Copy + ops::Add<T, Output=T>>
ops::Add<Typed<Unit, T>>
for Typed<Unit, T> {
    type Output = Typed<Unit, T>;
    fn add(self, rhs: Typed<Unit, T>) -> Typed<Unit, T> {
      Typed::new(self.0 + rhs.0)
    }
}

impl<Unit, T: Copy + ops::Mul<Output=T>>
ops::Mul<Typed<Unit, T>>
for Typed<Unit, T> {
    type Output = Typed<Unit, T>;
    fn mul(self, rhs: Typed<Unit, T>) -> Typed<Unit, T> {
      Typed(self.0 * rhs.0, PhantomData)
    }
}

impl<Unit, T: Zero+Copy> Zero for Typed<Unit, T> {
  fn zero() -> Typed<Unit, T> { Typed::new(T::zero()) }
  fn is_zero(&self) -> bool { self.0.is_zero() }
}

impl<Unit, T: One+Copy> One for Typed<Unit, T> {
  fn one() -> Typed<Unit, T> { Typed::new(T::one()) }
}

/// A scalar that is used to transform values from a space to another.
pub struct TransformComponent<Src, Dst, T>(pub T, PhantomData<(Src, Dst)>);

impl<From, To, T: Copy> Copy for TransformComponent<From, To, T> {}
impl<From, To, T: Copy> Clone for TransformComponent<From, To, T> {
  fn clone(&self) -> TransformComponent<From, To, T> { *self }
}

impl<From, To, T: Copy> TransformComponent<From, To, T> {
  pub fn new(val: T) -> TransformComponent<From, To, T> { TransformComponent(val, PhantomData) }
  pub fn inner(&self) -> T { self.0 }
}

impl<From, To, T: Copy + ops::Add<T, Output=T>>
ops::Add<TransformComponent<From, To, T>>
for TransformComponent<From, To, T> {
    type Output = TransformComponent<From, To, T>;
    fn add(self, rhs: TransformComponent<From, To, T>) -> TransformComponent<From, To, T> {
      TransformComponent::new(self.0 + rhs.0)
    }
}

impl<From, Inter, To, T: Copy + ops::Mul<Output=T>>
ops::Mul<TransformComponent<Inter, To, T>>
for TransformComponent<From, Inter, T> {
    type Output = TransformComponent<From, To, T>;
    fn mul(self, rhs: TransformComponent<Inter, To, T>) -> TransformComponent<From, To, T> {
      TransformComponent(self.0 * rhs.0, PhantomData)
    }
}

impl<From, To, T: Copy + ops::Mul<Output=T>>
ops::Mul<Typed<From, T>>
for TransformComponent<From, To, T> {
    type Output = Typed<To, T>;
    fn mul(self, rhs: Typed<From, T>) -> Typed<To, T> {
      Typed(self.0 * rhs.0, PhantomData)
    }
}

impl<From, To, T: Zero+Copy> Zero for TransformComponent<From, To, T> {
  fn zero() -> TransformComponent<From, To, T> { TransformComponent::new(T::zero()) }
  fn is_zero(&self) -> bool { self.0.is_zero() }
}

impl<From, To, T: One+Copy> CustomOne for TransformComponent<From, To, T> {
  fn new_one() -> TransformComponent<From, To, T> { TransformComponent::new(T::one()) }
}

impl<
  T12: ops::Add<T12, Output=T12>,
  T1: Copy + ops::Mul<T2, Output=T12>,
  T2: Copy
>
ops::Mul<Matrix4x4<T2>>
for Matrix4x4<T1> {

    type Output = Matrix4x4<T12>;

    #[inline]
    fn mul(self, rhs: Matrix4x4<T2>) -> Matrix4x4<T12> {
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
        };
    }
}

impl<
  T1: Copy,
  T2: Copy + ops::Add<T2, Output=T2>,
  T12: Copy + ops::Mul<T1, Output=T2>
>
ops::Mul<Point3D<T1>>
for Matrix4x4<T12> {

    type Output = Point3D<T2>;

    #[inline]
    fn mul(self, p: Point3D<T1>) -> Point3D<T2> {
        return Point3D::new(
            self.m11 * p.x + self.m21 * p.y + self.m31 * p.z,
            self.m12 * p.x + self.m22 * p.y + self.m32 * p.z,
            self.m13 * p.x + self.m23 * p.y + self.m33 * p.z
        );
    }
}

struct Untyped;
struct World;
struct Screen;

type WorldCoordinate = Typed<World, f32>;
type ScreenCoordinate = Typed<Screen, f32>;
type WorldPoint = Point3D<WorldCoordinate>;
type ScreenPoint = Point3D<ScreenCoordinate>;

type WorldTransform = TransformComponent<World, World, f32>;
type ScreenTransform = TransformComponent<Screen, Screen, f32>;
type WorldMat = Matrix4x4<WorldTransform>;
type ScreenMat = Matrix4x4<ScreenTransform>;

type WorldToScreen = TransformComponent<World, Screen, f32>;
type ProjMat = Matrix4x4<WorldToScreen>;

fn world(val: f32) -> WorldCoordinate { WorldCoordinate::new(val) }
fn screen(val: f32) -> ScreenCoordinate { ScreenCoordinate::new(val) }



// ----------------------------------------------------------------------------

// The example below shows that if you don't care about units, using this approach
// does not get in the way. In fact, it is identical to the code in v1.rs.
type Point = Point3D<f32>;

fn times_two(p: Point) -> Point {
  Point::new(p.x * 2.0, p.y * 2.0, p.z * 2.0)
}

// The two functions below illustrate a pretty common use case: A third party library
// provides some routines that operate on f32 values and we want to use them with our
// point struct (independently of the unit).
// The way we implement units here gets in the way of readability and forces us use
// more generic boilerplate than the approached showed in v1.rs.
use third_party::f32_thing;

/// Implement a function that operate on any Point3D type using f32
/// regardless of the unit.
/// Requires AsRef and From to be able to do anything meaningful with the
/// point's content.
fn generic_over_the_unit_f32<T: AsRef<f32>+From<f32>>(point: Point3D<T>) -> Point3D<T> {
  Point3D::new(T::from(f32_thing(*point.x.as_ref())),
               T::from(f32_thing(*point.y.as_ref())),
               T::from(f32_thing(*point.x.as_ref())))
}

/// Alternative implementation of the above function.
/// A bit less generic-heavy, but not very good since it doesn't work
/// with a simple Point3D<f32>, and still requires you to wrap and unwrap
/// your f32 values.
fn generic_over_the_unit_f32_v2<Unit>(point: Point3D<Typed<Unit, f32>>) -> Point3D<Typed<Unit, f32>> {
  Point3D::new(Typed::new(f32_thing(point.x.inner())),
               Typed::new(f32_thing(point.y.inner())),
               Typed::new(f32_thing(point.x.inner())))
}

// Same as above, except this time we use World and Screen spaces, and a
// projection patrix (World -> Screen) to transform between the two in a
// type-safe way.
#[test]
fn simple_no_unit() {
  type Mat4 = Matrix4x4<f32>;
  type Point = Point3D<f32>;

  let m1 = Mat4::identity();
  let m2 = Mat4::identity();
  let m3 = Mat4::identity();

  let proj = Mat4::identity();

  let world_pos = m1 * Point::new(1.0, 2.0, 3.0);

  let screen_pos = m2 * m3 * proj * world_pos;

  let m4 = Mat4::identity();

  let screen_pos2 = m4 * screen_pos;

  // These compile, but they are clearly logic errors:
  let error1 = proj * screen_pos; // project twice !?
  let error2 = proj * proj; // project twice !?
}

#[test]
fn simple_with_units() {
  let m1 = WorldMat::identity();
  let m2 = WorldMat::identity();
  let m3 = WorldMat::identity();

  // Takes WorldPoints and produces ScreenPoints
  let proj = ProjMat::identity();

  // world_pos is a WorldPoint
  let world_pos = m1 * WorldPoint::new(world(1.0), world(2.0), world(3.0));

  // screen_pos is a ScreenPoint.
  let screen_pos = m2 * m3 * proj * world_pos;

  // You can transform screen_pos with ScreenMat matrices but not WorldMat ones.
  let m4 = ScreenMat::identity();
  let screen_pos2 = m4 * screen_pos;

  // These do not compile, thanks to the type system.
  // let error1 = proj * screen_pos; // project twice !?
  // let error2 = proj * proj; // project twice !?
}

// The tests below don't build. They are meant to see what kind of error message
// ones gets when messing the units up.
// The main difference with the approach proposed in v1.rs is that error messages don't refer
// to the matrix, but to the TransformComponent and Typed wrappers, whereas in v1.rs the same
// error messages refer to Matrix4x4 and Point3D.

/*
#[test]
fn wrong_matrix_space() {

  let a = WorldMat::identity();
  let b = ScreenMat::identity();

  let c = a * b;

  // src/v2.rs:291:11: 291:16 error: the trait bound `v2::TransformComponent<v2::World, v2::World, f32>: std::ops::Mul<v2::TransformComponent<v2::Screen, v2::Screen, f32>>` is not satisfied [E0277]
  // src/v2.rs:291   let c = a * b;
  //                         ^~~~~
  // src/v2.rs:291:11: 291:16 help: run `rustc --explain E0277` to see a detailed explanation
  // src/v2.rs:291:11: 291:16 help: the following implementations were found:
  // src/v2.rs:291:11: 291:16 help:   <v2::TransformComponent<From, Inter, T> as std::ops::Mul<v2::TransformComponent<Inter, To, T>>>
  // src/v2.rs:291:11: 291:16 help:   <v2::TransformComponent<From, To, T> as std::ops::Mul<v2::Typed<From, T>>>
}
*/

/*
#[test]
fn wrong_vector_space() {

  let mat = WorldMat::identity();
  let p = ScreenPoint::new(screen(0.0), screen(1.0), screen(2.0));

  let _ = mat * p;

  // src/v2.rs:300:11: 300:18 error: the trait bound `v2::TransformComponent<v2::World, v2::World, f32>: std::ops::Mul<v2::Typed<v2::Screen, f32>>` is not satisfied [E0277]
  // src/v2.rs:300   let _ = mat * p;
  //                         ^~~~~~~
  // src/v2.rs:300:11: 300:18 help: run `rustc --explain E0277` to see a detailed explanation
  // src/v2.rs:300:11: 300:18 help: the following implementations were found:
  // src/v2.rs:300:11: 300:18 help:   <v2::TransformComponent<From, Inter, T> as std::ops::Mul<v2::TransformComponent<Inter, To, T>>>
  // src/v2.rs:300:11: 300:18 help:   <v2::TransformComponent<From, To, T> as std::ops::Mul<v2::Typed<From, T>>>
}
*/

/*
#[test]
fn wrong_matrix_Typed() {
  // This one is a tad particular: we create a matrix using Typed instead of
  // TransformComponent. It's natural thing to do since it's what we do with points.
  // It works for Matrix-Point multiplications but breaks when multiplying the
  // matrix with a matrix that actually uses TransformComponent.
  type WorldMat2 = Matrix4x4<Typed<World, f32>>;

  let world_mat = WorldMat2::identity();
  let proj = ProjMat::identity();

  let _ = world_mat * proj;

  // src/v2.rs:312:11: 312:27 error: the trait bound `v2::Typed<v2::World, f32>: std::ops::Mul<v2::TransformComponent<v2::World, v2::Screen, f32>>` is not satisfied [E0277]
  // src/v2.rs:312   let _ = world_mat * proj;
  //                         ^~~~~~~~~~~~~~~~
  // src/v2.rs:312:11: 312:27 help: run `rustc --explain E0277` to see a detailed explanation
  // src/v2.rs:312:11: 312:27 help: the following implementations were found:
  // src/v2.rs:312:11: 312:27 help:   <v2::Typed<Unit, T> as std::ops::Mul>
}
*/