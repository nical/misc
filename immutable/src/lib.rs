//! Shared and mutable vectors.
//!
//! This crate provides the following two types:
//! - `SharedVector<T>`, an immutable reference counted vector with a value-oriented API.
//! - `UniqueVector<T>`, an unique vector type with an API similar to `std::Vec<T>`.
//!
//! Internally these two types share the same representation: Their value is a single pointer to a buffer containing
//! - A 16 bytes header (plus possible padding for alignment),
//! - the contiguous sequence of items of type `T`.
//!
//! This allows very cheap conversion between the two:
//! - shared to mutable: a new allocation is made only if there are other handles to the same buffer (the reference
//!   count is greather than one).
//! - mutable to shared: always free since mutable buffers are guarantted to be unique.
//!
//! # Use cases
//!
//! ## `Arc<Vec<T>>` without the indirection.
//!
//! A mutable vector can be be built using a Vec-style API, and then made immutable and reference counted for various
//! use case (easy multi-threading or simply shared ownership).
//!
//! Using the standard library one might be tempted to firs build a `Vec<T>` and share it via `Arc<Vec<T>>`. This is
//! a fine approach at the cost of an extra pointer indirection.
//! Another approach is to share it as an `Arc<[T]>` which removes the indirection at the cost of the need to copy
//! from the vector.
//!
//! Using this crate there is no extra indirection in the resulting shared vector or copy between the mutable and
//! shared versions.
//!
//! ```
//! use immutable::UniqueVector;
//! let mut builder = UniqueVector::new();
//! builder.push(1u32);
//! builder.push(2);
//! builder.push(3);
//! // Make it reference counted, no allocation.
//! let mut shared = builder.into_shared();
//! // We can now create new references
//! let shared_2 = shared.new_ref();
//! let shared_3 = shared.new_ref();
//! ```
//!
//! ## You like immutable data structures and value-oriented APIs.
//!
//! You keep telling everyone around you about the "value of values"? That's OK! Maybe you'll enjoy using the
//! `RefCountedVector` type.
//!
//! ```
//! use immutable::SharedVector;
//! let mut a = SharedVector::new();
//! a.push(1u32);
//! a.push(2);
//! a.push(3);
//!
//! // `new_ref` (you can also use `clone`) creates a second reference to the same buffer.
//! let b = a.new_ref();
//! a.push(4);     // This push needs to allocate new storage because there multiple references.
//! a.push(5);     // This one does not.
//!
//! assert_eq!(a.as_slice(), &[1, 2, 3, 4, 5]);
//! assert_eq!(b.as_slice(), &[1, 2, 3]);
//! ```
//!
//! Note that `SharedVector` is *not* a RRB vector implementation.
//!
//! ## The slim value representation
//!
//! That's certainly niche but the representation being different than `std::Vec`'s that may be good or
//! bad for you depending on what you want to do with it.
//!
//! ```ascii
//!  +---+
//!  |   | SharedVector (8 bytes on 64bit systems)
//!  +---+
//!    |
//!    v
//!  +----------++----+----+----+----+----+----+----+----+
//!  |          ||    |    |    |    |    |    |    |    |
//!  +----------++----+----+----+----+----+----+----+----+
//!   \________/  \_____________________________________/
//!     Header                  Items
//!   (16 bytes)
//! ```
//!
//! Both `SharedVector` and `ImmutableVector` contain a single pointer, so they occupy a third of the space
//! of `Vec<T>` and half the space of `Box<[T]>`. Of course that comes at a price, most methods need to
//! have to read the header located at the beginning of the buffer. That may not matter if the most common
//! operation is to iterate over vector, but large numbers of random accesses will likely be slower.
//!
//! # Limitiations
//!
//! These vector types can hold at most `u32::MAX` elements.
//!

mod raw;
mod vector;
//pub mod store;
//pub mod chunked;
//pub mod value;

pub use vector::{SharedVector, AtomicSharedVector, UniqueVector};
