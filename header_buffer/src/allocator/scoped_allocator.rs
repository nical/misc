//! Bump allocator with stack-discipline rollback.
//!
//! [`ScopedAllocatorStorage`] wraps a [`BumpAllocatorStorage`] and exposes a [`scope`]
//! method. Allocations made inside the callback are reclaimed when the
//! callback returns; the underlying bump cursor is rewound to its prior
//! position and any chunks acquired during the scope are returned to the
//! chunk pool. Scopes can be nested via [`Scope::scope`].
//!
//! Allocator handles obtained from a [`Scope`] are bounded to the scope's
//! borrow, so the borrow checker prevents them (and any collections holding
//! them) from outliving the scope. Entering a nested scope requires `&mut`
//! access to the parent [`Scope`], so outstanding handles must be dropped
//! first.
//!
//! [`scope`]: ScopedAllocatorStorage::scope

use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::allocator::bump_allocator::{BumpAllocator, BumpAllocatorStorage, Snapshot};
use crate::allocator::chunk_pool::ChunkPool;
use crate::allocator::{AllocError, Allocator, Layout};

/// A bump allocator that supports scoped rollback.
///
/// Allocations are only available within the callback passed to
/// [`Self::scope`]. When the callback returns (normally or via panic) the
/// allocator's state is restored to what it was at scope entry.
pub struct ScopedAllocatorStorage {
    storage: BumpAllocatorStorage,
}

impl ScopedAllocatorStorage {
    pub fn new(chunks: ChunkPool) -> Self {
        ScopedAllocatorStorage {
            storage: BumpAllocatorStorage::new(chunks),
        }
    }

    /// Enter a scope. All memory allocated inside `f` is reclaimed when `f`
    /// returns.
    pub fn scope<F, R>(&mut self, f: F) -> R
    where
        F: for<'scope> FnOnce(Scope<'scope>) -> R,
    {
        run_scope(&self.storage, f)
    }
}

/// Per-scope context passed to [`ScopedAllocatorStorage::scope`] callbacks.
///
/// Use [`Self::allocator`] to obtain an [`Allocator`] handle and
/// [`Self::scope`] to enter a nested scope.
pub struct Scope<'scope> {
    storage: &'scope BumpAllocatorStorage,
}

impl<'scope> Scope<'scope> {
    /// Obtain an allocator handle valid for as long as `self` is borrowed.
    pub fn allocator(&self) -> ScopedAllocator<'_> {
        ScopedAllocator {
            inner: self.storage.allocator(),
            _phantom: PhantomData,
        }
    }

    /// Enter a nested scope. Requires `&mut self`, so any outstanding handles
    /// from `self.allocator()` (and collections holding them) must be dropped
    /// first.
    pub fn scope<F, R>(&mut self, f: F) -> R
    where
        F: for<'inner> FnOnce(Scope<'inner>) -> R,
    {
        run_scope(self.storage, f)
    }
}

/// Allocator handle for use within a [`Scope`].
///
/// Cloneable; the lifetime `'a` ties the handle to the borrow of the
/// originating [`Scope`].
#[derive(Clone)]
pub struct ScopedAllocator<'a> {
    inner: BumpAllocator,
    _phantom: PhantomData<&'a ()>,
}

unsafe impl<'a> Allocator for ScopedAllocator<'a> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.inner.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { self.inner.deallocate(ptr, layout) }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        unsafe { self.inner.grow(ptr, old_layout, new_layout) }
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        unsafe { self.inner.shrink(ptr, old_layout, new_layout) }
    }
}

fn run_scope<F, R>(storage: &BumpAllocatorStorage, f: F) -> R
where
    F: for<'scope> FnOnce(Scope<'scope>) -> R,
{
    struct Guard<'s> {
        storage: &'s BumpAllocatorStorage,
        snapshot: Snapshot,
    }
    impl Drop for Guard<'_> {
        fn drop(&mut self) {
            unsafe { self.storage.restore(&self.snapshot) }
        }
    }

    let _guard = Guard {
        storage,
        snapshot: storage.snapshot(),
    };
    let scope = Scope { storage };
    f(scope)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vector;

    fn make_alloc() -> ScopedAllocatorStorage {
        ScopedAllocatorStorage::new(ChunkPool::new())
    }

    #[test]
    fn empty_scope() {
        let mut alloc = make_alloc();
        alloc.scope(|_scope| {});
    }

    #[test]
    fn scope_returns_value() {
        let mut alloc = make_alloc();
        let sum = alloc.scope(|scope| {
            let mut v: Vector<i32, _> = Vector::new_in(scope.allocator());
            for i in 0..10 {
                v.push(i);
            }
            v.iter().sum::<i32>()
        });
        assert_eq!(sum, 45);
    }

    #[test]
    fn cursor_rewinds_after_scope() {
        let mut alloc = make_alloc();

        let cursor_before = alloc.storage.snapshot();
        alloc.scope(|scope| {
            let mut v: Vector<u64, _> = Vector::new_in(scope.allocator());
            for i in 0..32 {
                v.push(i);
            }
        });
        let cursor_after = alloc.storage.snapshot();

        assert_eq!(cursor_before.cursor_addr(), cursor_after.cursor_addr());
    }

    #[test]
    fn nested_scopes() {
        let mut alloc = make_alloc();
        alloc.scope(|mut outer| {
            let mut ov: Vector<i32, _> = Vector::new_in(outer.allocator());
            ov.extend_from_slice(&[1, 2, 3]);
            assert_eq!(&ov[..], &[1, 2, 3]);
            drop(ov);

            outer.scope(|inner| {
                let mut iv: Vector<i32, _> = Vector::new_in(inner.allocator());
                iv.extend_from_slice(&[10, 20, 30]);
                assert_eq!(&iv[..], &[10, 20, 30]);
            });

            let mut ov2: Vector<i32, _> = Vector::new_in(outer.allocator());
            ov2.extend_from_slice(&[4, 5]);
            assert_eq!(&ov2[..], &[4, 5]);
        });
    }

    #[test]
    fn scope_spans_multiple_chunks() {
        // Force enough allocations to overflow into additional chunks so the
        // restore path actually has chunks to recycle.
        let mut alloc = make_alloc();
        for _ in 0..3 {
            alloc.scope(|scope| {
                let a = scope.allocator();
                let mut bufs: Vec<Vector<u8, _>> = Vec::new();
                for _ in 0..16 {
                    let mut v: Vector<u8, _> = Vector::with_capacity_in(64 * 1024, a.clone());
                    for i in 0..64u8 {
                        v.push(i);
                    }
                    bufs.push(v);
                }
                let s: u64 = bufs.iter().map(|b| b.iter().map(|&x| x as u64).sum::<u64>()).sum();
                assert!(s > 0);
            });
        }
    }

    #[test]
    fn vec_of_strings_dropped_at_scope_end() {
        let mut alloc = make_alloc();
        alloc.scope(|scope| {
            let mut v: Vector<String, _> = Vector::new_in(scope.allocator());
            v.push("hello".to_string());
            v.push("foo bar baz this is a longer string to avoid SSO".to_string());
            assert_eq!(v[0], "hello");
        });
        // No leak assertion is built in, but if Drop didn't run, miri/asan
        // would catch it. The storage drop also asserts allocation_count == 0.
    }

    #[test]
    fn panic_in_scope_still_restores() {
        let mut alloc = make_alloc();
        let cursor_before = alloc.storage.snapshot();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            alloc.scope(|scope| {
                let mut v: Vector<u32, _> = Vector::new_in(scope.allocator());
                for i in 0..128 {
                    v.push(i);
                }
                panic!("intentional");
            });
        }));
        assert!(result.is_err());

        let cursor_after = alloc.storage.snapshot();
        assert_eq!(cursor_before.cursor_addr(), cursor_after.cursor_addr());

        // Allocator should still be usable after a panicked scope.
        alloc.scope(|scope| {
            let mut v: Vector<u32, _> = Vector::new_in(scope.allocator());
            v.push(42);
            assert_eq!(v[0], 42);
        });
    }

}
