use std::ops::Range;
use std::sync::atomic::{AtomicI32, Ordering};

pub struct BumpAllocator {
    start: AtomicI32,
    end: AtomicI32,
}

impl BumpAllocator {
    pub fn new(range: Range<i32>) -> Self {
        BumpAllocator {
            start: AtomicI32::new(range.start),
            end: AtomicI32::new(range.end),
        }
    }

    pub fn allocate_front(&self, size: i32) -> Option<Range<i32>> {
        let start = self.start.fetch_add(size, Ordering::SeqCst);
        let end = self.end.load(Ordering::SeqCst);
        if start + size > end {
            return None;
        }

        Some(start..(start + size))
    }

    pub fn allocate_back(&self, size: i32) -> Option<Range<i32>> {
        let end = self.end.fetch_sub(size, Ordering::SeqCst);
        let start = self.start.load(Ordering::SeqCst);
        if start + size > end {
            return None;
        }

        Some((end - size)..end)
    }

    pub fn split_front(&self, size: i32) -> Option<Self> {
        Some(BumpAllocator::new(self.allocate_front(size)?))
    }

    pub fn split_back(&self, size: i32) -> Option<Self> {
        Some(BumpAllocator::new(self.allocate_back(size)?))
    }

    pub fn reset(&self, range: Range<i32>) {
        // First set an empty range to avoid races with other threads trying to allocate.
        self.start.store(self.end.load(Ordering::SeqCst), Ordering::SeqCst);

        self.start.store(range.start, Ordering::SeqCst);
        self.end.store(range.end, Ordering::SeqCst);
    }
}

