use std::alloc::Layout;
use std::ptr::NonNull;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use header_buffer::allocator::bump_allocator::{BumpAllocator, BumpAllocatorStorage};
use header_buffer::allocator::chunk_pool::ChunkPool;
use header_buffer::allocator::{Allocator, Global};
use header_buffer::Vector;

type Item = [u32; 8];

#[inline]
fn val(i: usize) -> Item {
    black_box([i as u32; 8])
}

/// Raw `Allocator::allocate` throughput. Each iteration allocates `n` blocks
/// of `Item` layout. The bump allocator is reset between iterations by
/// recreating the storage (chunks are recycled via the shared pool).
fn bench_raw_allocate(c: &mut Criterion) {
    let counts: &[usize] = &[100, 1_000, 10_000];
    let layout = Layout::new::<Item>();

    let mut group = c.benchmark_group("raw_allocate");

    for &n in counts {
        group.bench_with_input(BenchmarkId::new("global", n), &n, |b, &n| {
            let global = Global;
            b.iter(|| {
                let mut ptrs: Vec<NonNull<u8>> = Vec::with_capacity(n);
                for _ in 0..n {
                    let p = global.allocate(layout).unwrap().cast();
                    ptrs.push(p);
                }
                for p in &ptrs {
                    unsafe { global.deallocate(*p, layout) };
                }
                black_box(ptrs);
            });
        });

        let pool = ChunkPool::new();
        group.bench_with_input(BenchmarkId::new("bump", n), &n, |b, &n| {
            b.iter(|| {
                let storage = BumpAllocatorStorage::new(pool.clone());
                let alloc = storage.allocator();
                let mut ptrs: Vec<NonNull<u8>> = Vec::with_capacity(n);
                for _ in 0..n {
                    let p = alloc.allocate(layout).unwrap().cast();
                    ptrs.push(p);
                }
                for p in &ptrs {
                    unsafe { alloc.deallocate(*p, layout) };
                }
                black_box(ptrs);
                drop(alloc);
                drop(storage);
            });
        });
    }

    group.finish();
}

/// `Vector::push` performance. Exercises the grow fast path of `BumpAllocator`
/// (the last allocation can be reallocated in place when there is room in the
/// current chunk).
fn bench_vec_push(c: &mut Criterion) {
    let counts: &[usize] = &[100, 1_000, 10_000];
    const CAP: usize = 16;

    let mut group = c.benchmark_group("vec_push");

    for &n in counts {
        group.bench_with_input(BenchmarkId::new("global", n), &n, |b, &n| {
            b.iter(|| {
                let mut v: Vec<Item> = Vec::with_capacity(CAP);
                for i in 0..n {
                    v.push(val(i));
                }
                v
            });
        });

        let pool = ChunkPool::new();
        group.bench_with_input(BenchmarkId::new("bump", n), &n, |b, &n| {
            b.iter(|| {
                let storage = BumpAllocatorStorage::new(pool.clone());
                let alloc = storage.allocator();
                let mut v: Vector<Item, BumpAllocator> =
                    Vector::with_capacity_in(CAP, alloc);
                for i in 0..n {
                    v.push(val(i));
                }
                drop(v);
                drop(storage);
            });
        });
    }

    group.finish();
}

/// Many small vectors built into the same arena. This is the typical use case
/// for a bump allocator: a large number of short-lived collections that all
/// share storage and are reclaimed together.
fn bench_many_small_vecs(c: &mut Criterion) {
    let configs: &[(usize, usize)] = &[(1_000, 8), (1_000, 32), (10_000, 8)];

    let mut group = c.benchmark_group("many_small_vecs");

    for &(n_vecs, len) in configs {
        let id = format!("{}x{}", n_vecs, len);

        let pool = ChunkPool::new();
        group.bench_with_input(
            BenchmarkId::new("global", &id),
            &(n_vecs, len),
            |b, &(n_vecs, len)| {
                b.iter(|| {
                    let mut all: Vec<Vec<Item>> = Vec::with_capacity(n_vecs);
                    for _ in 0..n_vecs {
                        let mut v: Vec<Item> = Vec::new();
                        for i in 0..len {
                            v.push(val(i));
                        }
                        all.push(v);
                    }
                    all
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("bump", &id),
            &(n_vecs, len),
            |b, &(n_vecs, len)| {
                b.iter(|| {
                    let storage = BumpAllocatorStorage::new(pool.clone());
                    let alloc = storage.allocator();
                    let mut all: Vec<Vector<Item, BumpAllocator>> =
                        Vec::with_capacity(n_vecs);
                    for _ in 0..n_vecs {
                        let mut v: Vector<Item, BumpAllocator> =
                            Vector::new_in(alloc.clone());
                        for i in 0..len {
                            v.push(val(i));
                        }
                        all.push(v);
                    }
                    drop(all);
                    drop(alloc);
                });
            },
        );
    }

    group.finish();
}

/// Mixed allocation sizes via the raw `Allocator` API. The bump allocator
/// should excel here since it does not need to track per-size free lists.
fn bench_mixed_sizes(c: &mut Criterion) {
    let counts: &[usize] = &[1_000, 10_000];
    let layouts: [Layout; 4] = [
        Layout::from_size_align(16, 8).unwrap(),
        Layout::from_size_align(64, 8).unwrap(),
        Layout::from_size_align(256, 8).unwrap(),
        Layout::from_size_align(1024, 8).unwrap(),
    ];

    let mut group = c.benchmark_group("mixed_sizes");

    for &n in counts {
        group.bench_with_input(BenchmarkId::new("global", n), &n, |b, &n| {
            let global = Global;
            b.iter(|| {
                let mut ptrs: Vec<(NonNull<u8>, Layout)> = Vec::with_capacity(n);
                for i in 0..n {
                    let layout = layouts[i & 3];
                    let p = global.allocate(layout).unwrap().cast();
                    ptrs.push((p, layout));
                }
                for (p, layout) in &ptrs {
                    unsafe { global.deallocate(*p, *layout) };
                }
                black_box(ptrs);
            });
        });

        let pool = ChunkPool::new();
        group.bench_with_input(BenchmarkId::new("bump", n), &n, |b, &n| {
            b.iter(|| {
                let storage = BumpAllocatorStorage::new(pool.clone());
                let alloc = storage.allocator();
                let mut ptrs: Vec<(NonNull<u8>, Layout)> = Vec::with_capacity(n);
                for i in 0..n {
                    let layout = layouts[i & 3];
                    let p = alloc.allocate(layout).unwrap().cast();
                    ptrs.push((p, layout));
                }
                for (p, layout) in &ptrs {
                    unsafe { alloc.deallocate(*p, *layout) };
                }
                black_box(ptrs);
                drop(alloc);
                drop(storage);
            });
        });
    }

    group.finish();
}

/// Cost of allocating + dropping a fresh `BumpAllocatorStorage` when chunks
/// are reused from a shared `ChunkPool` vs. when each iteration goes through
/// the system allocator. This isolates the pool's contribution.
fn bench_storage_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_reuse");

    group.bench_function("fresh_pool_per_iter", |b| {
        b.iter(|| {
            let pool = ChunkPool::new();
            let storage = BumpAllocatorStorage::new(pool);
            drop(storage);
        });
    });

    let pool = ChunkPool::new();
    // Prime the pool with one recycled chunk.
    {
        let s = BumpAllocatorStorage::new(pool.clone());
        drop(s);
    }
    group.bench_function("shared_pool", |b| {
        b.iter(|| {
            let storage = BumpAllocatorStorage::new(pool.clone());
            drop(storage);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_raw_allocate,
    bench_vec_push,
    bench_many_small_vecs,
    bench_mixed_sizes,
    bench_storage_reuse,
);
criterion_main!(benches);
