use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use header_buffer::global::Global;
use header_buffer::unmanaged::{AllocInit, UnmanagedVector};

fn bench_push(c: &mut Criterion) {
    let allocator = Global;

    let counts: &[usize] = &[100, 1_000, 10_000];

    let mut group = c.benchmark_group("push");

    type Item = [u32; 8];
    fn val(i: usize) -> Item {
        black_box([i as u32; 8])
    }

    const CAP: usize = 16;

    for &n in counts {
        group.bench_with_input(BenchmarkId::new("std_vec", n), &n, |b, &n| {
            b.iter(|| {
                let mut v = Vec::with_capacity(CAP);
                for i in 0..n {
                    v.push(val(i));
                }
                v
            });
        });

        group.bench_with_input(BenchmarkId::new("unmanaged_vector", n), &n, |b, &n| {
            b.iter(|| unsafe {
                let mut v = UnmanagedVector::with_capacity_in((), CAP, AllocInit::Uninit, &allocator);
                for i in 0..n {
                    v.push(val(i), &allocator);
                }
                v.deallocate_in(&allocator);
            });
        });

        //group.bench_with_input(BenchmarkId::new("vector", n), &n, |b, &n| {
        //    b.iter(|| {
        //        let mut v = header_buffer::Vector::<Item>::with_capacity(CAP);
        //        for i in 0..n {
        //            v.push(val(i));
        //        }
        //        v
        //    });
        //});

        group.bench_with_input(BenchmarkId::new("seg_vec1", n), &n, |b, &n| {
            b.iter(|| unsafe {
                let mut v = header_buffer::seg_vec::UnmanagedSegmentedVector::with_capacity_in(CAP, &allocator);
                for i in 0..n {
                    v.push(val(i), &allocator);
                }
                v.deallocate_in(&allocator);
            });
        });

        group.bench_with_input(BenchmarkId::new("seg_vec2", n), &n, |b, &n| {
            b.iter(|| unsafe {
                let mut v = header_buffer::seg_vec2::UnmanagedSegmentedVector::with_capacity_in(CAP, &allocator);
                for i in 0..n {
                    v.push(val(i), &allocator);
                }
                v.deallocate_in(&allocator);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_push);
criterion_main!(benches);
