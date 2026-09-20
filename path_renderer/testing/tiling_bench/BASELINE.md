# tiling_bench baselines

Headless CPU-side benchmark for the tiling renderer. Run with:

```
cargo run --release --bin tiling_bench -- demo/tiger.svg --frames 100 [--scale S] [--tolerance T] [--no-occlusion] [--top-paths N]
```

The numbers below are for comparing before/after a change on the same
machine. Re-measure both sides when the machine or config changes.

## 2026-09-20, branch tiler-perf (587669e + bench scaffolding)

100 frames (20 for paris-30k), tolerance 0.25, cpu occlusion on, default scale.

| scene | viewport | avg ms/frame | opaque tiles | mask tiles | edge bytes |
|---|---|---|---|---|---|
| tiger.svg | 900x900 | 1.01 | 443 | 4558 | 75860 |
| tiger.svg --scale 4 | 3600x3600 | 3.25 | 4422 | 18967 | 225256 |
| gearflowers.svg | 1366x1024 | 1.39 | 51 | 8657 | 104856 |
| paris-30k.svg | 5270x3594 | 53.4 | 532 | 131745 | 4368104 |

The tile/edge counts double as a cheap regression check: an optimization
that changes them is changing behavior, not just speed.
