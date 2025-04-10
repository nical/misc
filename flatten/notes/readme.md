# Algorthms

See [algorithms.md](algorithms.md)

# Performance

Benchmark results:
- Cubic bézier curves:
  - [AMD Ryzen Threadripper PRO 3975WXs desktop](bench-cubic-threadripper.md)
  - Apple M1 Max laptop (TODO)
- Quadratic bézier curves:
  - [AMD Ryzen Threadripper PRO 3975WXs desktop](bench-quadratic-threadripper.md)
  - Apple M1 Max laptop (TODO)

# Flattening quality

See [a comparison of the number of generated line segments for each test case](edge_count.md)

# Datasets

[Dataset page](../assets/readme.md)

# Visualization

![A visualization of the flattening for a few algorithms and curves](cubic-vis.svg)

To generate this image, modify [`src/show.rs`](../src/show.rs) and run:

```
FLATTEN_OUTPUT=visualization.svg cargo test --release -- print_cubic --nocapture
```
