# Performance

To run the benchmarks: `cargo bench`

The benchmark will pull all of the curves from the SVG documents in the `assets` directory. Use the `FLATTEN_INPUT` environment variable to apply a filter. For example:

```bash
FLATTEN_INPUT=nehab_ cargo bench
```

Will only use the files with "nehab_" in their name.

Benchmark results on an AMD Ryzen 7 PRO 6850U laptop:

![Cubic bézier benchmark results](results-cubic-all-zen3.svg)

![Quadratic bézier benchmark results](results-quadratic-all-zen3.svg)

Benchmark results on an AMD Ryzen Threadripper PRO 3975WXs desktop:

![Cubic bézier benchmark results](results-cubic-all-threadripper.svg)

![Quadratic bézier benchmark results](results-quadratic-all-threadripper.svg)

Still on the AMD Ryzen Threadripper PRO 3975WXs desktop, let's focus on fewer algorithms and more specific test cases:

`fonts-12.svg`:

![Cubic bézier fonts benchmark results](results-cubic-fonts-threadripper.svg)

GhostScript tiger:

![Cubic bézier fonts benchmark results](results-cubic-tiger-threadripper.svg)

Nehab test set:

![Cubic bézier fonts benchmark results](results-cubic-nehab-threadripper.svg)
