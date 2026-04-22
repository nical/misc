# Bands renderer

A path renderer inspired from [https://github.com/audulus/vger-rs](vger-rs), but using a different binning method.

Paths segments are decomposed into y-monotonic quadratic bézier segments and binned into horizontal bands on the CPU. Bands are rendered on the GPU via instanced quads. The fragment shader goes over all curves in its band and evaluates coverage.

The main differences between this implementation and vger is that the edges are binned in fixed height bands in screen space. This removes the ability to vary the band hight based on edge density, but makes the binning process faster, mostly thanks to not having to sort the edges.

# License

[MIT License](https://github.com/audulus/vger-rs/blob/057bf36539ee6d40fc33c1fa88282b05ef619a90/LICENSE#L1)
