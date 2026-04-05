# Vger renderer

A path renderer ported from https://github.com/audulus/vger-rs

Paths segments are decomposed into y-monotonic quadratic bézier segments and binned into horizontal bands on the CPU. Bands are rendered on the GPU via instanced quads. The fragment shader goes over all curves in its band and evaluates coverage.

A few interesting things about vger:

- The approach for evaluating coverage in the fragment shader:
  - A ray-line intersection against the baseline of the curve followed by loop-blinn's trick to check whether the pixel is between the curve and its baseline.
  - AA is evaluated using the quadratic bézier distance approximation from the RAVG paper.
- It does not use flatten curves (the shader deals with y-monotonic quadratic béziers).
- It strives to do little work on the CPU (in theory. The current implementation actually spends as much as the tiling renderer binning the curves. I expect that this can be improved). I expect that with a bit of work the CPU overhead of this approach should be somewhere halfway between the stencil renderer and the tiling renderer.

# TODO

- Non-zero fill rule.
- Render task clip.
- Faster binning on the CPU.
- A multi-sampled option for anti-alasing.


# License

[MIT License](https://github.com/audulus/vger-rs/blob/057bf36539ee6d40fc33c1fa88282b05ef619a90/LICENSE#L1)
