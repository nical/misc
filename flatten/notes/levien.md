# Raph Levien's flattening algorithm

[Implementation in this repository](https://github.com/nical/misc/blob/master/flatten/src/algorithms/levien.rs) ([simd version](https://github.com/nical/misc/blob/master/flatten/src/algorithms/levien_simd.rs))


Raph explains the maths behind the the flattening of quadratic bézier curves in this blog post: https://raphlinus.github.io/graphics/curves/2019/12/23/flatten-quadbez.html

Flattening cubic bézier curves is done by first approximating them with a series fo quadratic curves. However there is a twist: Rather than considering each quadratic bézier sub-curve individually, the algorithm integrates the number of edges that must be produced over multiple sub-curves using a "fractional subdivision" scheme. This avoids the need to insert split points between each sub-curve and can even skip over entire sub-curves.


# Structure of the algoritm

A very simplified version, in pseudo-code:

```rust
fn flatten_cubic(curve, tolerance, callback) {
    let quads_tolerance = tolerance * 0.1;
    let flatten_tolerance = tolerance * 0.9;

    let num_quads = num_quadratics(curve, tolerance);

    // This part is fairly arithmetic-heavy. It maps very well to SIMD.
    for i in 0..num_quads {
        sub_curves.push(flattening_params(curve, flatten_tolerance, i));
    }

    for quad in sub_curves {
        while let Some(u) = fractional_subdivision(quad) {
            // This part is also fairly arithmetic-heavy. It is a little
            // more difficult to optimize with SIMD because for some of
            // the datasets, there are few samples per sub-curves. (See
            // the stats below).
            let t = map_curve_parameter_for(u);
            let p = quad.sample(t)
            callback(p)
        }
    }

    callback(curve.to)
}
```

# Notes

This algorithm produces the "nicest" output, and is generally the most optimal in terms of number of line segments produced per curve. The up-front cost of the algorithm, however, is quite high. This cost is well amortized for curves that require a lot of line segments, but this algorithm probably does not hit the best tradeoff when dealing with a lot of very small curves that will produce few segments. See for example the [benchmark results for the font dataset](../benches/results-cubic-font-threadripper.svg). Thankfully, it should be fairly cheap to evaluate a rough proxy for the size of the curve and pick a different algorithm if it is very small.

# Stats

Some status that were collected to guide optimization:

- [Number of quadratic sub-curves per cubic curve](levien_quads_per_cubics.md)
- [Number of lines per quadratic sub-curve](levien_lines_per_quads.md)
