# Various notes

## general architecture

### Tiling

Similar to pathfinder which this is inspired from, the general idea is to break complex shapes into grids of small (16x16 pixels) tiles.
This is done via a "tiling" routine happening on the CPU.

I wrote about pathfinder a while back here: [https://nical.github.io/posts/a-look-at-pathfinder.html]((link)).
The post provides some context and motivation around tiled rendering for vector graphics.

In a nutshell:
 - The expensive parts of doing rasterizing paths are:
   - (A) the large amount of pixel memory that need to be written to (problem often referred to as "overdraw").
   - (B) the math related to dealing with edges and anti-aliasing.
   - (C) transfering large amounts of data to the GPU
 - Large paths (which are typically expensive to rasterize on the CPU) tend to have a lot of fully covered tiles that contain no edges when broken up into small tiles. Taking advantage of this is key to reducing the cost of (B).
 - More generally breaking large paths into many tiles with fewer edges per tiles takes a very complex problem and breaks it into much simpler ones. It is also key to making good use of the GPU's parrallel programming model.
 - On GPU we have access to a lot more memory bandwidth than on CPU if used well (good for addressing (A)).
 - Regular grids have plenty of useful properties. One of them is that it can be used for occlusion culling and further reduce memory bandwidth (good for (A) again).
 - The sparse tile grid reprensentation is very good at reducing the amount of data to transfer to the GPU (C).

On the GPU side a paths are first rasterized in a mask atlas texture in a single draw call. Then the main render pass which first draws all opaque tiles in a draw call via a simple shader with blending disabled, and all masked tiles in a second draw call whic reads from the mask and applies a pattern.

### Departures from Pathfinder

While this work is inspried from pathfinder it has a few key differences:

On the GPU side:

- Pathfinder renders masks into a float texture  using a quad per edge with addirive blending to build winding numbers.
- This renders the mask in an alpha texture with a quad per tile, looping over edges in the fragment shader. This further reduces pressure on the blending hardware / fill rate which tends to be the bottleneck.

The CPU side:
 - This prototypes binning algorithm first bins edges into tile rows, then sort them, and a scan pass bins them into tiles and accumulates backdrops/winding numbers.
 - Pathfinder's binning algorithm writes mask edges into a vertex buffer and allocate mask tiles directly and then scans tiles to propagate backdrops

Beyond the binning startegy, the main difference is that pathfinder does not attempt to perform occlusion culling. That allows the tiler to run in parallel. As a result pathfinder has better multi-core CPU time and worse single-core CPU time (withing 20% either way), and generates more work for the GPU.

It's possible that pathinder's binning algorithm with occlusion culling added would get the best single core performance (need to try and see).

Pathfinder's binning code is not necessarily much simpler but definitely a lot less verbose.

### Occlusion culling

As mentioned earlier and on the pathfinder article, overdraw is a big limiting factor for raserization performance. In addition we observe that a lot of content is built by stacking many shapes on top of one anoter, which means that the total cost in number of pixels written to is usually much larger than the output area.
breaking paths into a regular grid gives us a very poserfull tool: fully covered opaque tiles (tiles inside a shape that don't contain any edge) can easily be used to occlude content under them.
The easiest and most efficient way to take advantage of this is for paths to be tiled in front-to-back order. for each cell in the grid, a flag is set once an opaque full tile is encountered. After that, any content on that tile is skipped since it is occluded.
This speeds up rasterization a lot on many test cases, it also reduces the amount of data to send to the GPU and speeds up the tiling process itself.

Rendering on GPU still happens in traditional back-to-front order.

### clipping

TODO: clipping isn't implemented yet.

The plan is to tile clip paths before the clipped content so that fully clipped out tiles can be marked and used for occlusion culling the same way opaque tile are.

### Patterns

TODO: only solid colors are implemented.

The general idea is for patterns to be pre-rasterized just like the masks and to use the occlusion culling information to reduce the amount work here as well.

## Miscellaneous findings

### Tile sizes

Smaller tile sizes means more work on the CPU, but also more efficient occlusion culling and less mask pixels (less of the expensive shader and less memory used).

There seem to be a sweet spot around 16x16 tiles.

### Float precision

In general the tiling algorithm isn't sensitive to arithmetic precision. There are two exceptions:
 - Edges are assigned to tile rows and split to remove the parts below and above the tile. To compute tile backdrop winding numbers, we look at whether edges cross the tile's upper side. The combination of these two things introduces the need to be very careful about splitting the edges. Since it isn't a perfectly precise operation, we can end up splitting just below the tile's upper side where in theory the split point shoulf have been exactly on the upper side, and as a result fail to count the edge when computing the winding number later during the row processing pass. This is addressed by the code that splits the edge, by doing a bit of snapping and ensuring that an edge
 that crossed the tile's upper side is always split into something with an endpoint on it.
 - The line rasterizaton routine running on the GPU can divide by zero if the edge has a very specific slope. It's extremely unlikely but it could be fixed with an extra branch or by devising another efficient line rasterization function. Large tile sizes make it more likely to run into this.


### Parallelism

The tiling code runs in two phases:
 - the row assignment phase where edges are assigned to tile rows
 - the row processing phase where each row is processed, comuping winding numbers and building per-tile edge lists.

For a give path, the row assignment phase is very sequential by nature however
all rows can be processed concurrently during the row processing phase.

There is a dependency between the row processing phase of consecutive paths because each path benefits from the occlusion culling information built by previous paths.

I tried to use tow implementations of parallel for-each in the row processing phase only. For each path, only that phase is done in parallel and the main thread does not continue until the processing phase is complete
 - with rayon: The performance was catastrophic (5 times slower)
 - with parasol: The performance was close to the single-threaded performance

Why is that? In general it is because there is a non-trivial warmup cost for worker threads. These two job schedulers are much more efficient when the workers are already running code and can pick up the new work without going to sleep. Unfortunately this simple experiment had to fork and join once per path, effectively letting the workers go to sleep between each burst of parallel work.
Rayon's case was made worse by the fact that the main thread cannot help with the parallel workload, so no work is processed until a worker picks a task up, and at the end the main thread has to be woken up again which adds more latency. On the other hand, parasol is designed to avoid exactly that so it fared much better even though the results were still underwhelming.

Conclusion: in order to make the tiling efficiently run on multiple threads, we have to ensure that we can keep the workers fed with work throughout the entire workload (or at least large portions of it).

- One way to help with that could be to do more pipelining: The row assignment for N consecutive paths can happen in parallel. It could help with keeping at least one extra thread alive, although the row processing phase would still be alternating between wide and narrow due to dependency to previous paths.
- To improve upon the wide/narrow situation with the row processing pass, we relax the dependency to previous paths by splitting it into rows. rows only need to depend on the work produced on the same row by previous paths.

The two ideas above are not very easy to express with rayon/parasol today, however figuring out a safe way to express that with parasol would be useful for SWGL which has a similar per-band-of-pixels type of dependency on its own parallel scheduling.

### Other

- We observe that Firefox rendering performance is more often CPU than GPU-bound. So speeding up the tiling routine is important to take as little of the limited frame budget.
- Currently the tiling algorithm takes about 3 to 4 milliseconds to preprocess the filled paths of the ghostscript tiger at a resolution of 1800x1800 on a decent but not high end computer in high power state and about twice as much on low power state. This testcase is heavier than what is typically seen on the web (226 paths covering a large amount of pixels) although it also benefits more from the occlusion culling optimization than typical web content.
    - This fits reasonably well in a typical frame budget. This opens the opportunity for tiling paths during frame building (especially clip paths) and re-render them on the fly during zoom.
- Much of the tiling time is spent approximating curves with sequences of flattened paths. It is a pretty expensive operation in general. all curve are first approximated with sequences of quadratic bézier segments, a lot of time is spent flattening these quadratic bézier segments. The method usef for flattening is the one described at https://raphlinus.github.io/graphics/curves/2019/12/23/flatten-quadbez.html. I've tried a number of things with varying degress of success:
    - use a (simpler) recursive flattening algorithm. while a lot less ALU intensive it generates a lot more edges for the same approximation criteria which slows the rest of the process more than it speeds up flattening.
    - use the flattening algorithm described in the paper "Fast, Precise Flattening of Cubic Bézier Segment Offset Curves" (we use it in Firefox to render dashed strokes). It was in fact what I started with but Raph Levien's algorithm produced better output and as a result performed better.
    - use SIMD instructions to speed up the per-line segment operations. It was marginally faster, not enough to justify the complexity and platform-specific code. Probably worth revisiting.
    - detect when quads can be approximated by one or two line segments, which happens for significant amount of edges. That one paid off and was kept.
    - send quadratic bézier curves to the GPU and do the flattening in a shader
        - That turned out slower. I suspect that this because the fragment shader is not able to take advantage of uniform control flow within a tile. There is evidence show by the piet_gpu project, that this can be done very efficiently by a compute shader.
    - send quadratic bézier curves to the GPU and render them some other way
        - There is a project called vger which does that, via a quick in-out test and the quad signed distance approximation from the RAVG paper to antialias pixels close to the edge. I am worried that it might perform less well on low end hardware but well worth experimenting with.
- Pathologic cases can lead to rasterizing a large amount of masks which requires large texture allocations if all masks are rasterized before they the final render pass.
    - To prevent that the prototypr supports setting a maximum mask atlas allocation and alternating between rasterizing into the mask texture and rendering the main target.
- Other pathologic cases could lead to having a very large amount of paths in a single tile, at which point the edges represent a lot more data than the pixels of the tile.
    - To prevent this family of issues the tiler can fall back to rasterizing the mask on the CPU and upload it.

# GPU performance

Rasterizing the masks is very much ALU bound, while compositing is spends a faire amount of time waiting for texture fetches. This suggest that we would get speedups from rasterizing masks directly in the main alpha passes since the could overlap with fetching the patterns pre-rendered texture data, or at least remove the time spent fetching the mask.
The downside is adding more shader combinations and potentially breaking batches, especially if we want to support multiple mask types within a single path (multiple specialized mask shaders withed number of edges).

Right now always rasterizing masks in an atlas has the benefit of having any number of ways to generate the masks with little impact to the number of draw calls.

A solution could be to have much more complex alpha tile shaders that can rasterize the mask in the must common ways and default to reading the mask atlas. This should perform well on modern GPUs but it's not clear that older ones would handle that well.
