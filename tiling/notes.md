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
   - (B) the math related to dealing with edges, and anti-aliasing.
   - (C) transfering large amounts of data to the GPU
 - Large paths (which are typically expensive to rasterize on the CPU) tend to have a lot of fully covered tiles that contain no edges when broken up into small tiles. Taking advantage of this is key to reducing the cost of (B).
 - More generally breaking large paths into many tiles with fewer edges per tiles takes a very complex problem and breaks it into much simpler ones. It is also key to making good use of the GPU's parrallel programming model.
 - On GPU we have access to a lot more memory bandwidth than on CPU if used well (good for addressing (A)).
 - Regular grids have plenty of useful properties. One of them is that it can be used for occlusion culling and further reduce memory bandwidth (good for (A) again).
 - The sparse tile grid reprensentation is very good at reducing the amount of data to transfer to the GPU (C).

On the GPU side a paths are first rasterized in a mask atlas texture in a single draw call. Then the main render pass which first draws all opaque tiles in a draw call via a simple shader with blending disabled, and all masked tiles in a second draw call whic reads from the mask and applies a pattern.

### Departures from Pathfinder

While this work is inspried from pathfinder it has a few key differences:

- Pathfinder renders masks into a float texture  using a quad per edge with addirive blending to build winding numbers.
- This renders the mask in an alpha texture with a quad per tile, looping over edges in the fragment shader. This further reduces pressure on the blending hardware / fill rate which tends to be the bottleneck.

### Occlusion culling

As mentioned earlier and on the pathfinder article, overdraw is a big limiting factor for raserization performance. In addition we observe that a lot of content is built by stacking many shapes on top of one anoter, which means that the total cost in number of pixels written to is usually much larger than the output area.
breaking paths into a regular grid gives us a very poserfull tool: fully covered tiles (tiles inside a shape that don't contain any edge) can be used to occlude content under it in the same tile if it is opaque.
The easiest and most efficient way to take advantage of this is for paths to be tiled in front-to-back order. for each cell in the grid, a flag is set once an opaque full tile is encountered. After that, any content on that tile is skipped since it is occluded.
This speeds up rasterization a lot on many test cases, it also reduces the amount of data to send to the GPU and speeds up the tiling process itself.

Rendering on GPU still happens in traditional back-to-front order.

### Float precision

In general the tiling algorithm isn't sensitive to arithmetic precision. There are two exceptions:
 - Edges are assigned to tile rows and split to remove the parts below and above the tile. To compute tile backdrop winding numbers, we look at whether edges cross the tile's upper side. The combination of these two things introduces the need to be very careful about splitting the edges. Since it isn't a perfectly precise operation, we can end up splitting just below the tile's upper side where in theory the split point shoulf have been exactly on the upper side, and as a result fail to count the edge when computing the winding number later during the row processing pass. This is addressed by the code that splits the edge, by doing a bit of snapping and ensuring that an edge
 that crossed the tile's upper side is always split into something with an endpoint on it.
 - The line rasterizaton routine running on the GPU can divide by zero if the edge has a very specific slope. It's extremely unlikely but it could be fixed with an extra branch or by devising another efficient line rasterization function. Large tile sizes make it more likely to run into this.

### Tile sizes

Smaller tile sizes means more work on the CPU, but also more efficient occlusion culling and less mask pixels (less of the expensive shader and less memory used).

There seem to be a sweet spot around 16x16 tiles.

### clipping

TODO: clipping isn't implemented yet.

The plan is to tile clip paths before the clipped content so that fully clipped out tiles can be marked and used for occlusion culling the same way opaque tile are.

### Patterns

TODO: only solid colors are implemented.

The general idea is for patterns to be pre-rasterized just like the masks and to use the occlusion culling information to reduce the amount work here as well.

## Miscellaneous findings

- We observe that Firefox rendering performance is more often CPU than GPU-bound. So speeding up the tiling routine is important to take as little of the limited frame budget.
- Currently the tiling algorithm takes about 3 to 4 milliseconds to preprocess the filled paths of the ghostscript tiger at a resolution of 1800x1800 on a decent but not high end computer. This testcase is heavier than what is typically seen on the web (226 paths covering a large amount of pixels) although it also benefits more from the occlusion culling optimization than typical web content.
    - This fits quite well in a typical frame budget. This opens the opportunity for tiling paths during frame building (especially clip paths) and re-render them on the fly during zoom.
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

