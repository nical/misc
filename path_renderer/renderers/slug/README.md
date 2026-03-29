# Slug path renderer

A path renderer based on the Slug algorithm by Eric Lengyel.

https://jcgt.org/published/0006/02/02/

This a work-in-progress, not ready for use.

# Notes

This renderer uses a pretty basic acceleration structure that bins curves into both horizontal and vertical bands. This acceleration structure uses at most 64 bands which would work very well for small shapes (for example typicaly text sizes) but does not scale very well for very complex shapes covering a large number of pixels.
This is a pretty unfair disadvantage compared to some of the other renderers in this repository given that what's primarily being tested here are complex paths at a high resolution (each fragment ends up processing orders of magnitude more edges than, say, the tiled renderer would).

Slug's general approach (the interesting part being the math that is happening in the shader) is not incompatible with using a more appropriate binning structure for complex paths at high resolutions.
