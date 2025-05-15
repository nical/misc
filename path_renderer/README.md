# Path rendering experiments

Work-in-progress GPU vector graphics rendering experiments

# Demo

```bash
# Render the ghostscript tiger
cargo run --release ./demo/tiger.svg

# Run the demo with the paris-30k example and parallel versions
# of the algorithms when available.
cargo run --release ./demo/paris-30k.svg --parallel
```

While the demo is running, press:
- 'o' to toggle the debug overlay,
- 'f' to change the fill rendering method,
- 's' to change the stroke rendering method,
- the arrow keys or touch-pad scrolling to pan around,
- `page-up`/`page-down` to zoom in/out.
