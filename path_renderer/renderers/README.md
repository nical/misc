# Renderers

Renderers are defined in a somewhat unconventional way: Instead of exposing drawing functionality at the surface or render-pass level for example:

```rust
// We are *not* doing something like this:
ctx.fill_path(path, pattern)
ctx.fill_rectangle(rect, pattern)
```

Renderers externally expose APIs to paint into a context:

```rust
// Instead we do this:
tile_renderer.fill_path(&mut ctx, pattern);
rect_renderer.fill(&mut ctx, pattern);
```

The code is organized this way to make adding rendering algorthms easy. This means that there isn't a particular user facing API that renderers need to adhere to. Internally, renderers need to integrate with the batching system.
This is done by
implementing the `Renderer` trait which exposes a few important entry points:

## Batching

The batching phase is when a renderer is used to push commands into the context. Renderers hold one or more `BatchList` and push commands into it.

See `BatchList` and how it interacts with `Batcher` in [batch.rs](../core/src/batch.rs).

A rendering entry point typically looks like this:

```rust
// The API can be anythig as long as it takes a `RenderPassContext`.
pub fn fill_shape(
    &mut self,
    ctx: &mut RenderPassContext,
    transforms: &Transforms,
    shape: Shape,
    mut pattern: BuiltPattern,
) {
    let transform = transforms.current_id();
    let z_index = ctx.z_indices.push();

    // batches: BatchList<MyCustomItem, MyCustomPerBatchData>
    self.batches.add(
        ctx,
        &pattern.batch_key(),
        &shape.aabb(),
        BatchFlags::empty(),
        // If a batch needs to be created, call this:
        &mut || MyCustomPerBatchData {
                pattern,
                opaque_draw: None,
                masked_draw: None,
                blend_mode: pattern.blend_mode,
        },
        // Push on the appropriate batch:
        &mut |mut batch| {
            batch.push(MyCustomItem {
                shape: shape.clone(),
                pattern,
                transform,
                z_index,
            });
        }
    )
}
```

Once all commands are recorded, frame building starts. Internally frame building code will call the `prepare` and `render` functions.

## Prepare

The `prepare` method is called after all drawing commands have been recorded in the batcher. This is when the renderer is intended to do the bulk of the processing work (if any) to generate data to send to the GPU.

```rust
impl Renderer for MyRenderer {
    fn prepare(&mut self, ctx: &mut PrepareContext) {
        for batch_id in ctx.pass
            .batches()
            .iter()
            .filter(|batch| batch.renderer == self.renderer_id)
        {
            let (commands, surface, info) = self.batches.get_mut(batch_id.index);
            for cmd in &commands {
                // ...
            }
        }
    }
    // ...
}
```

## Render

The `render` method is where renderers push drawing commands into a `wgpu` render pass.

```rust
impl Renderer for MyRenderer {
    // ...

    fn render<'pass, 'resources: 'pass, 'tmp>(
        &self,
        batches: &[BatchId],
        surface_info: &SurfacePassConfig,
        ctx: core::RenderContext<'resources, 'tmp>,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        for batch_id in batches {
            let (_, _, batch) = self.batches.get(batch_id.index);
            // ...
            render_pass.set_pipeline(self.pipeline);
            render_pass.draw_indexed(indices, 0, 0..1);
            // etc.
        }
    }

    // ...
}
```
