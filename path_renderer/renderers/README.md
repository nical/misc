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

## Multi-threading

The core crate provides affordances for running the prepare pass in parallel:

### Worker Data

See [`worker.rs`](../core/src/worker.rs).

```rust
let num_workers = ctx.workers.num_workers();
let mut worker_data = Vec::with_capacity(num_workers);
for _ in 0..num_workers {
    worker_data.push(MyWorkerData {
        // put some per-worker state here
    });
}

let workers = ctx.workers.with_data(&mut worker_data);
workers.slice_for_each(&items[..], &mut |workers, sub_slice, _| {
    // This runs in parallel.
    // core_data contains common per-worker state that is useful to most
    // renderers such as buffers to store vertices, indices, etc.
    // my_data contains one of the `MyWorkerData` we pushed above, each
    // worker thread gets exclusive access to one.
    let (core_data, my_data) = worker.data();
    for item in sub_slice {
        // do some work.
    }
});

// ...

// back to the single-thread path, we can extract the contents of
// worker_data if needed.

for data in worker_data {
    // ...
}
```

### GPU Data

The data to send to the gpu each frame is typically generated in the prepare phase. Doing it in parallel can be challenging because parallelism prevents us from guaranteeing the order in which items are processed, while the ordering and location of the data in the GPU buffers and textures matters.

`GpuStreams` and `GpuStore` provides efficient ways to make this work in parallel (as well as on a single thread) without extra copies. Internally they leverage the fact that data is written into (out of order) into staging buffers, but only the order and offset in the desitnation (host-side) buffers matters, so gpu copies from staging buffers to destination buffers are produced such that final data is arranged in the desired way regardless of the order in which it was produced.

```rust

idx_stream = workers.data().0.indices.next_stream_id();

workers.slice_for_each(&items[..], &mut |workers, sub_slice, _| {
    let (common, my_data) = workers.data();
    let indices = &common.indices;
    let vertices = &common.vertices;

    // idx_writer is a `GpuStreamWriter`. We don't know as we are pushing
    // into it where the contents will land, but we have the guarantee that
    // all data pushed into writers associated with the same id (idx_stream)
    // will be contiguous in the destination buffer and ordered according to
    // the provided sort key (hence the use of the z-index as the sort sort
    // key here).
    let sort_key = items[0].z_index;
    let mut idx_writer = indices.write(idx_stream, sort_key)

    // vtx_writer is a `GpuStoreWriter`. Consecutive pushes aren't guaranteed
    // to be contiguous in the destination, however we know their destination
    // offset as soon as the push happens.
    let mut vtx_writer = vertices.write_items::<MyVertex>(),

    // Send a triangle to the GPU.
    let a = vtx_writer.push(MyVertex { /* ... */ });
    let b = vtx_writer.push(MyVertex { /* ... */ });
    let c = vtx_writer.push(MyVertex { /* ... */ });
    idx_writer.push(a.to_u32());
    idx_writer.push(b.to_u32());
    idx_writer.push(c.to_u32());
}
```
