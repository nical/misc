#import gpu_buffer

struct RenderTask {
    /// Bounds in pixels of the render task.
    bounds: vec4f,
    /// Offset in pixels to apply after transformations and clipping.
    offset: vec2f,
    /// The size of the destination target.
    target_rcp_size: vec2f,
};

fn render_task_fetch(address: u32) -> RenderTask {
    let data = f32_gpu_buffer_fetch_2(address);
    return RenderTask(
        data.data0,
        data.data1.xy,
        data.data1.zw,
    );
}

fn render_task_target_position(task: RenderTask, pos: vec2<f32>) -> vec4<f32> {
    var p = (pos - task.offset) * task.target_rcp_size * 2.0 - vec2<f32>(1.0);
    p.y = -p.y;

    return vec4<f32>(p, 0.0, 1.0);
}
