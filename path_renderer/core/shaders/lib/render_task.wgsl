#import gpu_buffer

struct RenderTask {
    /// Acts as a clip in the surface space of the render task.
    clip: vec4f,
    /// Offset in pixels to apply after transformations and clipping.
    content_offset: vec2f,
    /// The size of the destination target.
    rcp_target_size: vec2f,

    // Not part of this struct: image_source: vec4f
};

fn render_task_fetch(address: u32) -> RenderTask {
    let data = f32_gpu_buffer_fetch_2(address);
    return RenderTask(
        data.data0,
        data.data1.xy,
        data.data1.zw,
    );
}

fn render_task_fetch_image_source(address: u32) -> vec4<f32> {
    return f32_gpu_buffer_fetch_1(address + 2);
}


fn render_task_target_position(task: RenderTask, pos: vec2<f32>) -> vec4<f32> {
    var p = (pos - task.content_offset) * task.rcp_target_size * 2.0 - vec2<f32>(1.0);
    p.y = -p.y;

    return vec4<f32>(p, 0.0, 1.0);
}
