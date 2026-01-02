use core::RenderPassConfig;
use core::render_pass::RenderPassBuilder;
use core::render_task::RenderTask;
use core::units::SurfaceIntSize;
use core::wgpu;
use core::transfer::Transfer;
use core::units::SurfaceIntPoint;
use core::render_pass::ColorAttachment;
use core::resources::ResourceKey;
use core::resources::TextureKind;
use testing_utils::core::units::SurfaceIntVector;
use testing_utils::core::{self, BindingsId};

fn main() {
    println!("core tests");

    // Init

    let mut instance = testing_utils::init(wgpu::Backends::all());
    let binding_namespace = instance.create_bindings_namespace();

    // Frame

    let mut frame = instance.begin_frame();
    let mut f32_buffer = frame.f32_buffer.write();

    // Build render passes

    let mut pass_builder = RenderPassBuilder::new();

    let size = SurfaceIntSize::new(512, 512);

    let key = ResourceKey::texture(
        TextureKind::color()
            .with_attachment()
            .with_copy_src(),
        size,
    );
    let target_idx = frame.resources.allocate(key);
    let binding = BindingsId::new(binding_namespace, target_idx.index);

    let task = RenderTask::new(&mut f32_buffer, size, SurfaceIntVector::zero());

    pass_builder.begin(&task, RenderPassConfig::default());

    let mut pass = pass_builder.end();

    pass.set_color_attachments(&[ColorAttachment {
        non_msaa: Some(binding),
        msaa: None,
        load: false,
        store: true,
        clear: true,
    }]);

    // Schedule passes

    frame.passes.push_render_pass(pass);
    frame.passes.push_transfer(Transfer::ReadbackTexture {
        src: binding,
        size,
        src_offset: SurfaceIntPoint::zero(),
        callback: Some(Box::new(|result| {
            let image = result.unwrap();
            for px in image.data.chunks(4) {
                assert_eq!(px[0], 0);
                assert_eq!(px[1], 0);
                assert_eq!(px[2], 0);
                assert_eq!(px[3], 0);
            }
            println!("ok.");
        })),
    });

    // Render

    let mut encoder = instance.create_encoder();

    std::mem::drop(f32_buffer);
    instance.render_frame(frame, &[], &[], &mut encoder, &mut []);

    instance.queue.submit(Some(encoder.finish()));
    instance.end_frame();
}
