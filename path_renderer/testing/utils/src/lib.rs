use futures::executor::block_on;

pub use core;
use core::render_pass::{ColorAttachment, RenderPassBuilder, RenderPassContext};
use core::render_task::RenderTask;
use core::resources::{ResourceKey, TextureKind};
use core::transfer::Transfer;
use core::units::{SurfaceIntSize, SurfaceIntVector};
use core::{BindingsId, RenderPassConfig, wgpu};

pub mod args;

use core::{Instance};
use std::fs::File;
use std::io::BufReader;
use std::sync::mpsc::{TryRecvError, channel};
use std::time::Duration;

#[derive(Copy, Clone, Debug)]
pub struct Reftest {
    pub max_difference: u32,
    pub max_differing_pixels: u32,
}

pub struct Image {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
    // pub format: Format,
}

pub struct ImageComparison {
    // Number of pixels in the with a a difference greater or equal to
    // the index.
    pub differences: [u32; 255],
}

pub fn compare_images(
    img1: &Image,
    img2: &Image,
    check_alpha: bool,
) -> ImageComparison {
    let mut result = ImageComparison {
        differences: [0; 255],
    };

    let iter1 = img1.data.chunks(4);
    let iter2 = img2.data.chunks(4);
    for (p1, p2) in iter1.zip(iter2) {
        let mut diff = 0;
        diff = diff.max((p1[0] as i16 - p2[0] as i16).abs());
        diff = diff.max((p1[1] as i16 - p2[1] as i16).abs());
        diff = diff.max((p1[2] as i16 - p2[2] as i16).abs());
        if check_alpha {
            diff = diff.max((p1[3] as i16 - p2[3] as i16).abs());
        }
        diff = diff.min(255);

        result.differences[diff as usize] += 1;
    }

    let mut sum = 0;
    for diff in result.differences.iter_mut().rev() {
        *diff += sum;
        sum = *diff;
    }

    result
}

pub fn check_comparison(
    comparison: &ImageComparison,
    requirements: &[Reftest],
    extra_fuzz: u32,
) -> Result<(), Reftest> {
    if requirements.is_empty() {
        return check_comparison(
            comparison,
            &[Reftest {
                max_difference: 1 + extra_fuzz,
                max_differing_pixels: 0,
            }],
            0,
        )
    }

    for req in requirements {
        let idx = (req.max_difference + extra_fuzz).min(255).max(1) as usize;
        if comparison.differences[idx] > req.max_differing_pixels {
            return Err(*req);
        }
    }

    Ok(())
}

pub fn init(backends: wgpu::Backends) -> core::Instance {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends,
        ..wgpu::InstanceDescriptor::default()
    });

    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    })).unwrap();


    let (device, queue) = block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            required_features: wgpu::Features::default(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        },
    )).unwrap();

    Instance::new(&device, &queue, 0, false)
}

pub type RenderPassTestCallback = Box<dyn FnOnce(RenderPassContext)>;

pub enum ReftestImage {
    FromFile(String),
    Render(RenderPassTestCallback),
}

pub struct TestHarness {
    instance: Instance,
    extra_fuzz: u32,
}

impl TestHarness {
    pub fn new(backend: wgpu::Backends) -> Self {
        let instance = init(backend);

        TestHarness {
            instance,
            extra_fuzz: 0,
        }
    }
}

pub struct SinglePassReftest {
    pub name: &'static str,
    pub a: ReftestImage,
    pub b: ReftestImage,
    pub size: SurfaceIntSize,
    pub requirements: Vec<Reftest>,
}

impl SinglePassReftest {
    pub fn run(self, harness: &mut TestHarness) -> Result<(), Reftest> {

        let a = Self::run_one(self.a, self.size, &mut harness.instance);
        let b = Self::run_one(self.b, self.size, &mut harness.instance);

        let comparison = compare_images(&a, &b, true);

        check_comparison(
            &comparison,
            &self.requirements,
            harness.extra_fuzz,
        )
    }

    fn run_one(image: ReftestImage, size: SurfaceIntSize, instance: &mut Instance) -> Image {
        match image {
            ReftestImage::FromFile(name) => {
                let decoder = png::Decoder::new(BufReader::new(File::open(&name).unwrap()));
                let mut reader = decoder.read_info().unwrap();
                let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
                let info = reader.next_frame(&mut pixels).unwrap();
                Image {
                    width: info.width,
                    height: info.height,
                    data: pixels,
                }
            }
            ReftestImage::Render(callback) => {
                Self::run_callback(size, callback, instance)
            }
        }
    }

    fn run_callback(size: SurfaceIntSize, callback: RenderPassTestCallback, instance: &mut Instance) -> Image {
        let mut frame = instance.begin_frame();
        let mut f32_buffer = frame.f32_buffer.write();

        // Build render passes

        let mut pass_builder = RenderPassBuilder::new();

        let key = ResourceKey::texture(
            TextureKind::color()
                .with_attachment()
                .with_copy_src(),
            size,
        );
        let target_idx = frame.resources.allocate(key);
        let binding = BindingsId::temporary(target_idx.index);

        let task = RenderTask::new(&mut f32_buffer, size, SurfaceIntVector::zero());

        pass_builder.begin(&task, RenderPassConfig::default());

        let ctx = pass_builder.ctx();

        callback(ctx); // TODO

        let mut pass = pass_builder.end();

        pass.set_color_attachments(&[ColorAttachment {
            non_msaa: Some(binding),
            msaa: None,
            load: false,
            store: true,
            clear: true,
        }]);

        let (sender, receiver) = channel();

        frame.passes.push_render_pass(pass);
        frame.passes.push_transfer(Transfer::ReadbackTexture {
            src: binding,
            rect: None,
            callback: Some(Box::new(move|result| {
                let readback = result.unwrap();

                let mut data = Vec::with_capacity(readback.data.len());
                data.extend_from_slice(readback.data);
                let image = Image {
                    data,
                    width: readback.size.width as u32,
                    height: readback.size.height as u32,
                };

                sender.send(image).unwrap();
            })),
        });

        let mut encoder = instance.create_encoder();

        std::mem::drop(f32_buffer);
        instance.render_frame(frame, &[], &[], &mut encoder, &mut []);

        instance.queue.submit(Some(encoder.finish()));
        instance.end_frame();

        loop {
            match receiver.try_recv() {
                Ok(img) => {
                    return img;
                }
                Err(TryRecvError::Empty) => {
                    instance.device.poll(wgpu::PollType::Wait {
                        timeout: Some(Duration::from_millis(30)),
                        submission_index: None,
                    }).unwrap();
                }
                Err(e) => {
                    panic!("{e:?}");
                }
            }
        }
    }
}
