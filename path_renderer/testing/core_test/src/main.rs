use core::wgpu;
use core::units::SurfaceIntSize;
use pattern_color::SolidColorRenderer;
use rectangles::Rectangles;
use testing_utils::ReftestImage;
use testing_utils::SinglePassReftest;
use testing_utils::TestData;
use testing_utils::TestHarness;
use testing_utils::core::Color;
use testing_utils::core::units::*;
use testing_utils::core::{self};

fn main() {
    let mut harness = TestHarness::new(wgpu::Backends::all());
    let rectangles = Rectangles::new(&harness.instance.device, &mut harness.instance.shaders).new_renderer(0);
    let colors = SolidColorRenderer::register(&mut harness.instance.shaders);

    SinglePassReftest {
        name: "core test",
        data: TestData {
            renderers: (rectangles,),
            other: colors,
        },
        size: SurfaceIntSize::new(512, 512),
        a: ReftestImage::Render(Box::new(|mut ctx, data| {
            let (rectangles,) = &mut data.renderers;
            let blue = data.other.add(Color { r: 0, g: 0, b: 255, a: 255 });
            rectangles.fill_rect(
                &mut ctx,
                &SurfaceRect {
                    min: SurfacePoint::new(0.0, 0.0),
                    max: SurfacePoint::new(100.0, 200.0),
                },
                rectangles::Aa::empty(),
                blue,
            );
        })),
        b: ReftestImage::Render(Box::new(|mut ctx, data| {
            let (rectangles,) = &mut data.renderers;
            let blue = data.other.add(Color { r: 0, g: 0, b: 255, a: 255 });
            rectangles.fill_rect(
                &mut ctx,
                &SurfaceRect {
                    min: SurfacePoint::new(0.0, 0.0),
                    max: SurfacePoint::new(100.0, 200.0),
                },
                rectangles::Aa::empty(),
                blue,
            );
        })),
        requirements: vec![],
    }.run(&mut harness).unwrap();
}
