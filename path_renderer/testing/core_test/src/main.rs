use core::wgpu;
use core::units::SurfaceIntSize;
use testing_utils::ReftestImage;
use testing_utils::SinglePassReftest;
use testing_utils::TestHarness;
use testing_utils::core::{self};

fn main() {
    let mut harness = TestHarness::new(wgpu::Backends::all());

    SinglePassReftest {
        name: "core test",
        a: ReftestImage::Render(Box::new(|ctx| {
            println!("Run A");
        })),
        b: ReftestImage::Render(Box::new(|ctx| {
            println!("Run B");
        })),
        size: SurfaceIntSize::new(512, 512),
        requirements: vec![],
    }.run(&mut harness).unwrap();
}
