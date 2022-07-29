use crate::system::*;
use crate::types::{Primitive, BatchId};

pub struct RendererOptions {

}

pub struct Scene {
    primitives: Vec<Primitive>,
}

impl Scene {
    pub fn new() -> Self {
        Scene {
            primitives: Vec::new(),
        }
    }
}

pub struct Frame {
    render_passes: Vec<RenderPass>,
}

impl Frame {
    pub fn new() -> Self {
        Frame { render_passes: Vec::new() }
    }

    pub fn clear(&mut self) {
        self.render_passes.clear();
    }
}

pub struct RenderPass {
    batches: Vec<BatchId>,
}

pub struct Renderer {
    scene: Scene,
    frame: Frame,
    systems: Vec<Box<dyn RenderingSystem>>,
}

impl Renderer {
    pub fn new(_options: &RendererOptions, systems: Vec<Box<dyn RenderingSystem>>) -> Self {
        Renderer {
            systems,
            scene: Scene::new(),
            frame: Frame::new(),
        }
    }

    pub fn systems(&self) -> &[Box<dyn RenderingSystem>] {
        &self.systems
    }

    pub fn render(&mut self, par: &mut parasol::Context, device: &mut wgpu::Device) {
        self.frame.clear();

        for sys in &mut self.systems {
            sys.begin_frame(par, device);
        }

        self.compute_visibility(par);
        self.prepare(par);
        self.process(par);
        self.resolve(par);
        self.batch(par);
        self.build_drawing_commands(par, device);

        for sys in &mut self.systems {
            sys.end_frame(par, device);
        }
    }

    fn compute_visibility(&mut self, par: &mut parasol::Context) {

    }

    fn prepare(&mut self, par: &mut parasol::Context) {
        
    }

    fn process(&mut self, par: &mut parasol::Context) {
        par.for_each(&mut self.systems[..]).run(|par, mut system| {
            let mut ctx = ProcessContext {
                par,
            };
            system.process(&mut ctx);
        });
    }

    fn resolve(&mut self, par: &mut parasol::Context) {

    }

    fn batch(&mut self, par: &mut parasol::Context) {

    }

    fn build_drawing_commands(&mut self, par: &mut parasol::Context, device: &wgpu::Device) {
        par.for_each(&mut self.frame.render_passes[..]).run(|par, pass| {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: None,
            });

            let ctx = &mut DrawCallsContext {
                par,
                device,
                encoder: &mut encoder
            };

            for batch in &pass.batches {
                self.systems[batch.system as usize].generate_draw_call(batch.index, ctx);
            }
            
            
        });
    }
}

/*
fn generate_batches(
    batcher: &mut dyn crate::batching::Batcher,
    systems: &mut [Box<dyn RenderingSystem>],
    primitives: &[Primitive],
    target_rects: &[Rect],
) {
    for (primitive, rect) in primitives.iter().zip(target_rects.iter()) {
        systems[primitive.system as usize].batch_primitive(batcher, primitive.index, rect);
    }
}

fn generate_render_pass_draw_calls(
    systems: &mut [Box<dyn RenderingSystem>],
    ctx: &mut DrawCallsContext,
    batches: &[BatchId],
) {
}
*/