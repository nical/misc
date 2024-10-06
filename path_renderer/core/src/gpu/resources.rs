// TODO: name clash with renderer resources.

use crate::units::SurfaceIntSize;
use crate::context::SurfaceKind;
use crate::render_graph::ResourceKind;

pub enum ResourceDescriptor {
    Texture { size: SurfaceIntSize, format: SurfaceKind }
}

pub struct Resources {
    resource_kinds: Vec<ResourceDescriptor>,
}

impl Resources {
    pub fn register_resource_kind(&mut self, descriptor: ResourceDescriptor) -> ResourceKind {
        let id = ResourceKind(self.resource_kinds.len() as u8);
        self.resource_kinds.push(descriptor);

        id
    }


}
