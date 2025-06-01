use super::{BindGroupLayout, GeometryDescriptor, PatternDescriptor, WgslType};
use std::fmt::Write;

pub fn generate_shader_source(
    geometry: &GeometryDescriptor,
    pattern: Option<&PatternDescriptor>,
    base_bindings: &BindGroupLayout,
    geom_bindings: Option<&BindGroupLayout>,
    pattern_bindings: Option<&BindGroupLayout>,
) -> String {
    let mut source = String::new();

    // TODO: the base bindings expect this struct to be present, but it is being
    // replaced by the RenderTask infrastructure.
    writeln!(source, "struct RenderTarget {{").unwrap();
    writeln!(source, "   resolution: vec2<f32>,").unwrap();
    writeln!(source, "   inv_resolution: vec2<f32>,").unwrap();
    writeln!(source, "}};").unwrap();

    let mut group_index = 0;
    generate_bind_group_shader_source(&base_bindings, group_index, &mut source);

    group_index += 1;

    if let Some(bindings) = geom_bindings {
        generate_bind_group_shader_source(&bindings, group_index, &mut source);
        group_index += 1;
    }
    if let Some(bindings) = pattern_bindings {
        generate_bind_group_shader_source(&bindings, group_index, &mut source);
    }

    writeln!(source, "struct GeometryVertex {{").unwrap();
    writeln!(source, "    position: vec4<f32>,").unwrap();
    writeln!(source, "    pattern_position: vec2<f32>,").unwrap();
    writeln!(source, "    pattern_data: u32,").unwrap();
    for varying in &geometry.varyings {
        //println!("=====    {}: {},", varying.name, varying.kind.as_str());
        writeln!(source, "    {}: {},", varying.name, varying.kind.as_str()).unwrap();
    }
    writeln!(source, "}}").unwrap();

    writeln!(source, "#import {}", geometry.name).unwrap();
    writeln!(source, "").unwrap();

    if let Some(pattern) = pattern {
        writeln!(source, "struct Pattern {{").unwrap();
        for varying in &pattern.varyings {
            writeln!(source, "    {}: {},", varying.name, varying.kind.as_str()).unwrap();
        }
        writeln!(source, "}}").unwrap();
        writeln!(source, "#import {}", pattern.name).unwrap();
    }

    writeln!(source, "").unwrap();

    // vertex

    writeln!(source, "struct VertexOutput {{").unwrap();
    writeln!(source, "    @builtin(position) position: vec4<f32>,").unwrap();
    let mut idx = 0;
    for varying in &geometry.varyings {
        let interpolate = if varying.interpolated {
            "perspective"
        } else {
            "flat"
        };
        writeln!(
            source,
            "    @location({idx}) @interpolate({interpolate}) geom_{}: {},",
            varying.name,
            varying.kind.as_str()
        )
        .unwrap();
        idx += 1;
    }
    if let Some(pattern) = pattern {
        for varying in &pattern.varyings {
            let interpolate = if varying.interpolated {
                "perspective"
            } else {
                "flat"
            };
            writeln!(
                source,
                "    @location({idx}) @interpolate({interpolate}) pat_{}: {},",
                varying.name,
                varying.kind.as_str()
            )
            .unwrap();
            idx += 1;
        }
    }
    writeln!(source, "}}").unwrap();
    writeln!(source, "").unwrap();

    writeln!(source, "@vertex fn vs_main(").unwrap();
    writeln!(source, "    @builtin(vertex_index) vertex_index: u32,").unwrap();
    let mut attr_location = 0;
    for attrib in &geometry.vertex_attributes {
        writeln!(
            source,
            "    @location({attr_location}) vtx_{}: {},",
            attrib.name,
            attrib.kind.as_str()
        )
        .unwrap();
        attr_location += 1;
    }
    for attrib in &geometry.instance_attributes {
        writeln!(
            source,
            "    @location({attr_location}) inst_{}: {},",
            attrib.name,
            attrib.kind.as_str()
        )
        .unwrap();
        attr_location += 1;
    }
    writeln!(source, ") -> VertexOutput {{").unwrap();

    writeln!(source, "    var vertex = geometry_vertex(").unwrap();
    writeln!(source, "        vertex_index,").unwrap();
    for attrib in &geometry.vertex_attributes {
        writeln!(source, "        vtx_{},", attrib.name).unwrap();
    }
    for attrib in &geometry.instance_attributes {
        writeln!(source, "        inst_{},", attrib.name).unwrap();
    }
    writeln!(source, "    );").unwrap();

    if pattern.is_some() {
        writeln!(
            source,
            "    var pattern = pattern_vertex(vertex.pattern_position, vertex.pattern_data);"
        )
        .unwrap();
    }

    writeln!(source, "    return VertexOutput(").unwrap();
    writeln!(source, "        vertex.position,").unwrap();
    for varying in &geometry.varyings {
        writeln!(source, "        vertex.{},", varying.name).unwrap();
    }
    if let Some(pattern) = pattern {
        for varying in &pattern.varyings {
            writeln!(source, "        pattern.{},", varying.name).unwrap();
        }
    }
    writeln!(source, "    );").unwrap();
    writeln!(source, "}}").unwrap();
    writeln!(source, "").unwrap();

    // fragment

    writeln!(source, "@fragment fn fs_main(").unwrap();
    let mut idx = 0;
    for varying in &geometry.varyings {
        let interpolate = if varying.interpolated {
            "perspective"
        } else {
            "flat"
        };
        writeln!(
            source,
            "    @location({idx}) @interpolate({interpolate}) geom_{}: {},",
            varying.name,
            varying.kind.as_str()
        )
        .unwrap();
        idx += 1;
    }
    if let Some(pattern) = pattern {
        for varying in &pattern.varyings {
            let interpolate = if varying.interpolated {
                "perspective"
            } else {
                "flat"
            };
            writeln!(
                source,
                "    @location({idx}) @interpolate({interpolate}) pat_{}: {},",
                varying.name,
                varying.kind.as_str()
            )
            .unwrap();
            idx += 1;
        }
    }
    writeln!(source, ") -> @location(0) vec4<f32> {{").unwrap();
    writeln!(source, "    var color = vec4<f32>(1.0);").unwrap();

    writeln!(source, "    color.a *= geometry_fragment(").unwrap();
    for varying in &geometry.varyings {
        writeln!(source, "        geom_{},", varying.name).unwrap();
    }
    writeln!(source, "    );").unwrap();

    if let Some(pattern) = pattern {
        writeln!(source, "    color *= pattern_fragment(Pattern(").unwrap();
        for varying in &pattern.varyings {
            writeln!(source, "        pat_{},", varying.name).unwrap();
        }
        writeln!(source, "    ));").unwrap();
    }

    writeln!(source, "    // Premultiply").unwrap();
    writeln!(source, "    color.r *= color.a;").unwrap();
    writeln!(source, "    color.g *= color.a;").unwrap();
    writeln!(source, "    color.b *= color.a;").unwrap();

    writeln!(source, "    return color;").unwrap();

    writeln!(source, "}}").unwrap();

    source
}

fn generate_bind_group_shader_source(bgl: &BindGroupLayout, index: u32, source: &mut String) {
    let mut binding = 0;
    for entry in &bgl.entries {
        write!(source, "@group({index}) @binding({binding}) ").unwrap();
        let name = &entry.name;
        let struct_ty = &entry.struct_type;
        match entry.ty {
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                ..
            } => {
                writeln!(source, "var<uniform> {name}: {struct_ty};").unwrap();
            }
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { .. },
                ..
            } => {
                writeln!(source, "var<storage> {name}: {struct_ty};").unwrap();
            }
            wgpu::BindingType::Sampler(..) => {
                writeln!(source, "var {name}: sampler;").unwrap();
            }
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                ..
            } => {
                writeln!(source, "var {name}: texture_depth_2d;").unwrap();
            }
            wgpu::BindingType::Texture {
                sample_type,
                view_dimension,
                ..
            } => {
                let dim = match view_dimension {
                    wgpu::TextureViewDimension::D1 => "1",
                    wgpu::TextureViewDimension::D2 => "2",
                    wgpu::TextureViewDimension::D3 => "3",
                    _ => unimplemented!(),
                };
                let sample_type = match sample_type {
                    wgpu::TextureSampleType::Float { .. } => "f32",
                    wgpu::TextureSampleType::Sint { .. } => "i32",
                    wgpu::TextureSampleType::Uint { .. } => "u32",
                    _ => "error",
                };
                writeln!(source, "var {name}: texture_{dim}d<{sample_type}>;").unwrap();
            }
            wgpu::BindingType::StorageTexture { .. } => {
                todo!();
            }
            wgpu::BindingType::AccelerationStructure { .. } => {
                unimplemented!();
            }
        }

        binding += 1;
    }
}

impl WgslType {
    fn as_str(self) -> &'static str {
        match self {
            WgslType::Float32 => "f32",
            WgslType::Uint32 => "u32",
            WgslType::Sint32 => "i32",
            WgslType::Bool => "bool",
            WgslType::Float32x2 => "vec2<f32>",
            WgslType::Float32x3 => "vec3<f32>",
            WgslType::Float32x4 => "vec4<f32>",
            WgslType::Uint32x2 => "vec2<u32>",
            WgslType::Uint32x3 => "vec3<u32>",
            WgslType::Uint32x4 => "vec4<u32>",
            WgslType::Sint32x2 => "vec2<i32>",
            WgslType::Sint32x3 => "vec3<i32>",
            WgslType::Sint32x4 => "vec4<i32>",
        }
    }
}
