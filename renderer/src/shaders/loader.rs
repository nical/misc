use super::ShaderDescriptor;
use super::shader_list::{primary_features, secondary_features};

use std::collections::HashMap;
use std::io::Read;

struct Source {
    src: String,
    already_imported: bool,
}

#[derive(Debug)]
struct PatternNames {
    matcher: String,
    library: String, 
    descriptor: String,
    fetch_pattern: String,
    sample_pattern: String,
}

// TODO separate the loader from a Libraries struct which contains shared dependencies.
// TODO support #ifdef
pub struct Loader {
    name_prefix: String,
    feature_prefix: String,
    imported: HashMap<String, Source>,
    patterns: HashMap<String, PatternNames>,
}

impl Loader {
    pub fn new(name_prefix: String) -> Self {
        Loader {
            name_prefix,
            feature_prefix: "FEATURE_".to_string(),
            imported: HashMap::new(),
            patterns: HashMap::new(),
        }
    }

    pub fn set_pattern(&mut self, id: &str, name: Option<&str>) {
        if let Some(name) = name {
            let lower_case = name.to_lowercase();

            self.patterns.insert(id.to_string(), PatternNames {
                matcher: format!("<{}>", id),
                library: format!("pattern::{}", lower_case),
                descriptor: name.to_string(),
                fetch_pattern: format!("fetch_{}", lower_case),
                sample_pattern: format!("sample_{}", lower_case),
            });
        } else {
            self.patterns.remove(id);
        }
    }

    pub fn load_shader_source(&mut self, descriptor: &ShaderDescriptor, permutation: usize) -> Result<String, std::io::Error> {
        for src in self.imported.values_mut() {
            src.already_imported = false;
        }

        let file_name = self.name_prefix.clone() + descriptor.file_name;

        let src = self.take_source(&file_name)?.unwrap();

        let mut output = String::new();

        output.push_str(&format!("// {}\n", descriptor.file_name));
        primary_features::feature_strings(descriptor.permutations[permutation].0, &mut |feature| {
            output.push_str(&format!("#define {}{}\n", self.feature_prefix, feature));
        });

        secondary_features::feature_strings(descriptor.permutations[permutation].1, &mut |feature| {
            output.push_str(&format!("#define {}{}\n", self.feature_prefix, feature));
        });

        output.push('\n');

        let status = self.parse(&src, &mut output);

        self.restore_source(&file_name, src);

        status?;

        Ok(output)
    }

    fn load_from_disk(&self, name: &str) -> Result<String, std::io::Error> {
        let file_name = self.name_prefix.clone() + name;

        println!("Loading {:?} from disk", file_name);

        let mut src = String::new();
        let mut file = std::fs::File::open(&file_name)?;
        file.read_to_string(&mut src)?;

        Ok(src)
    }

    fn take_source(&mut self, name: &str) -> Result<Option<String>, std::io::Error> {
        if !self.imported.contains_key(name) {
            let src = self.load_from_disk(name)?;
            self.imported.insert(name.to_string(), Source {
                src: String::new(),
                already_imported: true,
            });

            return Ok(Some(src));
        }

        let mut src = String::new();

        let mut imported = self.imported.get_mut(name).unwrap();
        if imported.already_imported {
            return Ok(None);
        }

        imported.already_imported = true;

        std::mem::swap(&mut src, &mut imported.src);

        Ok(Some(src))
    }

    fn restore_source(&mut self, name: &str, src: String) {
        self.imported.get_mut(name).unwrap().src = src;
    }

    fn parse(&mut self, src: &str, output: &mut String) -> Result<(), std::io::Error> {
        let mut line_buffer = String::new();
        const IMPORT_DIRECTIVE: &'static str = "#import ";
        for mut line in src.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with(IMPORT_DIRECTIVE) {
                let mut name = &trimmed[IMPORT_DIRECTIVE.len()..];
                let mut pattern_name = String::new();
                if name.starts_with("#pattern") {
                    for (_, pattern) in &self.patterns {
                        if name.contains(&pattern.matcher) {
                            pattern_name = pattern.library.clone();
                        }
                    }
                }

                if !pattern_name.is_empty() {
                    name = &pattern_name;
                }

                if let Some(src) = self.take_source(name)? {
                    output.push_str(&format!("// {}\n", name));

                    // Don't unwind before restoring the source we just took.
                    let status = self.parse(&src, output);

                    self.restore_source(name, src);

                    status?;
                }
            } else {
                if line.contains("#PatternDescriptor")
                || line.contains("#fetch_pattern")
                || line.contains("#sample_pattern") {

                    line_buffer.clear();
                    line_buffer.push_str(line);
                    for (_, pattern) in &self.patterns {
                        if line.contains(&pattern.matcher) {
                            let desc = format!("#PatternDescriptor{}", pattern.matcher);
                            let fetch = format!("#fetch_pattern{}", pattern.matcher);
                            let sample = format!("#sample_pattern{}", pattern.matcher);
                            line_buffer = line_buffer
                                .replace(&desc, &pattern.descriptor)
                                .replace(&fetch, &pattern.fetch_pattern)
                                .replace(&sample, &pattern.sample_pattern);
                            }
                    }

                    line = &line_buffer;
                }

                output.push_str(line);
                output.push('\n');
            }
        }

        Ok(())
    }

    pub fn add_source(&mut self, name: String, src: String) {
        self.imported.insert(name, Source { src, already_imported: false });
    }

    pub fn remove_source(&mut self, name: &str) {
        self.imported.remove(name);
    }

    pub fn dump_source(&self, source: &str) {
        let mut i: u32 = 0;
        for line in source.split("\n") {
            println!("{}\t|{}", i, line);
            i += 1;
        }
    }
}

#[test]
fn simple() {
    let mut loader = Loader::new("".into());

    loader.add_source(
        "A".to_string(),
"Hello(A),
#import B
#import B
#import A
end of A".to_string()
    );
    loader.add_source(
        "B".to_string(),
"Hello(B),
#import A
#import B
end of B".to_string()
    );


    use super::shader_list::{primary_features::*, secondary_features::*};

    let output = loader.load_shader_source(
        &ShaderDescriptor {
            name: super::ShaderName(0),
            file_name: "A",
            permutations: vec![(TEXTURE_2D, REPETITIONS, super::InitFlags::CACHED)],
        },
        0,
    ).unwrap();


    loader.dump_source(&output);

    assert_eq!(output.matches("Hello(A)").count(), 1);
    assert_eq!(output.matches("end of A").count(), 1);
    assert_eq!(output.matches("Hello(B)").count(), 1);
    assert_eq!(output.matches("end of B").count(), 1);
    assert_eq!(output.matches("#define FEATURE_TEXTURE_2D").count(), 1);
}


#[test]
fn patterns() {
    let mut loader = Loader::new(String::new());

    loader.set_pattern("color", Some("Image"));

    println!("patterns: {:?}", loader.patterns);

    loader.add_source(
        "pattern::image".to_string(),
"
#import image_sampler
#import rectangle_store

struct ImageDescriptor {
    @location(10) @interpolate(flat) rect: vec4<f32>,
};

fn fetch_image(id: u32) -> ImageDescriptor {
    return ImagePatern(
        rectangle_store_get(id),
    );
}

fn sample_image(uv: vec2<f32>, image: ImageDescriptor) -> vec4<f32> {
    return textureLoad(image_texture, uv, 0);
}
".to_string(),
    );

    loader.add_source(
        "image_sampler".to_string(),
        "".to_string(),
    );

    loader.add_source(
        "target".to_string(),
        "".to_string(),
    );

    loader.add_source(
        "rectangle_store".to_string(),
        "".to_string(),
    );

    loader.add_source(
        "shader".to_string(),
"
#import #pattern<color>
#import target

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(linear) uv: vec2<f32>,
    pattern: #PatternDescriptor<color>,
}

@vertex
fn main(
    @location(0) a_rect: vec4<f32>,
    @location(1) a_pattern: u32,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var vertices = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0)
    );

    var uv = vertices[vertex_index];

    var pattern = #fetch_pattern<color>(a_pattern);

    var screen_pos = (mix(a_rect.xy, a_rect.zw, uv) / target.resolution) * 2.0 - vec2<f32>(1.0);

    return VertexOutput(
        screen_pos,
        uv,
        pattern
    )
}

@fragment
fn main(
    @location(0) @interpolate(linear) uv: vec2<f32>,
    pattern: #PatternDescriptor<color>
) -> @location(0) vec4<f32> {
    return #sample_pattern<color>(uv, pattern);
}
".to_string()
    );

    use super::shader_list::{primary_features::*, secondary_features::*};

    let output = loader.load_shader_source(
        &ShaderDescriptor {
            name: super::ShaderName(0),
            file_name: "shader",
            permutations: vec![(TEXTURE_2D, REPETITIONS, super::InitFlags::CACHED)],
        },
        0,
    ).unwrap();

    println!("{}", output);
}
