use self::secondary_features::{NO_SECONDARY_FEATURES, ANTIALIASING};

use super::{
    InitFlags, ShaderDescriptor, PrimaryFeatures, SecondaryFeatures,
};
use super::CreatePipelineFn;
use super::preprocessor::{Preprocessor, SourceLoader, Source};

use std::sync::Arc;
use std::collections::HashMap;


macro_rules! decl_features {
    (
        $featurety:ident {
            $($name:ident : $bit:literal)*
        }
    ) => (
        $( pub const $name: $featurety = $featurety(1 << $bit); )*
        pub fn feature_strings(features: $featurety, strings: &mut dyn FnMut(&'static str)) {
            $( if (features & $name).0 != 0 { strings(stringify!($name)); } )*
        }
    )
}

pub mod shader_names {
    use crate::shaders::ShaderName;
    pub const IMAGE: ShaderName = ShaderName(1);
    pub const SOLID: ShaderName = ShaderName(2);
}

pub mod primary_features {
    use crate::shaders::PrimaryFeatures;
    pub const NO_PRIMARY_FEATURES: PrimaryFeatures = PrimaryFeatures(0);

    decl_features!(PrimaryFeatures {
        // name : bit
        ALPHA_PASS: 0
        TEXTURE_2D: 1
        TEXTURE_EXTERNAL: 2
        TEXTURE_RECT: 3
    });
}

pub mod secondary_features {
    use crate::shaders::SecondaryFeatures;
    pub const NO_SECONDARY_FEATURES: SecondaryFeatures = SecondaryFeatures(0);

    decl_features!(SecondaryFeatures {
        // name : bit
        ANTIALIASING: 0
        REPETITIONS: 1
    });
}

pub struct Namespace {
    pub name: String,
    pub sources: Vec<String>,
    pub children: Vec<Namespace>,
}

impl Namespace {
    pub fn new(name: impl Into<String>, sources: Vec<impl Into<String>>, children: Vec<Namespace>) -> Self {
        let mut src = Vec::new();
        for s in sources {
            src.push(s.into());
        }

        Namespace {
            name: name.into(),
            sources: src,
            children,
        }
    }
}

pub fn build_shader_source_list() -> Vec<Namespace> {
    vec![
        Namespace::new(
            "core",
            vec![
                "tiling",
                "target",
                "quad",
            ],
            vec![
                Namespace::new(
                    "store",
                    vec![
                        "gpu_cache",
                        "transform",
                        "atlas",
                    ],
                    vec![],
                ),
                Namespace::new(
                    "raster",
                    vec![
                        "lines",
                        "quadratic_beziers",
                    ],
                    vec![]
                ),
                Namespace::new(
                    "pattern",
                    vec![
                        "solid",
                        "image",
                        "linear_gradient",
                    ],
                    vec![]
                ),
            ]
        ),
    ]
}

pub fn load_shader_sources_from_disk(
    path: &str,
    prefix: &str,
    namespaces: &[Namespace],
    output: &mut HashMap<String, Source>,
) -> Result<(), std::io::Error> {
    use std::io::Read;
    for namespace in namespaces {
        for src in &namespace.sources {
            let shader_name = format!("{}{}::{}", prefix, namespace.name, src);
            let file_name = format!("{}{}/{}.wgsl", path, namespace.name, src);
            let mut file = std::fs::File::open(&file_name)?;
            let mut source = String::new();
            file.read_to_string(&mut source)?;
            output.insert(shader_name, source.into());
        }

        let child_prefix = format!("{}{}::", prefix, namespace.name);
        let child_path = format!("{}{}/", path, namespace.name);
        load_shader_sources_from_disk(
            &child_path,
            &child_prefix,
            &namespace.children,
            output,
        )?;
    }

    Ok(())
}

pub fn build_shader_list() -> Vec<ShaderDescriptor> {
    use primary_features::*;
    use secondary_features::*;
    
    vec![
        ShaderDescriptor {
            name: shader_names::IMAGE,
            string_name: "image",
            permutations: vec![
                (TEXTURE_2D, REPETITIONS, InitFlags::CACHED),
                (TEXTURE_2D, NO_SECONDARY_FEATURES, InitFlags::CACHED),
                (TEXTURE_2D | ALPHA_PASS, ANTIALIASING | REPETITIONS, InitFlags::CACHED),
                (TEXTURE_2D | ALPHA_PASS, NO_SECONDARY_FEATURES, InitFlags::CACHED),

                (TEXTURE_EXTERNAL, REPETITIONS, InitFlags::CACHED),
                (TEXTURE_EXTERNAL, NO_SECONDARY_FEATURES, InitFlags::CACHED),
                (TEXTURE_EXTERNAL | ALPHA_PASS, ANTIALIASING | REPETITIONS, InitFlags::CACHED),
                (TEXTURE_EXTERNAL | ALPHA_PASS, NO_SECONDARY_FEATURES, InitFlags::CACHED),

                (TEXTURE_RECT, REPETITIONS, InitFlags::CACHED),
                (TEXTURE_RECT, NO_SECONDARY_FEATURES, InitFlags::CACHED),
                (TEXTURE_RECT | ALPHA_PASS, ANTIALIASING | REPETITIONS, InitFlags::CACHED),
                (TEXTURE_RECT | ALPHA_PASS, NO_SECONDARY_FEATURES, InitFlags::CACHED),
            ],
        },

        ShaderDescriptor {
            name: shader_names::SOLID,
            string_name: "solid",
            permutations: vec![
                (NO_PRIMARY_FEATURES, NO_SECONDARY_FEATURES, InitFlags::CACHED),
                (ALPHA_PASS, NO_SECONDARY_FEATURES, InitFlags::CACHED),
                (ALPHA_PASS, ANTIALIASING, InitFlags::CACHED),
            ],
        },

        // ..
    ]
}

use super::{Shaders, ShaderKey};


pub fn register_shaders(
    registry: &mut Shaders,
    descriptors: &[ShaderDescriptor],
    create_fn: Arc<CreatePipelineFn>,
) {
    let mut prev_group_key = None;
    let mut builds = Vec::new();
    let mut current_group = Vec::new();
    for shader in descriptors {
        for permutation in &shader.permutations {
            let group_key = (shader.name, permutation.0);
            if Some(group_key) != prev_group_key {
                if !current_group.is_empty() {
                    registry.register_group(
                        shader.name,
                        permutation.0,
                        &current_group,
                        create_fn.clone(),
                    );
                    current_group.clear();
                }
                prev_group_key = Some(group_key);
            }
            if !permutation.2.lazy() {
                builds.push(ShaderKey {
                    shader: shader.name,
                    primary_features: permutation.0,
                    secondary_features: permutation.1,
                });
            }
            current_group.push(permutation.1);
        }
    }
}

pub fn generate_shader_sources(
    preprocessor: &mut Preprocessor,
    loader: &dyn SourceLoader,
    descriptors: &[ShaderDescriptor],
    output: &mut dyn FnMut(ShaderKey, String),
) {
    for shader in descriptors {
        for permutation in &shader.permutations {
            let key = ShaderKey {
                shader: shader.name,
                primary_features: permutation.0,
                secondary_features: permutation.1,
            };

            preprocessor.reset_defines();
            primary_features::feature_strings(
                permutation.0,
                &mut|feature_string| {
                    let define = format!("FEATURE_{}", feature_string);
                    preprocessor.define(&define);
                }
            );
            secondary_features::feature_strings(
                permutation.1,
                &mut|feature_string| {
                    let define = format!("FEATURE_{}", feature_string);
                    preprocessor.define(&define);
                }
            );

            let built = preprocessor.preprocess(
                shader.string_name,
                loader,
            ).unwrap();

            output(key, built);
        }
    }
}
