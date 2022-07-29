use self::secondary_features::{NO_SECONDARY_FEATURES, ANTIALIASING};

use super::{
    InitFlags, ShaderDescriptor, PrimaryFeatures, SecondaryFeatures,
};
use super::ShaderName;


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

pub struct ShaderGroupDesc {
    pub name: ShaderName,
    pub file_name: String,
    pub variants: Vec<Features>,
}

pub struct Features {
    pub primary: PrimaryFeatures,
    pub secondary: Vec<SecondaryFeatures>,
}

pub fn build_shader_list2() -> Vec<ShaderGroupDesc> {
    use primary_features::*;
    use secondary_features::*;

    vec![
        ShaderGroupDesc {
            name: shader_names::IMAGE,
            file_name: "image.wgsl".to_string(),
            variants: vec![
                Features {
                    primary: TEXTURE_2D, 
                    secondary: vec![
                        NO_SECONDARY_FEATURES,
                        REPETITIONS,
                    ]
                },
                Features {
                    primary: TEXTURE_2D | ALPHA_PASS,
                    secondary: vec![
                        NO_SECONDARY_FEATURES,
                        ANTIALIASING | REPETITIONS,
                    ]
                },
            ]
        },
    ]
}

pub fn build_shader_list() -> Vec<ShaderDescriptor> {
    use primary_features::*;
    use secondary_features::*;
    
    vec![
        ShaderDescriptor {
            name: shader_names::IMAGE,
            file_name: "image.wgsl",
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
            file_name: "solid.wgsl",
            permutations: vec![
                (NO_PRIMARY_FEATURES, NO_SECONDARY_FEATURES, InitFlags::CACHED),
                (ALPHA_PASS, NO_SECONDARY_FEATURES, InitFlags::CACHED),
                (ALPHA_PASS, ANTIALIASING, InitFlags::CACHED),
            ],
        },

        // ..
    ]
}
