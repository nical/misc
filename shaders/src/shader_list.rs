use crate::{
    ShaderName, PrimaryFeatures, SecondaryFeatures, InitFlags, ShaderDescriptor,
};

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
    use crate::ShaderName;
    pub const IMAGE: ShaderName = ShaderName(1);
    pub const SOLID: ShaderName = ShaderName(2);
}

pub mod primary_features {
    use crate::PrimaryFeatures;
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
    use crate::SecondaryFeatures;
    pub const NO_SECONDARY_FEATURES: SecondaryFeatures = SecondaryFeatures(0);

    decl_features!(SecondaryFeatures {
        // name : bit
        ANTIALIASING: 0
        REPETITIONS: 1
    });
}


use primary_features::*;
use secondary_features::*;
pub fn build_shader_list() -> Vec<ShaderDescriptor> {
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
