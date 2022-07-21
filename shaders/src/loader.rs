use crate::ShaderDescriptor;
use crate::shader_list::{primary_features, secondary_features};

use std::collections::HashMap;
use std::io::Read;

struct Source {
    src: String,
    already_imported: bool,
}

pub struct Loader {
    name_prefix: String,
    feature_prefix: String,
    imported: HashMap<String, Source>,
}

impl Loader {
    pub fn new(name_prefix: String) -> Self {
        Loader {
            name_prefix,
            feature_prefix: "FEATURE_".to_string(),
            imported: HashMap::new(),
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
        const IMPORT_DIRECTIVE: &'static str = "#import ";
        for line in src.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with(IMPORT_DIRECTIVE) {
                let name = &trimmed[IMPORT_DIRECTIVE.len()..];
                if let Some(src) = self.take_source(name)? {
                    output.push_str(&format!("// {}\n", name));

                    // Don't unwind before restoring the source we just took.
                    let status = self.parse(&src, output);

                    self.restore_source(name, src);

                    status?;
                }
            } else {
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


    use crate::shader_list::{primary_features::*, secondary_features::*};

    let output = loader.load_shader_source(
        &ShaderDescriptor {
            name: crate::ShaderName(0),
            file_name: "A",
            permutations: vec![(TEXTURE_2D, REPETITIONS, crate::InitFlags::CACHED)],
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
