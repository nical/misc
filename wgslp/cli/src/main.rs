#[macro_use]
extern crate serde_derive;

use wgslp::preprocessor::{Preprocessor, Source};
use std::{io, path::{PathBuf}};
use std::fs::File;
use std::io::Read;
use std::collections::HashMap;

type SourceLibrary = HashMap<String, Source>;

use clap::{Parser};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    #[clap(value_parser)]
    shader: String,
    #[clap(short, long, value_parser)]
    shader_directory: Option<PathBuf>,
    #[clap(long, value_parser)]
    features: Option<String>,
    #[clap(short, long, value_parser)]
    output: Option<String>,
    #[clap(short, long, action)]
    validate: bool,
    #[clap(long, action)]
    verbose: bool,
}

#[derive(Deserialize)]
struct Config {
    pub shader_directory: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();

    let verbose = args.verbose;

    let config = File::open("wgslp.toml").ok().map(|mut file| {
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).expect("Failed to read global config file.");
        let config: Config = toml::from_slice(&buf[..]).unwrap();

        config
    });

    if verbose && config.is_some() {
        println!(" - Found a wgslp.toml config file");
    }

    let shader_dir = args.shader_directory
        .or_else(|| { config.map(|cfg| cfg.shader_directory).flatten() })
        .unwrap_or_else(|| { std::env::current_dir().unwrap() });

    if verbose {
        println!(" - Shader directory: {:?}", shader_dir);
    }

    let mut features = Vec::new();
    if let Some(features_str) = args.features.as_ref() {
        for feature in features_str.split(' ') {
            features.push(feature.to_string());
        }
    }

    let mut preprocessor = Preprocessor::new();

    let library = build_library(&shader_dir, verbose).unwrap();

    if verbose {
        println!(" - Preprocessing shader {:?}", args.shader);
        println!(" - Features: {:?}", features);
    }

    for feature in &features {
        preprocessor.define(feature);
    }
    let source = library.get(&args.shader).cloned().unwrap();
    let shader_source = preprocessor.preprocess(&args.shader, &source, &library).unwrap();

    if args.validate {
        let mut validator = wgslp::validator::Validator::new();

        match validator.validate(&shader_source) {
            Ok(_) => {}
            Err(error) => {
                error.emit_to_stderr(&shader_source);
                std::process::exit(1);
            }
        }    
    }

    if let Some(dst) = args.output.as_ref() {
        write_to_stream(&shader_source, dst);
    }
}

pub fn build_library(
    shader_dir: &std::path::Path,
    verbose: bool,
) -> Result<SourceLibrary, io::Error> {

    let mut library = SourceLibrary::default();

    for entry in walkdir::WalkDir::new(&shader_dir) {
        if let Ok(entry) = entry {
            if entry.metadata().map(|md| md.is_dir()).unwrap_or(true) {
                continue;
            }
            let is_wgsl = entry.file_name().to_str().map(|s| s.ends_with(".wgsl")).unwrap_or(false);
            if !is_wgsl {
                continue;
            }

            let absolute_path = entry.into_path();
            let relative_path = match absolute_path.strip_prefix(&shader_dir) {
                Ok(path) => path,
                Err(_) => {
                    continue;
                }
            };

            let shader_name = shader_name(&relative_path);

            let mut src = String::new();
            let mut file = std::fs::File::open(&absolute_path)?;
            file.read_to_string(&mut src)?;
            if verbose {
                println!(" - Loaded {} ({:?})", shader_name, relative_path);
            }
            library.insert(shader_name, src.into());
        }
    }

    Ok(library)
}

pub fn shader_name(relative_path: &std::path::Path) -> String {
    let mut first = true;
    let mut shader_name = String::new();
    for item in relative_path.components() {
        if first {
            first = false;
        } else {
            shader_name.push_str("::")
        }
        shader_name.push_str(item.as_os_str().to_str().unwrap());
    }
    for _ in 0..".wgsl".len() {
        shader_name.pop();
    }

    shader_name
}

fn write_to_stream(shader_source: &str, dst: &str) {
    let mut stdout = std::io::stdout();
    let mut _out_file = None;
    let output: &mut dyn std::io::Write = match dst {
        "stdout" => {
            &mut stdout
        }
        path => {
            let file = match std::fs::File::create(path) {
                Ok(file) => file,
                Err(error) => {
                    eprintln!("{}", error.to_string());
                    std::process::exit(1);
                }
            };
            _out_file = Some(file);
            
            _out_file.as_mut().unwrap()
        }
    };

    let _ = write!(output, "{}", shader_source);
}
