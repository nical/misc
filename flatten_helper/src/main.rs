use std::{collections::HashMap, env, path::PathBuf};

use criterion::{read_markdown_table, ToleranceKey};
use table::MarkdownTable;

pub mod table;
pub mod criterion;
pub mod graph;

const TOLERANCES: &[f32] = &[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0];

const ALGORITHMS: &[&str] = &[
    "levien",
    "levien-simd",
    "levien-linear",
    "linear",
    "recursive",
    "hain",
    "wang",
    "wang-simd",
    "yzerman",
    "yzerman-simd",
    "fwd-diff",
    "hfd",
];

const USE_COLORS: &[(&str, (u8, u8, u8))] = &[
    ("levien",      (50, 125, 209)),
    ("levien-simd", (126, 176, 213)),
    ("levien-linear",(119, 211, 179)),
    ("wang",        (253, 204, 229)),
    ("wang-simd",   (199, 126, 190)),
    ("yzerman",     (120, 68, 33)),
    ("yzerman-simd",(200, 113, 55)),
    ("linear",      (128, 214, 87)),
    ("recursive",   (255, 238, 101)),
    ("fwd-diff",    (253, 127, 111)),
    ("hain",        (255, 181, 90)),
];

pub type BenchResults = HashMap<String, HashMap<ToleranceKey, f32>>;

fn main() {
    let mut args = std::env::args();
    let _first = args.next().unwrap();
    let cmd = args.next().expect("Expected a command (table, graph)");
    let mut curve_kind = None;
    if cmd == "criterion" {
        curve_kind = Some(args.next().expect("Expected the benchmark kind (cubic|quadratic)"));
    }

    let mut arg_title = false;
    let mut arg_subtitle = false;
    let mut arg_in = false;
    let mut arg_out = false;
    let mut title = String::new();
    let mut sub_title = "Normalized performance score (lower is better)".to_string();
    let mut input = None;
    let mut output = None;
    let mut color_scheme = graph::ColorScheme::Dark;
    let mut verbose = false;

    while let Some(arg) = args.next() {
        if arg_title {
            arg_title = false;
            title = arg;
            continue;
        }
        if arg_subtitle {
            arg_subtitle = false;
            sub_title = arg;
            continue;
        }
        if arg_in {
            arg_in = false;
            input = Some(arg);
            continue;
        }
        if arg_out {
            arg_out = false;
            output = Some(arg);
            continue;
        }

        match arg.as_str() {
            "-i" | "--input" => {
                arg_in = true;
            }
            "-o" | "--output" => {
                arg_out = true;
            }
            "-t" | "--title" => {
                arg_title = true;
            }
            "-s" | "--subtitle" => {
                arg_subtitle = true;
            }
            "--dark" => {
                color_scheme = graph::ColorScheme::Dark;
            }
            "--light" => {
                color_scheme = graph::ColorScheme::Light;
            }
            "-v" | "--verbose" => {
                verbose = true;
            }
            "--help" => {
                println!("Commands:");
                println!("  criterion: Import criterion's results and produce a markdown table");
                println!("  normalize: Read a markdown table, normalize the result so that the highest number is 1000 and output another markdown table.");
                println!("  graph: Read a markdown table and produce an SVG graph.");
                println!("Optional arguments:");
                println!("  -i / --input: Relative path for the input file.");
                println!("  -o / --output: Relative path for the output file.");
                println!("  -t / --title: A title to display in the graph.");
                println!("  -s / --subtitle: A sub-title to display in the graph.");
                println!("  --light: Use a light color scheme.");
                println!("  --dark: Use a dark color scheme.");
            }

            _ => {
                println!("Unrecognized arg: {arg:?}");
            }
        }
    }

    let mut _std_in = None;
    let mut _in_file = None;
    let input_reader: Option<&mut dyn std::io::Read> = if cmd.as_str() != "criterion" {
        Some(match &input {
            Some(file_name) => {
                if verbose {
                    println!("reading from file {file_name:?}");
                }
                let path: PathBuf = file_name.into();
                _in_file = Some(std::fs::File::open(path).unwrap());
                _in_file.as_mut().unwrap()
            }
            _ => {
                if verbose {
                    println!("reading from stdin");
                }
                _std_in = Some(std::io::stdin().lock());
                _std_in.as_mut().unwrap()
            }
        })
    } else {
        None
    };

    let mut _std_out = None;
    let mut _out_file = None;
    let output: &mut dyn std::io::Write = match &output {
        None => {
            if verbose {
                println!("writing into stdout");
            }
            _std_out = Some(std::io::stdout().lock());
            _std_out.as_mut().unwrap()
        }
        Some(file_name) => {
            if verbose {
                println!("writing into file {file_name:?}");
            }
            let path: PathBuf = file_name.into();
            _out_file = Some(std::fs::File::create(path).unwrap());
            _out_file.as_mut().unwrap()
        }
    };


    match cmd.as_str() {
        "criterion" => {
            let current_dir = env::current_dir().unwrap();
            let base_dir = input.unwrap_or("".into());

            let mut results: BenchResults = HashMap::new();

            criterion::read_criterion_results(
                curve_kind.unwrap().as_str(),
                &current_dir,
                &base_dir,
                verbose,
                &mut results
            );

            let mut normalized: BenchResults = HashMap::new();
            normalize_results(&results, None, &mut normalized);

            table::print_markdown_table(&normalized, output).unwrap();
        }
        "normalize" => {
            let mut results: BenchResults = HashMap::new();
            let table = MarkdownTable::parse(input_reader.unwrap()).unwrap();
            read_markdown_table(&table, &mut results);

            let mut normalized: BenchResults = HashMap::new();
            normalize_results(&results, None, &mut normalized);

            table::print_markdown_table(&normalized, output).unwrap();
        }
        "graph" => {
            let mut results: BenchResults = HashMap::new();
            let table = MarkdownTable::parse(input_reader.unwrap()).unwrap();
            read_markdown_table(&table, &mut results);

            graph::plot_graph(
                title.as_str(),
                sub_title.as_str(),
                &results,
                color_scheme,
                USE_COLORS,
                output,
            ).unwrap();
        }
        _ => {
            panic!("Unexpected command {cmd:?}")
        }
    }
}

pub fn normalize_results(input: &BenchResults, max: Option<f32>, output: &mut BenchResults) {
    let max_val = max.unwrap_or_else(&|| {
        let mut max_val: f32 = 0.0;
        for (_, results) in input {
            for value in results.values() {
                max_val = max_val.max(*value);
            }
        }
        max_val
    });

    for (algo, results) in input {
        let mut values = HashMap::new();
        for (tol, value) in results {
            values.insert(*tol, value * 1000.0 / max_val);
        }
        output.insert(algo.clone(), values);
    }
}
