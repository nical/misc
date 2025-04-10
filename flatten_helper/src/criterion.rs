use std::{collections::HashMap, error::Error, fs::File, io, path::PathBuf};
use serde::Deserialize;

use crate::{table::MarkdownTable, BenchResults, ALGORITHMS, TOLERANCES};


#[derive(Deserialize)]
pub struct CondidenceInterval {
    pub confidence_level: f32,
    pub upper_bound: f32,
    pub lower_bound: f32,
}

#[derive(Deserialize)]
pub struct BenchmarkResult {
    pub confidence_interval: CondidenceInterval,
    pub point_estimate: f32,
    pub standard_error: f32,
}

#[derive(Deserialize)]
pub struct BenchmarkResults {
    pub mean: BenchmarkResult,
    pub median: Option<BenchmarkResult>,
    pub median_abs_dev: Option<BenchmarkResult>,
    pub slope: Option<BenchmarkResult>,
    pub std_dev: Option<BenchmarkResult>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ToleranceKey(u32);
impl ToleranceKey {
    pub fn new(tol: f32) -> ToleranceKey {
        ToleranceKey(unsafe { std::mem::transmute(tol) })
    }
    pub fn get(&self) -> f32 {
        unsafe { std::mem::transmute(self.0) }
    }
}


pub fn read_bench(path: &PathBuf) -> Result<BenchmarkResults, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = io::BufReader::new(file);
    let val: BenchmarkResults = serde_json::from_reader(reader)?;

    Ok(val)
}

pub fn read_criterion_results(
    kind: &str,
    current_dir: &PathBuf,
    base_dir: &str,
    verbose: bool,
    results: &mut BenchResults,
) {
    for algo in ALGORITHMS {
        for tol in TOLERANCES {
            let mut path = current_dir.clone();
            path.push(base_dir);
            path.push(&format!("target/criterion/{kind}/{algo}/{tol}/new/estimates.json"));

            match read_bench(&path) {
                Ok(res) => {
                    //println!("{path:?}");
                    let score = res.mean.point_estimate;
                    let entry = results.entry(algo.to_string()).or_insert(HashMap::new());
                    entry.insert(ToleranceKey::new(*tol), score);
                }
                Err(e) => {
                    if verbose {
                        println!("Failed to read {path:?} {e:?}");
                    }
                }
            }
        }
    }
}

pub fn read_markdown_table(
    table: &MarkdownTable,
    results: &mut BenchResults,
) {
    for (i, row) in table.data.iter().enumerate() {
        let algo = table.row_labels[i].clone();

        let mut values = HashMap::new();

        for (tol_str, val) in table.headers[1..].iter().zip(row.iter()) {
            let tol: f32 = tol_str.parse().expect("Expected a tolerance value, got {tol_str:?}");
            values.insert(ToleranceKey::new(tol), *val);
        }

        results.insert(algo, values);
    }
}
