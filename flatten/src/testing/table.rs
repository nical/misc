use std::io::Write;

use crate::TOLERANCES;

pub fn get_flatten_output() -> Option<String> {
    let mut out_name = None;
    for (var, val) in std::env::vars() {
        if var == "FLATTEN_OUTPUT" {
            out_name = Some(val);
        }
    }

    out_name
}


pub fn print_first_row_md(output: &mut dyn Write) {
    let _ = write!(output, "| tolerance  ");
    for tolerance in &TOLERANCES {
        let _ = write!(output, "|  {} ", tolerance);
    }
    let _ = writeln!(output, "|");
    let _ = write!(output, "|-----------");
    for _ in 0..TOLERANCES.len() {
        let _ = write!(output, "| -----:");
    }
    let _ = writeln!(output, "|");
}

pub fn print_row_md<T: std::fmt::Display>(output: &mut dyn Write, name: &str, vals: &[T]) {
    let _ = write!(output, "|{}", name);
    for val in vals {
        let _ = write!(output, "| {:.4} ", val);
    }
    let _ = writeln!(output, "|");
}
