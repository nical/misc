use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::io::{self, BufRead, BufReader, Read, Write};

use crate::criterion::ToleranceKey;
use crate::{BenchResults, ALGORITHMS, TOLERANCES};

#[derive(Debug)]
pub struct MarkdownTable {
    pub headers: Vec<String>,
    pub row_labels: Vec<String>,
    pub data: Vec<Vec<f32>>,
}

#[derive(Debug)]
pub enum ParseError {
    InvalidTableFormat,
    EmptyTable,
    HeaderParseError,
    DataParseError(usize, usize),
    IoError(io::Error),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ParseError::InvalidTableFormat => write!(f, "Invalid markdown table format"),
            ParseError::EmptyTable => write!(f, "Table is empty"),
            ParseError::HeaderParseError => write!(f, "Failed to parse headers"),
            ParseError::DataParseError(row, col) => write!(f, "Failed to parse data at row {}, column {}", row, col),
            ParseError::IoError(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl Error for ParseError {}

impl From<io::Error> for ParseError {
    fn from(error: io::Error) -> Self {
        ParseError::IoError(error)
    }
}

impl MarkdownTable {
    pub fn parse<R: Read>(reader: R) -> Result<Self, ParseError> {
        let buf_reader = BufReader::new(reader);
        let lines: Vec<String> = buf_reader
            .lines()
            .collect::<Result<Vec<String>, io::Error>>()?;

        let filtered_lines: Vec<&str> = lines
            .iter()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        if filtered_lines.len() < 3 {
            return Err(ParseError::InvalidTableFormat);
        }

        // Parse headers from first line
        let headers = parse_headers(filtered_lines[0].trim())?;

        // Validate separator line
        if !is_valid_separator(filtered_lines[1].trim(), headers.len()) {
            return Err(ParseError::InvalidTableFormat);
        }

        // Parse data rows with string first column
        let mut row_labels = Vec::new();
        let mut data = Vec::new();

        for (idx, line) in filtered_lines.iter().skip(2).enumerate() {
            let (label, row) = parse_mixed_data_row(line.trim(), headers.len(), idx)?;
            row_labels.push(label);
            data.push(row);
        }

        Ok(MarkdownTable { headers, row_labels, data })
    }

    pub fn headers(&self) -> &[String] {
        &self.headers
    }

    pub fn row_labels(&self) -> &[String] {
        &self.row_labels
    }

    pub fn data(&self) -> &[Vec<f32>] {
        &self.data
    }
}

fn parse_headers(line: &str) -> Result<Vec<String>, ParseError> {
    if !line.starts_with('|') || !line.ends_with('|') {
        return Err(ParseError::HeaderParseError);
    }

    let headers: Vec<String> = line
        .trim_matches('|')
        .split('|')
        .map(|s| s.trim().to_string())
        .collect();

    if headers.is_empty() {
        return Err(ParseError::HeaderParseError);
    }

    Ok(headers)
}

fn is_valid_separator(line: &str, expected_cols: usize) -> bool {
    if !line.starts_with('|') || !line.ends_with('|') {
        return false;
    }

    let separators: Vec<&str> = line
        .trim_matches('|')
        .split('|')
        .map(str::trim)
        .collect();

    if separators.len() != expected_cols {
        return false;
    }

    separators.iter().all(|s| s.chars().all(|c| c == '-' || c == ':'))
}

fn parse_mixed_data_row(line: &str, expected_cols: usize, row_idx: usize) -> Result<(String, Vec<f32>), ParseError> {
    if !line.starts_with('|') || !line.ends_with('|') {
        return Err(ParseError::DataParseError(row_idx, 0));
    }

    let cells: Vec<&str> = line
        .trim_matches('|')
        .split('|')
        .map(str::trim)
        .collect();

    if cells.len() != expected_cols {
        return Err(ParseError::DataParseError(row_idx, 0));
    }

    // First column is a string label
    let label = cells[0].to_string();

    // Remaining columns are floating point values
    let mut row = Vec::with_capacity(cells.len() - 1);
    for (col_idx, cell) in cells.iter().skip(1).enumerate() {
        // Add 1 to col_idx when reporting errors to account for the skipped first column
        match cell.parse::<f32>() {
            Ok(value) => row.push(value),
            Err(_) => return Err(ParseError::DataParseError(row_idx, col_idx + 1)),
        }
    }

    Ok((label, row))
}

pub fn print_markdown_table(results: &BenchResults, output: &mut dyn Write) -> std::io::Result<()> {
    const ALGO_CHARS: usize = 14;

    write!(output, "| tolerance    ")?;
    for tolerance in TOLERANCES {
        write!(output, "|   {} ", tolerance)?;
    }
    writeln!(output, "|")?;
    write!(output, "|--------------")?;
    for _ in 0..TOLERANCES.len() {
        write!(output, "| ------:")?;
    }
    writeln!(output, "|")?;
    let mut seen = HashSet::new();
    let mut ordered = Vec::new();
    for algo in ALGORITHMS {
        if results.contains_key(*algo) {
            seen.insert(*algo);
            ordered.push(*algo);
        }
    }
    for algo in results.keys() {
        if !seen.contains(algo.as_str()) {
            ordered.push(algo.as_str());
        }
    }

    for algo in ordered {
        if let Some(scores) = results.get(algo) {
            seen.insert(algo);
            write!(output, "|{}", algo)?;
            if algo.len() < ALGO_CHARS {
                for _ in 0..ALGO_CHARS - algo.len() {
                    write!(output, " ")?;
                }
            }
            for tol in TOLERANCES {
                match scores.get(&ToleranceKey::new(*tol)) {
                    Some(val) => write!(output, "| {:.2} ", val)?,
                    None => write!(output, "|    ")?,
                }
            }
            writeln!(output, "|")?;
        }
    }


    Ok(())
}


#[test]
fn parse() {
    // Example with a string that implements Read (through Cursor)
    let markdown_table = r#"
        | Label | Value | Score |
        |-------|-------|-------|
        | Apple | 42.0  | 98.6  |
        | Banana| 15.7  | 77.1  |
        | Cherry| 33.2  | 88.9  |


    "#;

    let cursor = std::io::Cursor::new(markdown_table);
    let table = MarkdownTable::parse(cursor).unwrap();

    println!("Headers: {:?}", table.headers());
    println!("Row Labels and Data:");
    for (i, row) in table.data().iter().enumerate() {
        println!("{}: {:?}", table.row_labels()[i], row);
    }

    // Example with a file (commented out)
    /*
    let file = std::fs::File::open("table.md")?;
    let table = MarkdownTable::parse(file)?;

    println!("Headers from file: {:?}", table.headers());
    // ... rest of processing
    */
}
