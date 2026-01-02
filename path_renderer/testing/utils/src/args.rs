use clap::Parser;

#[derive(Parser, Debug)]
pub struct Args {
    pub test_name: String,

    pub backend: Option<String>,

    pub fuzzy: Option<u32>,
}

pub fn parse() -> Args {
    Args::parse()
}
