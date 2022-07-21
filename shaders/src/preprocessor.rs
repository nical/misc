use std::collections::HashMap;
use std::io::Write;

struct Iter<'l> {
    c: char,
    iter: std::str::Chars<'l>,
}

impl<'l> Iter<'l> {
    fn advance(&mut self) -> bool {
        self.c = self.iter.next();
        self.c.is_some()
    }
}

enum Error {
    InvalidMacroName { name: String },
    UnexpectedToken { token: String },
}

pub struct Parser {
    defines: HashMap<String, String>,
    writing: bool,
    buffer: String,
}

impl Parser {
    fn process_line(&mut self, line: &str, output: &mut Write) {
        let line = line
            .strip_suffix("\r\n")
            .or_else(|| line.strip_suffix('\n'))
            .unwrap_or(line);

        let mut iter = line.chars();
        while let Some(c) = iter.next();
            if c == '#' {
                self.parse_macro(&mut iter);
            } else if self.writing {
                self.buffer.push(c);
            }
        }

        if !self.buffer.is_empty() {
            output.write(self.buffer.as_bytes());
        }
    }

    fn skip_whitespaces(&mut self, iter: mut Iter) {
        while let Some(c) = iter.c {
            if !c.is_whitespace() {
                break;
            }

            iter.advance();
        }
    }

    fn parse_ident(&mut self, iter: &mut Iter, output: &mut String) {
        self.skip_whitespaces(iter);

        while let Some(c) = iter.c {
            if !c.is_alphanumeric() && c != '_' {
                break;
            }

            output.push(c);

            iter.advance();
        }

    }

    fn parse_macro(&mut self, iter: &mut Iter) {
        let mut name = String::new();

        while let Some(c) = c {
            if x.is_alphanumeric() {
                name.push(c);
            } else {
                break;
            }

            c = iter.next();
        }

        self.skip_whitespaces(iter);

        match &name {
            "define" => {
                let mut macro_name = String::new();
                let mut macro_body = String::new();
                self.parse_ident(iter, &mut macro_name);
                self.skip_whitespaces(iter);

                if macro_name.is_empty() {
                    unimplemented!(); // TODO: error.
                }

                if iter.c.is_none() {
                    self.add_simple_define(macro_name);
                } else if let Some('(') {
                    self.parse_macro_params(iter);
                }
            }
            "undef" => {
                let mut macro_name = String::new();
                self.parse_ident(iter, &mut macro_name);                
                if macro_name.is_empty() {
                    unimplemented!(); // TODO: error.
                }
            }
            "if" =>
            "ifdef" =>
            "else" =>
            "endif" =>
            "include" =>
            _ =>
        }
    }

    fn parse_macro_params(&mut self, iter: &mut Iter) {
        unimplemented!();
    }

    fn add_simple_define(&mut self, name: String) {
        unimplemented!();
    }
}