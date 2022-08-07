use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::sync::Arc;
use std::io;

pub type Source = Arc<str>;

/// A trait to obtain source code from custom providers during preprocessing..
///
/// The module provides a few implementations of this trait:
/// - the unit type `()` (which will always fail to load the requested source),
/// - `LoadFromDisk`,
/// - `HasMap<String, Source>`,
/// - `(&impl SourceLoader, &impl SourceLoader)` which tries the first element in the
///   tuple and defaults to the second one.
pub trait SourceLoader {
    fn load_source(&self, name: &str) -> Result<Source, io::Error>;
}

impl SourceLoader for () {
    fn load_source(&self, name: &str) -> Result<Source, io::Error> {
        Err(io::Error::new(io::ErrorKind::NotFound, name))
    }
}

/// Loads source code from disk.
pub struct LoadFromDisk {
    // TODO: use Path?
    prefix: String,
}

impl LoadFromDisk {
    pub fn new() -> Self {
        LoadFromDisk::with_prefix(String::new())
    }

    pub fn with_prefix(mut prefix: String) -> Self {
        if !prefix.is_empty() && !prefix.ends_with("/") {
            prefix.push('/');
        }
        LoadFromDisk { prefix }
    }
}

impl SourceLoader for LoadFromDisk {
    fn load_source(&self, name: &str) -> Result<Source, io::Error> {
        let mut src = String::new();
        let file_name = format!("{}{}.wgsl", self.prefix, name.replace("::","/"));
        let mut file = std::fs::File::open(&file_name)?;
        file.read_to_string(&mut src)?;

        Ok(src.into())    
    }
}

impl SourceLoader for HashMap<String, Source> {
    fn load_source(&self, name: &str) -> Result<Source, io::Error> {
        if let Some(src) = self.get(name) {
            return Ok(src.clone());
        }

        Err(io::Error::new(io::ErrorKind::NotFound, name))
    }
}

impl<'l, T1, T2> SourceLoader for (&'l mut T1, &'l mut T2)
where
    T1: SourceLoader,
    T2: SourceLoader,
{
    fn load_source(&self, name: &str) -> Result<Source, io::Error> {
        if let Ok(src) = self.0.load_source(name) {
            return Ok(src)
        }

        self.1.load_source(name)
    }
}

#[derive(Debug)]
pub enum SourceError {
    Io(std::io::Error),
    Parse(String),
}

impl From<std::io::Error> for SourceError {
    fn from(e: std::io::Error) -> Self {
        SourceError::Io(e)
    }
}

/// A rudimentary source code preprocessor.
///
/// It can:
/// - Import source code from a source library (#import statement).
/// - Include or exclude parts of the code via static #if FOO { } #else blocks.
///   These currently only support checking whether a symbol is defined.
pub struct Preprocessor {
    imported: HashSet<String>,
    defines: HashSet<String>,
    discarding: i32,
    in_mixin: bool,
}

impl Preprocessor {
    pub fn new() -> Self {
        Preprocessor {
            imported: HashSet::default(),
            defines: HashSet::new(),
            discarding: 0,
            in_mixin: false,
        }
    }

    pub fn preprocess(
        &mut self,
        name: &str,
        loader: &dyn SourceLoader,
    ) -> Result<String, SourceError> {
        self.imported.clear();
        self.discarding = 0;
        self.in_mixin = false;
        
        let src = self.import_source(&name, loader)?.unwrap();

        let mut output = String::with_capacity(4096);

        output.push_str(&format!("// {}\n", name));
        for define in &self.defines {
            output.push_str(&format!("// * {}\n", define));
        }

        let mut tokenizer = Tokenizer::new(name, &src);
        self.parse(&mut tokenizer, &mut output, loader)?;

        Ok(output)
    }

    pub fn reset_defines(&mut self) {
        self.defines.clear();
    }

    pub fn define(&mut self, name: &str) {
        self.defines.insert(name.to_string());
    }

    pub fn undefine(&mut self, name: &str) -> bool {
        self.defines.remove(name)
    }

    fn import_source<'lib>(
        &mut self,
        name: &str,
        loader: &dyn SourceLoader,
    ) -> Result<Option<Arc<str>>, std::io::Error> {
        if self.imported.contains(name) {
            return Ok(None);
        }

        self.imported.insert(name.to_string());

        let src = loader.load_source(name)?;
        
        Ok(Some(src))
    }

    fn parse(&mut self, src: &mut Tokenizer, output: &mut String, loader: &dyn SourceLoader) -> Result<(), SourceError> {
        while let Some(token) = src.current() {
            src.advance();
            match token {
                Token::Import => {
                    let name = self.parse_import(src)?;
                    if let Some(src) = self.import_source(&name, loader)? {
                        output.push_str(&format!("// {}\n", name));
                        self.parse(&mut Tokenizer::new(&name, &src), output, loader)?;
                    }
                }
                Token::Mixin => {
                    if self.in_mixin {
                        src.parse_error("Nested mixin")?;
                    }
                    let name = self.parse_import(src)?;
                    let src = loader.load_source(&name)?;
                    self.in_mixin = true;
                    self.parse(&mut Tokenizer::new(&name, &src), output, loader)?;
                    self.in_mixin = false;
                }
                Token::StaticIf => {
                    self.parse_static_if(src, output)?;
                }
                Token::StaticElse => {
                    // Note: Here we assume that we are after the end of a static if block
                    // which was not discarded (and therefore the #else gets discarded).
                    // But we don't check. So a static else block that doesn't come after
                    // a static if block is always discarded while it would make more sense
                    // to treat it as an error.
                    self.parse_discarded_static_else(src)?;
                }
                other => {
                    output.push_str(other.as_str())
                }
            }
        }

        Ok(())
    }

    fn parse_static_if(&mut self, src: &mut Tokenizer, output: &mut String) -> Result<(), SourceError> {
        let cond = self.parse_static_if_condition(src)?;
        if !cond {
            self.parse_discarded_blocks(src)?;
        }
        self.parse_static_else(src, cond, output)?;

        Ok(())
    }

    fn parse_discarded_static_else(&mut self, src: &mut Tokenizer) -> Result<(), SourceError> {
        while let Some(token) = src.current() {
            match token {
                Token::WhiteSpace(_)
                | Token::NewLine
                | Token::Comment(_)
                => {}
                Token::OpenBracket => {
                    break;
                }
                Token::Basic("if") => {
                    src.parse_error("#else if is not supported yet")?;
                }
                other => {
                    src.parse_error(format!("Expected '{{', got {:?}", other.as_str()))?;
                }
            }
            src.advance();
        }

        if src.current().is_none() {
            src.parse_error("Expected '{'")?;
        }

        self.parse_discarded_blocks(src)?;

        Ok(())
    }

    fn parse_import(&mut self, src: &mut Tokenizer) -> Result<String, SourceError> {
        src.consume_whitespace();
        let mut name = String::with_capacity(32);

        while let Some(token) = src.current() {
            match token {
                Token::WhiteSpace(_)
                | Token::NewLine
                | Token::Comment(_)
                => {
                    break;
                }
                Token::Basic(text) => {
                    name.push_str(text);
                    src.advance();
                }
                _ => {
                    src.parse_error(format!("Unexpected {:?} in import declaration", token.as_str()))?;
                }
            }
        }

        Ok(name)
    }

    fn parse_static_if_condition(&mut self, src: &mut Tokenizer) -> Result<bool, SourceError> {

        while src.consume_whitespace()
        || src.consume_comment()
        || src.consume_new_line() {}

        let cond;
        match src.current() {
            Some(Token::Basic(text)) => {
                let mut name = text;
                let mut neg = false;
                while name.starts_with('!') {
                    neg = !neg;
                    name = &name[1..];
                }

                cond = self.defines.contains(name) != neg;
            }
            Some(other) => {
                src.parse_error(format!("Unexpected {:?} in static if condition", other.as_str()))?;
                unreachable!();
            }
            None => {
                src.parse_error("Unterminated static if condition")?;
                unreachable!();
            }
        }
        src.advance();

        while src.consume_whitespace()
        || src.consume_comment()
        || src.consume_new_line() {}

        match src.current() {
            Some(Token::OpenBracket) => {}
            Some(other) => {
                src.parse_error(format!("Expected '{{', got {:?} after static if condition", other.as_str()))?;
            }
            None => {
                src.parse_error("Unterminated static if condition")?;
            }
        }

        Ok(cond)
    }

    fn parse_discarded_blocks(&mut self, src: &mut Tokenizer) -> Result<(), SourceError> {
        let mut blocks: i32 = 0;
        while let Some(token) = src.current() {
            src.advance();
            match token {
                Token::OpenBracket => { blocks += 1 }
                Token::CloseBracket => { blocks -= 1 }
                _ => {}
            }
            if blocks == 0 {
                break;
            }
            assert!(blocks > 0);
        }

        if blocks != 0 {
            src.parse_error("Unmatched {} block")?;
        }

        Ok(())
    }

    fn parse_static_else(&mut self, src: &mut Tokenizer, discard: bool, output: &mut String) -> Result<(), SourceError> {
        while let Some(token) = src.current() {
            match token {
                Token::StaticElse => {
                    src.advance();
                    if discard {
                        self.parse_discarded_blocks(src)?;
                    }

                    return Ok(());
                }
                Token::WhiteSpace(_)
                | Token::NewLine
                | Token::Comment(_)
                => {
                    src.advance();
                    output.push_str(token.as_str());
                }
                _ => {
                    break;
                }
            }
        }

        Ok(())
    }
}

pub fn print_source(source: &str) {
    let mut i: u32 = 0;
    for line in source.split("\n") {
        println!("{}\t|{}", i, line);
        i += 1;
    }
}

#[derive(Clone)]
struct Tokenizer<'l> {
    src: &'l str,
    line: u32,
    src_name: &'l str,
    current: Option<Token<'l>>
}

#[derive(Copy, Clone, Debug)]
enum Token<'l> {
    Basic(&'l str),
    WhiteSpace(&'l str),
    Comment(&'l str),
    StaticIf,
    StaticElse,
    Import,
    Mixin,
    NewLine,
    OpenBracket,
    CloseBracket,
}

impl<'l> Token<'l> {
    fn as_str(&self) -> &'l str {
        match self {
            Token::Comment(text)
            | Token::Basic(text)
            | Token::WhiteSpace(text) => text,
            Token::NewLine => &"\n",
            Token::OpenBracket => &"{",
            Token::CloseBracket => &"}",
            Token::Import => &"#import",
            Token::Mixin => &"#mixin",
            Token::StaticIf => &"#if",
            Token::StaticElse => &"#else",
        }
    }
}

impl<'l> Tokenizer<'l> {
    fn new(src_name: &'l str, src: &'l str) -> Self {
        let mut tokenizer = Tokenizer {
            src_name,
            src,
            line: 1,
            current: None
        };
        tokenizer.advance();
        
        tokenizer
    }

    fn parse_error(&self, msg: impl AsRef<str>) -> Result<(), SourceError> {
        let message = format!("{}, line {}: {}", self.src_name, self.line, msg.as_ref());
        Err(SourceError::Parse(message))
    }

    
    fn advance(&mut self) -> bool {
        self.current = self.read_token_impl();
        self.current.is_some()
    }

    fn current(&self) -> Option<Token<'l>> {
        self.current
    }

    fn consume_whitespace(&mut self) -> bool {
        if let Some(Token::WhiteSpace(_)) = self.current {
            self.advance();
            return true;
        }

        false
    }

    fn consume_comment(&mut self) -> bool {
        if let Some(Token::Comment(_)) = self.current {
            self.advance();
            return true;
        }

        false
    }

    fn consume_new_line(&mut self) -> bool {
        if let Some(Token::NewLine) = self.current {
            self.advance();
            return true;
        }

        false
    }

    fn read_token_impl(&mut self) -> Option<Token<'l>> {
        let mut chars = self.src.chars();
        let c1 = chars.next();
        let c2 = chars.next();
        match (c1, c2) {
            (Some('/'), Some('/')) => {
                let split = self.src.find('\n').unwrap_or(self.src.len());
                let comment = &self.src[..split];
                self.src = &self.src[split..];
                return Some(Token::Comment(comment));
            }
            (Some('/'), Some('*')) => {
                let split = self.src.find("*/").unwrap_or(self.src.len());
                let comment = &self.src[..split];
                self.src = &self.src[split..];
                return Some(Token::Comment(comment));
            }
            (Some('#'), _) => {
                let split = self.src[1..]
                    .find(|c: char| !c.is_alphanumeric())
                    .map(|offset| offset + 1)
                    .unwrap_or(self.src.len());
                let keyword = &self.src[..split];
                self.src = &self.src[split..];
                return Some(match keyword {
                    "#import" => Token::Import,
                    "#mixin" => Token::Mixin,
                    "#if" => Token::StaticIf,
                    "#else" => Token::StaticElse,
                    _ => Token::Basic(keyword),
                });
            }
            (Some('{'), _) => {
                self.src = &self.src[1..];
                return Some(Token::OpenBracket);
            }
            (Some('}'), _) => {
                self.src = &self.src[1..];
                return Some(Token::CloseBracket);
            }
            (Some('\n'), _) => {
                self.src = &self.src[1..];
                self.line += 1;
                return Some(Token::NewLine);
            }
            (Some(c), _) if c.is_whitespace() => {
                let split = self.src
                    .find(|c: char| !c.is_whitespace() && c != '\n')
                    .unwrap_or(self.src.len());
                let space = &self.src[..split];
                self.src = &self.src[split..];
                return Some(Token::WhiteSpace(space));
            }
            (Some(_), _) => {
                let split = self.src[1..]
                    .find(|c: char| !c.is_alphanumeric())
                    .map(|offset| offset + 1)
                    .unwrap_or(self.src.len());
                let text = &self.src[..split];
                self.src = &self.src[split..];
                assert!(!text.is_empty());
                return Some(Token::Basic(text));
            }
            (None, _) => {}
        }

        None
    }
}

#[test]
fn tokenizer() {
    let mut tokenizer = Tokenizer::new(
        &"main",
        &"fn main(a: u32) {
            // a comment
            // another comment
            #import foo
        }
        "
    );

    while let Some(tok) = tokenizer.current() {
        println!(" - {:?}", tok);
        tokenizer.advance();
    }
}

#[test]
fn static_if() {
    let mut lib = HashMap::new();
    let mut preprocessor = Preprocessor::new();
    lib.insert("src".into(), "
fn bar() {
    #if FOO {
        var i: i32 = 0
        loop {
            if i > 10 {
                break;
            }
            // Comment }}} {{
            foo is defined
        }
    } #else {
        {
            {

            }
            {} {{}}
            foo is not defined
        }
    }
    regular code
    a = #if BAZ { baz is defined } #else { baz is not defined };
}".into()
    );

    let output = preprocessor.preprocess("src", &lib).unwrap();
    println!("{}", output);
    assert_eq!(output.matches("foo is defined").count(), 0);
    assert_eq!(output.matches("baz is defined").count(), 0);
    assert_eq!(output.matches("foo is not defined").count(), 1);
    assert_eq!(output.matches("baz is not defined").count(), 1);
    assert_eq!(output.matches("regular code").count(), 1);

    preprocessor.define("FOO");
    preprocessor.define("BAZ");
    let output = preprocessor.preprocess("src", &lib).unwrap();
    println!("----\n{}", output);
    assert_eq!(output.matches("foo is defined").count(), 1);
    assert_eq!(output.matches("baz is defined").count(), 1);
    assert_eq!(output.matches("foo is not defined").count(), 0);
    assert_eq!(output.matches("baz is not defined").count(), 0);
    assert_eq!(output.matches("regular code").count(), 1);
}

#[test]
fn import() {
    let mut lib: HashMap<String, Source> = HashMap::new();
    let mut loader = Preprocessor::new();

    lib.insert(
        "A".into(),
"Hello(A),
#import B
#import B
#import A
end of A".into()
    );
    lib.insert(
        "B".into(),
"Hello(B),
#import A
#import B
end of B".into()
    );

    let output = loader.preprocess("A", &lib).unwrap();

    print_source(&output);

    assert_eq!(output.matches("Hello(A)").count(), 1);
    assert_eq!(output.matches("end of A").count(), 1);
    assert_eq!(output.matches("Hello(B)").count(), 1);
    assert_eq!(output.matches("end of B").count(), 1);
}
