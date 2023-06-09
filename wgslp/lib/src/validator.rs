use naga::front::wgsl;

pub type ValidationError = naga::WithSpan<naga::valid::ValidationError>;

pub enum Error {
    Parser(wgsl::ParseError),
    Validation(ValidationError),
}

impl Error {
    pub fn emit_to_stderr(&self, source: &str) {
        match self {
            Error::Parser(error) => {
                error.emit_to_stderr(source);
            }
            Error::Validation(error) => {
                eprintln!("{}", error.to_string());
            }
        }
    }
}

pub struct  Validator {
    parser: wgsl::Frontend,
    validator: naga::valid::Validator,
}

impl Validator {
    pub fn new() -> Self {
        Validator {
            parser: wgsl::Frontend::new(),
            validator: naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            ),
        }
    }

    pub fn validate(&mut self, shader_source: &str) -> Result<(), Error> {
        let module = match self.parser.parse(&shader_source) {
            Ok(module) => module,
            Err(error) => {
                return Err(Error::Parser(error));
            }
        };

        match self.validator.validate(&module) {
            Ok(_) => {}
            Err(error) => {
                return Err(Error::Validation(error));
            }
        }
    
        Ok(())
    }
}
