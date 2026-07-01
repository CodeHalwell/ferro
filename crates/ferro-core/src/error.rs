use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Two shapes could not be broadcast/aligned for an op.
    ShapeMismatch { op: &'static str, lhs: Vec<usize>, rhs: Vec<usize> },
    /// A shape argument was invalid for the requested op (e.g. bad reshape).
    InvalidShape { op: &'static str, msg: String },
    /// An op does not yet support the given rank/config in this MVP.
    Unsupported { op: &'static str, msg: String },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ShapeMismatch { op, lhs, rhs } => {
                write!(f, "{op}: incompatible shapes {lhs:?} and {rhs:?}")
            }
            Error::InvalidShape { op, msg } => write!(f, "{op}: invalid shape: {msg}"),
            Error::Unsupported { op, msg } => write!(f, "{op}: unsupported: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
