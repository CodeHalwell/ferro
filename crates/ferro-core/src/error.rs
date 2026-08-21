use std::fmt;

use crate::device::Device;
use crate::dtype::DType;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Two shapes could not be broadcast/aligned for an op.
    ShapeMismatch { op: &'static str, lhs: Vec<usize>, rhs: Vec<usize> },
    /// A shape argument was invalid for the requested op (e.g. bad reshape).
    InvalidShape { op: &'static str, msg: String },
    /// An op does not yet support the given rank/config in this MVP.
    Unsupported { op: &'static str, msg: String },
    /// Operands live on different devices.
    DeviceMismatch { op: &'static str, lhs: Device, rhs: Device },
    /// An operand has the wrong dtype for the op (float math is f32-only).
    DtypeMismatch { op: &'static str, expected: DType, got: DType },
    /// A filesystem read/write failed (path and OS error in the message).
    Io { op: &'static str, msg: String },
    /// A serialized file is malformed or uses an unsupported encoding.
    Format { op: &'static str, msg: String },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ShapeMismatch { op, lhs, rhs } => {
                write!(f, "{op}: incompatible shapes {lhs:?} and {rhs:?}")
            }
            Error::InvalidShape { op, msg } => write!(f, "{op}: invalid shape: {msg}"),
            Error::Unsupported { op, msg } => write!(f, "{op}: unsupported: {msg}"),
            Error::DeviceMismatch { op, lhs, rhs } => {
                write!(f, "{op}: operands on different devices ({lhs} vs {rhs})")
            }
            Error::DtypeMismatch { op, expected, got } => {
                write!(f, "{op}: expected {expected} tensor, got {got}")
            }
            Error::Io { op, msg } => write!(f, "{op}: io error: {msg}"),
            Error::Format { op, msg } => write!(f, "{op}: malformed file: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
