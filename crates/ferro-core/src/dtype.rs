//! Element dtypes for tensor storage. Float math and autograd are f32-only;
//! F64/I64 tensors carry data (indices, targets, high-precision buffers) and
//! enter compute via explicit `to_dtype(DType::F32)`.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F64,
    I64,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::I64 => "i64",
        })
    }
}
