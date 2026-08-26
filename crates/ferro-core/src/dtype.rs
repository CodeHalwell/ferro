//! Element dtypes for tensor storage. Float math and autograd are f32-only;
//! F64/I64 tensors carry data (indices, targets, high-precision buffers) and
//! F16/BF16 tensors carry half-precision weights (the checkpoint formats of
//! milestone M3), all entering compute via explicit `to_dtype(DType::F32)`.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F64,
    I64,
    /// IEEE binary16; stored as raw bits, materialized through `half`.
    F16,
    /// bfloat16 (f32's top half); stored as raw bits.
    BF16,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::I64 => "i64",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
        })
    }
}
