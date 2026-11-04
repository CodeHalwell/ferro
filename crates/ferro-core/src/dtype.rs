//! Element dtypes for tensor storage. F32 uses backend kernels; CPU F64 has
//! native basic math and first-order autograd. I64 arithmetic and reductions
//! retain exact integers. Bool is nondifferentiable; F16/BF16 are storage-only
//! and enter compute through an explicit floating-point cast.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F64,
    I64,
    /// Boolean values, stored as one byte per element on the CPU.
    Bool,
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
            DType::Bool => "bool",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
        })
    }
}
