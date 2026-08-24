//! Autocast scaffold: f32 master weights with bf16 compute casts. True bf16
//! element storage would touch Storage/dtype/to_vec across the whole core and
//! every backend buffer format, so this module implements the f32-master-
//! weights pattern entirely here: tensors stay f32 at rest, and autocast ops
//! round-trip inputs through bf16 precision before compute. Accumulators and
//! outputs return to f32, matching torch's "matmuls in bf16, reductions in
//! fp32" promotion policy.
//!
//! Promotion rules mirrored from torch.autocast (dtype table):
//!   matmul/bmm/linear        -> bf16 compute
//!   pointwise add/mul/relu.. -> f32 (inputs are cast back if already bf16)
//!   softmax/log_softmax/norm -> f32 (reductions stay fp32)
//!   pow/div with scalar      -> f32
//!
//! bf16 quantization is round-to-nearest-even on the upper 16 bits: 1 sign,
//! 8 exponent, 7 mantissa. It is a step function, so its true derivative is
//! zero almost everywhere; backward uses the straight-through estimator
//! (d round/dx := 1), which is exactly matmul's chain rule on the rounded
//! copies. The smooth `quantized_matmul` op carries the grad_checkable
//! backward; `amp_matmul` bridges it to the f32 masters.

use crate::error::{Error, Result};
use crate::tensor::{raw_matmul, raw_matmul_t, Tensor};

/// Round x to bf16 precision, returning the nearest representable f32.
pub fn bf16_round(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let bits = x.to_bits();
    let lsb = 1u32 << 16;
    let rounded = bits.wrapping_add(lsb >> 1) & 0xFFFF_0000;
    // Crossing into the sign bit means |x| exceeded bf16 max (~3.39e38);
    // saturate to infinity like a real bf16 conversion.
    let out =
        if rounded & 0x7FFF_FFFF == 0 && (bits & 0x7FFF_FFFF) != 0 && rounded & 0x8000_0000 != 0 {
            f32::INFINITY.to_bits()
        } else {
            rounded
        };
    f32::from_bits(out)
}

impl Tensor {
    /// Quantize to bf16 precision (still stored as f32). Detached leaf; this
    /// is the "cast" side of the autocast pattern and carries no autograd.
    pub fn cast_to_bf16(&self) -> Result<Tensor> {
        if self.dtype() != crate::DType::F32 {
            return Err(Error::DtypeMismatch {
                op: "cast_to_bf16",
                expected: crate::DType::F32,
                got: self.dtype(),
            });
        }
        let v = self.to_vec();
        let y: Vec<f32> = v.iter().map(|&x| bf16_round(x)).collect();
        Tensor::from_vec(y, self.shape())
    }

    /// Master-weights pattern: outputs of autocast compute are already f32, so
    /// casting back is the identity on data. Kept explicit so call sites read
    /// as cast -> compute -> cast_back and a future true-bf16 storage lands
    /// without changing callers.
    pub fn cast_back(&self) -> Result<Tensor> {
        if self.dtype() != crate::DType::F32 {
            return Err(Error::DtypeMismatch {
                op: "cast_back",
                expected: crate::DType::F32,
                got: self.dtype(),
            });
        }
        Ok(self.detach_copy())
    }
}

/// Which side of the torch autocast dtype table an op falls in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpClass {
    /// Matmul-like ops: run in reduced precision.
    Matmul,
    /// Pointwise and reduction ops: stay fp32.
    Fp32,
}

/// Autocast context. Mirrors torch.autocast(enabled=...): inside the context,
/// `enter` casts operands per the promotion table above; outside or disabled,
/// it passes tensors through untouched.
pub struct Autocast {
    pub enabled: bool,
}

impl Default for Autocast {
    fn default() -> Self {
        Autocast { enabled: true }
    }
}

impl Autocast {
    pub fn new() -> Autocast {
        Autocast::default()
    }

    /// Cast inputs for `class` per the promotion table: Matmul goes through
    /// bf16 quantization, Fp32 returns clones unchanged.
    pub fn enter(&self, class: OpClass, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        match (self.enabled, class) {
            (true, OpClass::Matmul) => inputs.iter().map(|t| t.cast_to_bf16()).collect(),
            _ => Ok(inputs.iter().map(|t| (*t).clone()).collect()),
        }
    }
}

/// Matmul over already-quantized operands: bf16 values held in f32 storage,
/// f32 products and accumulation. Backward is the exact matmul chain rule on
/// those operands (the straight-through estimator), so this op is smooth and
/// grad_checkable.
pub fn quantized_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(Error::Unsupported {
            op: "quantized_matmul",
            msg: "only 2-D supported".into(),
        });
    }
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let n = b.shape()[1];
    if b.shape()[0] != k {
        return Err(Error::ShapeMismatch {
            op: "quantized_matmul",
            lhs: a.shape().to_vec(),
            rhs: b.shape().to_vec(),
        });
    }
    let out = raw_matmul(a, b)?;
    let (aa, bb) = (a.clone(), b.clone());
    Ok(out.record_fn(vec![a.clone(), b.clone()], move |g| {
        vec![
            raw_matmul_t(&g, &bb, false, true).unwrap(),
            raw_matmul_t(&aa, &g, true, false).unwrap(),
        ]
    }))
}

/// Reference autocast op: cast both f32 masters to bf16, multiply with f32
/// accumulation, return an f32 master output. Gradients reach the masters via
/// the straight-through estimator - identical formulas to quantized_matmul's
/// backward evaluated at the rounded copies.
pub fn amp_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(Error::Unsupported {
            op: "amp_matmul",
            msg: "only 2-D supported".into(),
        });
    }
    if b.shape()[0] != a.shape()[1] {
        return Err(Error::ShapeMismatch {
            op: "amp_matmul",
            lhs: a.shape().to_vec(),
            rhs: b.shape().to_vec(),
        });
    }
    let ac = Autocast::new().enter(OpClass::Matmul, &[a, b])?;
    let qa = ac[0].detach_copy();
    let qb = ac[1].detach_copy();
    let out = quantized_matmul(&ac[0], &ac[1])?;
    Ok(out.record_fn(vec![a.clone(), b.clone()], move |g| {
        vec![
            raw_matmul_t(&g, &qb, false, true).unwrap(),
            raw_matmul_t(&qa, &g, true, false).unwrap(),
        ]
    }))
}
