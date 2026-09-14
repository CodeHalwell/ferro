//! `cat` concatenation along a dimension. Row-major layout: for each `outer`
//! index (prod of dims before `dim`), each input contributes a contiguous
//! block of `shape[dim] * inner` elements (inner = prod of dims after `dim`).

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn cat(tensors: &[Tensor], dim: usize) -> Result<Tensor> {
        let first = tensors.first().ok_or_else(|| Error::InvalidShape {
            op: "cat",
            msg: "expected a non-empty list of tensors".into(),
        })?;
        for t in tensors {
            if t.dtype() != first.dtype() {
                return Err(Error::DtypeMismatch {
                    op: "cat",
                    expected: first.dtype(),
                    got: t.dtype(),
                });
            }
        }
        for t in &tensors[1..] {
            if t.device() != first.device() {
                return Err(Error::DeviceMismatch {
                    op: "cat",
                    lhs: first.device(),
                    rhs: t.device(),
                });
            }
        }
        let base = first.shape().to_vec();
        if dim >= base.len() {
            return Err(Error::InvalidShape {
                op: "cat",
                msg: format!("dim {dim} out of range for rank {}", base.len()),
            });
        }
        for t in &tensors[1..] {
            let s = t.shape();
            let same_rank = s.len() == base.len();
            let ok = same_rank
                && s.iter()
                    .zip(&base)
                    .enumerate()
                    .all(|(d, (a, b))| d == dim || a == b);
            if !ok {
                return Err(Error::ShapeMismatch {
                    op: "cat",
                    lhs: base.clone(),
                    rhs: s.to_vec(),
                });
            }
        }

        let outer: usize = base[..dim].iter().product();
        let inner: usize = base[dim + 1..].iter().product();
        let dim_sizes: Vec<usize> = tensors.iter().map(|t| t.shape()[dim]).collect();
        let cat_size = dim_sizes
            .iter()
            .try_fold(0usize, |n, &d| n.checked_add(d))
            .ok_or_else(|| Error::InvalidShape {
                op: "cat",
                msg: "axis size overflow".into(),
            })?;

        let mut out_shape = base;
        out_shape[dim] = cat_size;
        crate::shape::checked_numel("cat", &out_shape)?;
        let len = outer
            .checked_mul(cat_size)
            .and_then(|n| n.checked_mul(inner))
            .ok_or_else(|| Error::InvalidShape {
                op: "cat",
                msg: "output size overflow".into(),
            })?;
        // No device cat kernel: typed host materialization returns CPU storage.
        macro_rules! concat {
            ($sources:expr, $ctor:ident) => {{
                let sources: Vec<_> = $sources;
                let mut dst = Vec::with_capacity(len);
                if len != 0 {
                    for o in 0..outer {
                        for (src, &d) in sources.iter().zip(&dim_sizes) {
                            let block = d * inner;
                            dst.extend_from_slice(&src[o * block..(o + 1) * block]);
                        }
                    }
                }
                Tensor::$ctor(dst, &out_shape)?
            }};
        }
        let out = match first.dtype() {
            DType::F32 => concat!(tensors.iter().map(Tensor::to_vec).collect(), from_vec),
            DType::F64 => concat!(
                tensors.iter().map(Tensor::to_vec_f64).collect(),
                from_vec_f64
            ),
            DType::I64 => concat!(
                tensors.iter().map(Tensor::to_vec_i64).collect(),
                from_vec_i64
            ),
            DType::F16 => concat!(
                tensors
                    .iter()
                    .map(Tensor::to_vec_f16_bits)
                    .collect::<Result<Vec<_>>>()?,
                from_vec_f16_bits
            ),
            DType::BF16 => concat!(
                tensors
                    .iter()
                    .map(Tensor::to_vec_bf16_bits)
                    .collect::<Result<Vec<_>>>()?,
                from_vec_bf16_bits
            ),
        };
        if first.dtype() != DType::F32 {
            return Ok(out);
        }

        let in_shapes: Vec<Vec<usize>> = tensors.iter().map(|t| t.shape().to_vec()).collect();
        Ok(out.record_fn(tensors.to_vec(), move |g| {
            let g_data = g.to_vec();
            let mut grads = Vec::with_capacity(in_shapes.len());
            let mut offset = 0usize;
            for (shape, &d) in in_shapes.iter().zip(&dim_sizes) {
                let block = d * inner;
                let mut gi = vec![0.0f32; outer * block];
                for o in 0..outer {
                    let src = o * cat_size * inner + offset * inner;
                    gi[o * block..(o + 1) * block].copy_from_slice(&g_data[src..src + block]);
                }
                grads.push(Tensor::from_vec(gi, shape).unwrap());
                offset += d;
            }
            grads
        }))
    }
}
