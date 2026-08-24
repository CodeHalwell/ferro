//! `argmax`/`argmin` along one dimension, returning an I64 index tensor.
//! Torch semantics: NaN wins (the first NaN along the slice), remaining ties
//! break toward the lowest index. Index outputs are not differentiable, so
//! nothing is recorded.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn argmax(&self, dim: usize, keepdim: bool) -> Result<Tensor> {
        arg_reduce(self, "argmax", dim, keepdim, |cand, best| cand > best)
    }

    pub fn argmin(&self, dim: usize, keepdim: bool) -> Result<Tensor> {
        arg_reduce(self, "argmin", dim, keepdim, |cand, best| cand < best)
    }
}

fn arg_reduce(
    t: &Tensor,
    op: &'static str,
    dim: usize,
    keepdim: bool,
    better: impl Fn(f32, f32) -> bool,
) -> Result<Tensor> {
    let ndim = t.ndim();
    if dim >= ndim {
        return Err(Error::InvalidShape {
            op,
            msg: format!("dim {dim} out of range for rank {ndim}"),
        });
    }
    let shape = t.shape().to_vec();
    let size = shape[dim];
    if size == 0 {
        return Err(Error::InvalidShape {
            op,
            msg: "cannot reduce an empty dimension".into(),
        });
    }
    let stride: usize = shape[dim + 1..].iter().product();
    let outer: usize = shape[..dim].iter().product();

    let x = t.to_vec();
    let mut idx = Vec::with_capacity(outer * stride);
    for o in 0..outer {
        for i in 0..stride {
            let base = o * size * stride + i;
            let mut arg = 0usize;
            for k in 0..size {
                let v = x[base + k * stride];
                if v.is_nan() {
                    arg = k;
                    break;
                }
                if better(v, x[base + arg * stride]) {
                    arg = k;
                }
            }
            idx.push(arg as i64);
        }
    }

    let mut out_shape = shape;
    if keepdim {
        out_shape[dim] = 1;
    } else {
        out_shape.remove(dim);
    }
    Tensor::from_vec_i64(idx, &out_shape)
}
