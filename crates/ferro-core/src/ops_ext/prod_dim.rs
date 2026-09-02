//! `prod_dim`: product reduction over one dimension, with keepdim. No
//! raw_prod_dim kernel exists, so forward loops over host data using the
//! outer/size/stride split (as in cumsum/logsumexp). Backward:
//! d(prod)/dx_i = prod / x_i, so dx_i = g * prod / x_i; this is only defined
//! for non-zero x_i, so callers (and the grad_check below) must keep inputs
//! strictly non-zero.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn prod_dim(&self, dim: usize, keepdim: bool) -> Result<Tensor> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape {
                op: "prod_dim",
                msg: format!("dim {dim} out of range for rank {ndim}"),
            });
        }
        let shape = self.shape().to_vec();
        let size = shape[dim];
        let stride: usize = shape[dim + 1..].iter().product();
        let outer: usize = shape[..dim].iter().product();
        let mut out_shape = shape.clone();
        if keepdim {
            out_shape[dim] = 1;
        } else {
            out_shape.remove(dim);
        }

        let x = self.to_vec();
        let mut prod = vec![0.0f32; outer * stride];
        for o in 0..outer {
            for i in 0..stride {
                let base = o * size * stride + i;
                let mut p = 1.0f32;
                for k in 0..size {
                    p *= x[base + k * stride];
                }
                prod[o * stride + i] = p;
            }
        }
        let out = Tensor::from_vec(prod.clone(), &out_shape)?;
        if !self.requires_grad() {
            return Ok(out);
        }
        Ok(out.record_fn(vec![self.clone()], move |g| {
            let gd = g.to_vec();
            let mut dx = vec![0.0f32; x.len()];
            for o in 0..outer {
                for i in 0..stride {
                    let base = o * size * stride + i;
                    let lg = gd[o * stride + i];
                    let p = prod[o * stride + i];
                    for k in 0..size {
                        let idx = base + k * stride;
                        dx[idx] = lg * p / x[idx];
                    }
                }
            }
            vec![Tensor::from_vec(dx, &shape).unwrap()]
        }))
    }
}
