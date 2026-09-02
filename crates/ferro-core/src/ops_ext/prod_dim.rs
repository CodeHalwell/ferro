//! `prod_dim`: product reduction over one dimension, with keepdim. No
//! raw_prod_dim kernel exists, so forward loops over host data using the
//! outer/size/stride split (as in cumsum/logsumexp). Backward:
//! d(prod)/dx_i is the product of every OTHER element in the reduced slice,
//! not prod/x_i (which is 0/0 when x_i is itself zero). Per slice: with no
//! zeros, prod/x_i is equivalent and used directly; with exactly one zero,
//! only that position gets a nonzero gradient (the product of the rest);
//! with two or more zeros, every gradient in the slice is 0 (removing any
//! single element still leaves a zero factor behind).

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
        // Per slice: the full product (fast path when no zeros) and the
        // product of only the nonzero elements plus how many zeros there
        // were (used to get the zero-containing case right in backward).
        let mut nonzero_prod = vec![0.0f32; outer * stride];
        let mut zero_count = vec![0u32; outer * stride];
        for o in 0..outer {
            for i in 0..stride {
                let base = o * size * stride + i;
                let mut p = 1.0f32;
                let mut np = 1.0f32;
                let mut zc = 0u32;
                for k in 0..size {
                    let v = x[base + k * stride];
                    p *= v;
                    if v == 0.0 {
                        zc += 1;
                    } else {
                        np *= v;
                    }
                }
                prod[o * stride + i] = p;
                nonzero_prod[o * stride + i] = np;
                zero_count[o * stride + i] = zc;
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
                    let np = nonzero_prod[o * stride + i];
                    let zc = zero_count[o * stride + i];
                    for k in 0..size {
                        let idx = base + k * stride;
                        let xi = x[idx];
                        dx[idx] = if xi == 0.0 {
                            if zc == 1 {
                                lg * np
                            } else {
                                0.0
                            }
                        } else if zc == 0 {
                            lg * p / xi
                        } else {
                            0.0
                        };
                    }
                }
            }
            vec![Tensor::from_vec(dx, &shape).unwrap()]
        }))
    }
}
