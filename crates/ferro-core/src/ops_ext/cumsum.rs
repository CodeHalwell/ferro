//! Cumulative sum along one dimension (inclusive prefix sum, torch semantics).
//! Backward is the reversed cumulative sum of the incoming grad along the same
//! dim: dL/dx_i = sum_{j >= i} g_j, because x_i contributes to every y_j with
//! j >= i.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn cumsum(&self, dim: usize) -> Result<Tensor> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape {
                op: "cumsum",
                msg: format!("dim {dim} out of range for rank {ndim}"),
            });
        }
        let shape = self.shape().to_vec();
        let size = shape[dim];
        let stride: usize = shape[dim + 1..].iter().product();
        let outer: usize = shape[..dim].iter().product();

        let x = self.to_vec();
        let mut y = vec![0.0f32; x.len()];
        for o in 0..outer {
            for i in 0..stride {
                let base = o * size * stride + i;
                let mut acc = 0.0f32;
                for k in 0..size {
                    acc += x[base + k * stride];
                    y[base + k * stride] = acc;
                }
            }
        }
        let out = Tensor::from_vec(y, &shape)?;
        if !self.requires_grad() {
            return Ok(out);
        }
        Ok(out.record_fn(vec![self.clone()], move |g| {
            let gd = g.to_vec();
            let mut dx = vec![0.0f32; gd.len()];
            for o in 0..outer {
                for i in 0..stride {
                    let base = o * size * stride + i;
                    let mut acc = 0.0f32;
                    for k in (0..size).rev() {
                        acc += gd[base + k * stride];
                        dx[base + k * stride] = acc;
                    }
                }
            }
            vec![Tensor::from_vec(dx, &shape).unwrap()]
        }))
    }
}
