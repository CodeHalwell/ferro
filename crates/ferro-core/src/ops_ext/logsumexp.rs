//! `logsumexp` along one dimension: numerically stable
//! lse = max + log(sum(exp(x - max))) per 1-D slice, keepdim=false.
//! Backward: dx_i = g_i * exp(x_i - lse), i.e. the softmax of the slice.

use crate::error::{Error, Result};
use crate::reduce::pairwise_sum_strided;
use crate::tensor::Tensor;

impl Tensor {
    pub fn logsumexp(&self, dim: usize) -> Result<Tensor> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape {
                op: "logsumexp",
                msg: format!("dim {dim} out of range for rank {ndim}"),
            });
        }
        let shape = self.shape().to_vec();
        let size = shape[dim];
        let stride: usize = shape[dim + 1..].iter().product();
        let outer: usize = shape[..dim].iter().product();
        let mut out_shape = shape.clone();
        out_shape.remove(dim);
        let out_numel: usize = outer * stride;

        let x = self.to_vec();
        let mut lse = vec![0.0f32; out_numel];
        for o in 0..outer {
            for i in 0..stride {
                let base = o * size * stride + i;
                let mut mx = f32::NEG_INFINITY;
                for k in 0..size {
                    mx = mx.max(x[base + k * stride]);
                }
                let mut acc = Vec::with_capacity(size);
                for k in 0..size {
                    acc.push((x[base + k * stride] - mx).exp());
                }
                lse[o * stride + i] = mx + pairwise_sum_strided(&acc, 0, size, 1).ln();
            }
        }
        let out = Tensor::from_vec(lse.clone(), &out_shape)?;
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
                    let lv = lse[o * stride + i];
                    for k in 0..size {
                        let idx = base + k * stride;
                        dx[idx] = lg * (x[idx] - lv).exp();
                    }
                }
            }
            vec![Tensor::from_vec(dx, &shape).unwrap()]
        }))
    }
}
