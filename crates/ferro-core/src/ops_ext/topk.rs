//! `topk`: the k largest values along one dimension, sorted descending, plus
//! their I64 indices (torch semantics; ties break toward the lower index, NaN
//! sorts above everything like torch's max). Values are differentiable -
//! backward scatter-adds each value grad to its source position; indices are
//! a detached I64 tensor.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn topk(&self, k: usize, dim: usize) -> Result<(Tensor, Tensor)> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape {
                op: "topk",
                msg: format!("dim {dim} out of range for rank {ndim}"),
            });
        }
        let shape = self.shape().to_vec();
        let size = shape[dim];
        if k > size {
            return Err(Error::InvalidShape {
                op: "topk",
                msg: format!("k {k} out of range for dim {dim} with size {size}"),
            });
        }
        let stride: usize = shape[dim + 1..].iter().product();
        let outer: usize = shape[..dim].iter().product();

        let x = self.to_vec();
        let mut vals = Vec::with_capacity(outer * k * stride);
        let mut idxs = Vec::with_capacity(outer * k * stride);
        let mut order: Vec<usize> = Vec::with_capacity(size);
        // Flat offsets into the input for each selected value, in output order,
        // captured for the backward scatter.
        let mut src = Vec::with_capacity(outer * k * stride);
        for o in 0..outer {
            for i in 0..stride {
                let base = o * size * stride + i;
                order.clear();
                order.extend(0..size);
                order.sort_by(|&a, &b| {
                    let (va, vb) = (x[base + a * stride], x[base + b * stride]);
                    match (va.is_nan(), vb.is_nan()) {
                        (true, false) => std::cmp::Ordering::Less,
                        (false, true) => std::cmp::Ordering::Greater,
                        _ => vb
                            .partial_cmp(&va)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then(a.cmp(&b)),
                    }
                });
                for &j in &order[..k] {
                    vals.push(x[base + j * stride]);
                    idxs.push(j as i64);
                    src.push(base + j * stride);
                }
            }
        }

        // Selected entries are laid out [outer, stride, k] by the loop above;
        // permute to the output layout [outer, k, stride].
        let mut out_shape = shape.clone();
        out_shape[dim] = k;
        let n = outer * k * stride;
        let mut v_out = vec![0.0f32; n];
        let mut i_out = vec![0i64; n];
        let mut s_out = vec![0usize; n];
        for o in 0..outer {
            for i in 0..stride {
                for j in 0..k {
                    let from = (o * stride + i) * k + j;
                    let to = o * k * stride + j * stride + i;
                    v_out[to] = vals[from];
                    i_out[to] = idxs[from];
                    s_out[to] = src[from];
                }
            }
        }

        let indices = Tensor::from_vec_i64(i_out, &out_shape)?;
        let values = Tensor::from_vec(v_out, &out_shape)?;
        if !self.requires_grad() {
            return Ok((values, indices));
        }
        let in_shape = shape;
        let values = values.record_fn(vec![self.clone()], move |g| {
            let gd = g.to_vec();
            let mut gx = vec![0.0f32; in_shape.iter().product()];
            for (i, &o) in s_out.iter().enumerate() {
                gx[o] += gd[i];
            }
            vec![Tensor::from_vec(gx, &in_shape).unwrap()]
        });
        Ok((values, indices))
    }
}
