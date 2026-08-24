//! Numerically stable softmax over one dimension. Forward computes
//! `y = exp(x - max) / sum(exp(x - max))` per 1-D slice along `dim`.
//! Backward is the softmax Jacobian-vector product:
//! `dx_i = y_i * (g_i - sum_k g_k * y_k)`.
//!
//! Device-resident whole buffers take the backend's `softmax_dev` row kernel
//! (last-dim only); the backward is then composed from resident tensor ops
//! (mul/sub/sum_dim) so no host round trip happens in either direction.

use crate::error::{Error, Result};
use crate::reduce::pairwise_sum_strided;
use crate::tensor::{raw_row_softmax, Tensor};

impl Tensor {
    pub fn softmax(&self, dim: usize) -> Result<Tensor> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape {
                op: "softmax",
                msg: format!("dim {dim} out of range for rank {ndim}"),
            });
        }
        let shape = self.shape().to_vec();
        if let Some(out) = raw_row_softmax(self, dim, false) {
            if !self.requires_grad() {
                return Ok(out);
            }
            let y = out.detach_copy();
            return Ok(out.record_fn(vec![self.clone()], move |g| {
                // dx = y * (g - sum(g*y, dim)) with keepdim sum broadcasting.
                let gy = g.mul(&y).unwrap();
                let s = gy.sum_dim(dim, true).unwrap();
                vec![g.sub(&s).unwrap().mul(&y).unwrap()]
            }));
        }
        let x = self.to_vec();
        let y_data = softmax_forward(&x, &shape, dim);
        // Host-composed op: return to the input's device so chained
        // device-resident ops stay on-device.
        let out = Tensor::from_vec(y_data, &shape)?.to_device(self.device())?;
        if !self.requires_grad() {
            return Ok(out);
        }
        let y = out.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            let dx = softmax_backward(&g.to_vec(), &y.to_vec(), &shape, dim);
            vec![Tensor::from_vec(dx, &shape).unwrap()]
        }))
    }
}

// Strides for iterating one slice along `dim`: `n` slices in the outer block,
// `stride` contiguous elements in the inner block, `size` steps along `dim`.
fn slice_dims(shape: &[usize], dim: usize) -> (usize, usize, usize) {
    let size = shape[dim];
    let stride: usize = shape[dim + 1..].iter().product();
    let outer: usize = shape[..dim].iter().product();
    (outer, size, stride)
}

fn softmax_forward(x: &[f32], shape: &[usize], dim: usize) -> Vec<f32> {
    let (outer, size, stride) = slice_dims(shape, dim);
    let mut y = vec![0.0f32; x.len()];
    for o in 0..outer {
        for i in 0..stride {
            let base = o * size * stride + i;
            let mut m = f32::NEG_INFINITY;
            for k in 0..size {
                m = m.max(x[base + k * stride]);
            }
            for k in 0..size {
                y[base + k * stride] = (x[base + k * stride] - m).exp();
            }
            let sum = pairwise_sum_strided(&y, base, size, stride);
            for k in 0..size {
                y[base + k * stride] /= sum;
            }
        }
    }
    y
}

fn softmax_backward(g: &[f32], y: &[f32], shape: &[usize], dim: usize) -> Vec<f32> {
    let (outer, size, stride) = slice_dims(shape, dim);
    let mut dx = vec![0.0f32; g.len()];
    let mut gy = vec![0.0f32; g.len()];
    for o in 0..outer {
        for i in 0..stride {
            let base = o * size * stride + i;
            for k in 0..size {
                let idx = base + k * stride;
                gy[idx] = g[idx] * y[idx];
            }
            let dot = pairwise_sum_strided(&gy, base, size, stride);
            for k in 0..size {
                let idx = base + k * stride;
                dx[idx] = y[idx] * (g[idx] - dot);
            }
        }
    }
    dx
}
