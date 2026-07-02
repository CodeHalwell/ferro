//! Numerically stable softmax over one dimension. Forward computes
//! `y = exp(x - max) / sum(exp(x - max))` per 1-D slice along `dim`.
//! Backward is the softmax Jacobian-vector product:
//! `dx_i = y_i * (g_i - sum_k g_k * y_k)`.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn softmax(&self, dim: usize) -> Result<Tensor> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape { op: "softmax", msg: format!("dim {dim} out of range for rank {ndim}") });
        }
        let shape = self.shape().to_vec();
        let x = self.to_vec();
        let y_data = softmax_forward(&x, &shape, dim);
        let out = Tensor::from_vec(y_data, &shape).unwrap();
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
            let mut sum = 0.0f32;
            for k in 0..size {
                let e = (x[base + k * stride] - m).exp();
                y[base + k * stride] = e;
                sum += e;
            }
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
    for o in 0..outer {
        for i in 0..stride {
            let base = o * size * stride + i;
            let mut dot = 0.0f32;
            for k in 0..size {
                dot += g[base + k * stride] * y[base + k * stride];
            }
            for k in 0..size {
                let idx = base + k * stride;
                dx[idx] = y[idx] * (g[idx] - dot);
            }
        }
    }
    dx
}
