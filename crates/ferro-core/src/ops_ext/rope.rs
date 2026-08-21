//! Rotary position embedding (RoPE), half-split convention (LLaMA/HF): the
//! last dim is split into halves (x1, x2) and each pair (x1[j], x2[j]) is
//! rotated by pos * base^(-2j/d):
//!   y1 = x1*cos - x2*sin,  y2 = x2*cos + x1*sin.
//! Input is [..., seq, head_dim] with head_dim even; `positions` is a 1-D I64
//! tensor of length seq (explicit so KV-cache decode can offset positions).
//! The rotation is orthogonal and linear, so backward applies the inverse
//! rotation (negated sin) to the incoming grad.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn rope(&self, positions: &Tensor, base: f32) -> Result<Tensor> {
        let ndim = self.ndim();
        if ndim < 2 {
            return Err(Error::InvalidShape { op: "rope", msg: format!("input must be at least 2-D [seq, head_dim], got {:?}", self.shape()) });
        }
        let shape = self.shape().to_vec();
        let (seq, dim) = (shape[ndim - 2], shape[ndim - 1]);
        if dim % 2 != 0 {
            return Err(Error::InvalidShape { op: "rope", msg: format!("head_dim {dim} must be even") });
        }
        if positions.dtype() != DType::I64 {
            return Err(Error::DtypeMismatch { op: "rope", expected: DType::I64, got: positions.dtype() });
        }
        if positions.ndim() != 1 || positions.shape()[0] != seq {
            return Err(Error::InvalidShape {
                op: "rope",
                msg: format!("positions must be 1-D of length {seq}, got shape {:?}", positions.shape()),
            });
        }

        let pos = positions.to_vec_i64();
        let half = dim / 2;
        let mut cos = vec![0.0f32; seq * half];
        let mut sin = vec![0.0f32; seq * half];
        for s in 0..seq {
            for j in 0..half {
                let theta = pos[s] as f32 * base.powf(-2.0 * j as f32 / dim as f32);
                cos[s * half + j] = theta.cos();
                sin[s * half + j] = theta.sin();
            }
        }

        let batch: usize = shape[..ndim - 2].iter().product();
        let x = self.to_vec();
        let y = rotate(&x, &cos, &sin, batch, seq, half, 1.0);
        let out = Tensor::from_vec(y, &shape)?;
        if !self.requires_grad() {
            return Ok(out);
        }
        Ok(out.record_fn(vec![self.clone()], move |g| {
            let dx = rotate(&g.to_vec(), &cos, &sin, batch, seq, half, -1.0);
            vec![Tensor::from_vec(dx, &shape).unwrap()]
        }))
    }
}

fn rotate(x: &[f32], cos: &[f32], sin: &[f32], batch: usize, seq: usize, half: usize, sign: f32) -> Vec<f32> {
    let dim = 2 * half;
    let mut y = vec![0.0f32; x.len()];
    for b in 0..batch {
        for s in 0..seq {
            let row = (b * seq + s) * dim;
            for j in 0..half {
                let (c, sn) = (cos[s * half + j], sign * sin[s * half + j]);
                let (x1, x2) = (x[row + j], x[row + half + j]);
                y[row + j] = x1 * c - x2 * sn;
                y[row + half + j] = x2 * c + x1 * sn;
            }
        }
    }
    y
}
