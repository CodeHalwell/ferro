//! `max_pool2d` over NCHW input with a square kernel and stride, no padding.
//! Forward records the flat argmax index per output cell (first NaN in a
//! window wins, ties break to the lowest flat index); backward scatter-adds
//! the output grad at those indices so overlapping windows accumulate.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn max_pool2d(&self, kernel: usize, stride: usize) -> Result<Tensor> {
        let shape = self.shape().to_vec();
        if shape.len() != 4 {
            return Err(Error::InvalidShape { op: "max_pool2d", msg: format!("expected NCHW rank-4 input, got rank {}", shape.len()) });
        }
        if kernel < 1 || stride < 1 {
            return Err(Error::InvalidShape { op: "max_pool2d", msg: format!("kernel ({kernel}) and stride ({stride}) must be >= 1") });
        }
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        if h < kernel || w < kernel {
            return Err(Error::InvalidShape { op: "max_pool2d", msg: format!("kernel {kernel} too large for input {h}x{w}") });
        }
        let out_h = (h - kernel) / stride + 1;
        let out_w = (w - kernel) / stride + 1;

        let data = self.to_vec();
        let mut out_data = Vec::with_capacity(n * c * out_h * out_w);
        let mut argmax = Vec::with_capacity(n * c * out_h * out_w);
        for plane in 0..n * c {
            let base = plane * h * w;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut arg = base + oh * stride * w + ow * stride;
                    'window: for kh in 0..kernel {
                        for kw in 0..kernel {
                            let idx = base + (oh * stride + kh) * w + (ow * stride + kw);
                            if data[idx].is_nan() {
                                arg = idx;
                                break 'window;
                            }
                            if data[idx] > data[arg] {
                                arg = idx;
                            }
                        }
                    }
                    out_data.push(data[arg]);
                    argmax.push(arg);
                }
            }
        }
        let out = Tensor::from_vec(out_data, &[n, c, out_h, out_w])?;

        Ok(out.record_fn(vec![self.clone()], move |g| {
            let gd = g.to_vec();
            let mut gx = vec![0.0f32; shape.iter().product()];
            for (cell, &arg) in argmax.iter().enumerate() {
                gx[arg] += gd[cell];
            }
            vec![Tensor::from_vec(gx, &shape).unwrap()]
        }))
    }
}
