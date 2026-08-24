//! `avg_pool2d` over NCHW input with a square kernel and stride, no padding.
//! Every window element contributes 1/(k*k); backward distributes the output
//! grad uniformly over each window so overlapping windows accumulate.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn avg_pool2d(&self, kernel: usize, stride: usize) -> Result<Tensor> {
        let op = "avg_pool2d";
        let shape = self.shape().to_vec();
        if shape.len() != 4 {
            return Err(Error::InvalidShape {
                op,
                msg: format!("expected NCHW rank-4 input, got rank {}", shape.len()),
            });
        }
        if kernel < 1 || stride < 1 {
            return Err(Error::InvalidShape {
                op,
                msg: format!("kernel ({kernel}) and stride ({stride}) must be >= 1"),
            });
        }
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        if h < kernel || w < kernel {
            return Err(Error::InvalidShape {
                op,
                msg: format!("kernel {kernel} too large for input {h}x{w}"),
            });
        }
        let out_h = (h - kernel) / stride + 1;
        let out_w = (w - kernel) / stride + 1;

        let data = self.to_vec();
        let inv = 1.0 / (kernel * kernel) as f32;
        let mut out_data = Vec::with_capacity(n * c * out_h * out_w);
        for plane in 0..n * c {
            let base = plane * h * w;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = 0.0f32;
                    for kh in 0..kernel {
                        for kw in 0..kernel {
                            acc += data[base + (oh * stride + kh) * w + (ow * stride + kw)];
                        }
                    }
                    out_data.push(acc * inv);
                }
            }
        }
        let out_shape = [n, c, out_h, out_w];
        let out = Tensor::from_vec(out_data, &out_shape)?;
        if !self.requires_grad() {
            return Ok(out);
        }

        Ok(out.record_fn(vec![self.clone()], move |g| {
            let gd = g.to_vec();
            let mut gx = vec![0.0f32; shape.iter().product()];
            for plane in 0..n * c {
                let base = plane * h * w;
                for cell in 0..out_h * out_w {
                    let oh = cell / out_w;
                    let ow = cell % out_w;
                    for kh in 0..kernel {
                        for kw in 0..kernel {
                            gx[base + (oh * stride + kh) * w + (ow * stride + kw)] +=
                                gd[plane * out_h * out_w + cell] * inv;
                        }
                    }
                }
            }
            vec![Tensor::from_vec(gx, &shape).unwrap()]
        }))
    }
}
