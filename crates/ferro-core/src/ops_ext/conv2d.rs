//! `conv2d` direct convolution. self (n, c_in, h, w) with weight
//! (c_out, c_in, kh, kw) -> (n, c_out, out_h, out_w). Zero padding, no bias,
//! no dilation/groups (MVP). The output tap at (oh, ow) reads input at
//! ih = oh*stride + r - padding, iw = ow*stride + c - padding; out-of-bounds
//! taps are the zero pad and are skipped. Backward reuses the same relation:
//! d_input[ih][iw] += g[oh][ow] * weight[r][c] and
//! d_weight[r][c] += g[oh][ow] * input[ih][iw].

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn conv2d(&self, weight: &Tensor, stride: usize, padding: usize) -> Result<Tensor> {
        if self.device() != weight.device() {
            return Err(Error::DeviceMismatch { op: "conv2d", lhs: self.device(), rhs: weight.device() });
        }
        if self.ndim() != 4 || weight.ndim() != 4 {
            return Err(Error::Unsupported {
                op: "conv2d",
                msg: "input and weight must be rank 4 (NCHW / OIHW)".into(),
            });
        }
        if stride < 1 {
            return Err(Error::Unsupported { op: "conv2d", msg: "stride must be >= 1".into() });
        }
        let (in_shape, w_shape) = (self.shape(), weight.shape());
        let (n, c_in, h, w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let (c_out, kh, kw) = (w_shape[0], w_shape[2], w_shape[3]);
        if w_shape[1] != c_in {
            return Err(Error::ShapeMismatch {
                op: "conv2d",
                lhs: in_shape.to_vec(),
                rhs: w_shape.to_vec(),
            });
        }
        let (ph, pw) = (h + 2 * padding, w + 2 * padding);
        if kh == 0 || kw == 0 || kh > ph || kw > pw {
            return Err(Error::InvalidShape {
                op: "conv2d",
                msg: format!("kernel ({kh},{kw}) does not fit padded input ({ph},{pw})"),
            });
        }
        let out_h = (ph - kh) / stride + 1;
        let out_w = (pw - kw) / stride + 1;

        let x = self.to_vec();
        let wt = weight.to_vec();
        let mut out = vec![0.0f32; n * c_out * out_h * out_w];
        for ni in 0..n {
            for co in 0..c_out {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut s = 0.0f32;
                        for ci in 0..c_in {
                            for r in 0..kh {
                                // Taps left of the pad wrap to huge usize and
                                // fail the `>= h` check, i.e. zero padding.
                                let ih = (oh * stride + r).wrapping_sub(padding);
                                if ih >= h {
                                    continue;
                                }
                                for c in 0..kw {
                                    let iw = (ow * stride + c).wrapping_sub(padding);
                                    if iw >= w {
                                        continue;
                                    }
                                    let xi = ((ni * c_in + ci) * h + ih) * w + iw;
                                    let wi = ((co * c_in + ci) * kh + r) * kw + c;
                                    s += x[xi] * wt[wi];
                                }
                            }
                        }
                        out[((ni * c_out + co) * out_h + oh) * out_w + ow] = s;
                    }
                }
            }
        }
        let out_t = Tensor::from_vec(out, &[n, c_out, out_h, out_w])?;

        Ok(out_t.record_fn(vec![self.clone(), weight.clone()], move |g| {
            let g_data = g.to_vec();
            let mut dx = vec![0.0f32; n * c_in * h * w];
            let mut dw = vec![0.0f32; c_out * c_in * kh * kw];
            for ni in 0..n {
                for co in 0..c_out {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let gv = g_data[((ni * c_out + co) * out_h + oh) * out_w + ow];
                            for ci in 0..c_in {
                                for r in 0..kh {
                                    let ih = (oh * stride + r).wrapping_sub(padding);
                                    if ih >= h {
                                        continue;
                                    }
                                    for c in 0..kw {
                                        let iw = (ow * stride + c).wrapping_sub(padding);
                                        if iw >= w {
                                            continue;
                                        }
                                        let xi = ((ni * c_in + ci) * h + ih) * w + iw;
                                        let wi = ((co * c_in + ci) * kh + r) * kw + c;
                                        dx[xi] += gv * wt[wi];
                                        dw[wi] += gv * x[xi];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            vec![
                Tensor::from_vec(dx, &[n, c_in, h, w]).unwrap(),
                Tensor::from_vec(dw, &[c_out, c_in, kh, kw]).unwrap(),
            ]
        }))
    }
}
