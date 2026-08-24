//! `conv2d` via im2col + GEMM. self (n, c_in, h, w) with weight
//! (c_out, c_in, kh, kw) -> (n, c_out, out_h, out_w). Zero padding, no bias,
//! no dilation/groups (MVP). The output tap at (oh, ow) reads input at
//! ih = oh*stride + r - padding, iw = ow*stride + c - padding; out-of-bounds
//! taps are the zero pad (implemented via wrapping_sub + bounds check).
//!
//! Forward lowers each image to col[Cin*KH*KW, OH*OW] (rows are taps,
//! columns are output positions) and multiplies by the weight viewed as
//! W_mat[Cout, Cin*KH*KW] - already contiguous row-major in OIHW, no copy
//! needed: out_i = W_mat @ col_i is exactly the NCHW output block for image
//! i. One col buffer is reused across images to bound memory.
//!
//! Backward is the adjoint pair (both route through the same GEMM): d_weight
//! accumulates dOut_i @ col_i^T over images (col transposed into a reused
//! scratch buffer, since forward's [taps, positions] orientation is what
//! im2col naturally produces); d_input is col2im(W_mat^T @ dOut_i), where
//! col2im scatter-adds overlapping taps back into the unpadded input - the
//! exact adjoint of im2col, skipping the same padded taps im2col zero-fills.
//! W_mat^T is materialized once per backward call, not once per image.

use crate::device::Device;
use crate::dispatch::backend_for;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

// Lower one image's input at `img_off` into col[Cin*KH*KW, OH*OW]; row
// (ci,r,c) is the tap weight[_, ci, r, c] reads, column (oh,ow) the output
// position it feeds. col is fully overwritten every call (including its
// zero-padded taps), so it is safe to reuse across images without clearing.
fn im2col(
    x: &[f32],
    img_off: usize,
    c_in: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    out_h: usize,
    out_w: usize,
    col: &mut [f32],
) {
    let out_hw = out_h * out_w;
    for ci in 0..c_in {
        for r in 0..kh {
            for c in 0..kw {
                let row_off = ((ci * kh + r) * kw + c) * out_hw;
                for oh in 0..out_h {
                    let dst = row_off + oh * out_w;
                    let ih = (oh * stride + r).wrapping_sub(padding);
                    if ih >= h {
                        col[dst..dst + out_w].fill(0.0);
                        continue;
                    }
                    let src = img_off + (ci * h + ih) * w;
                    for ow in 0..out_w {
                        let iw = (ow * stride + c).wrapping_sub(padding);
                        col[dst + ow] = if iw >= w { 0.0 } else { x[src + iw] };
                    }
                }
            }
        }
    }
}

// Adjoint of im2col: scatter-add dcol[Cin*KH*KW, OH*OW] into dx at image
// `img_off`, skipping the same out-of-bounds taps im2col zero-filled.
fn col2im_add(
    dcol: &[f32],
    dx: &mut [f32],
    img_off: usize,
    c_in: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    out_h: usize,
    out_w: usize,
) {
    let out_hw = out_h * out_w;
    for ci in 0..c_in {
        for r in 0..kh {
            for c in 0..kw {
                let row_off = ((ci * kh + r) * kw + c) * out_hw;
                for oh in 0..out_h {
                    let ih = (oh * stride + r).wrapping_sub(padding);
                    if ih >= h {
                        continue;
                    }
                    let src = row_off + oh * out_w;
                    let dst = img_off + (ci * h + ih) * w;
                    for ow in 0..out_w {
                        let iw = (ow * stride + c).wrapping_sub(padding);
                        if iw >= w {
                            continue;
                        }
                        dx[dst + iw] += dcol[src + ow];
                    }
                }
            }
        }
    }
}

fn transpose(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

impl Tensor {
    pub fn conv2d(&self, weight: &Tensor, stride: usize, padding: usize) -> Result<Tensor> {
        if self.device() != weight.device() {
            return Err(Error::DeviceMismatch {
                op: "conv2d",
                lhs: self.device(),
                rhs: weight.device(),
            });
        }
        if self.ndim() != 4 || weight.ndim() != 4 {
            return Err(Error::Unsupported {
                op: "conv2d",
                msg: "input and weight must be rank 4 (NCHW / OIHW)".into(),
            });
        }
        if stride < 1 {
            return Err(Error::Unsupported {
                op: "conv2d",
                msg: "stride must be >= 1".into(),
            });
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
        let out_hw = out_h * out_w;
        let taps = c_in * kh * kw;

        let x = self.to_vec();
        let wt = weight.to_vec();
        let backend = backend_for(Device::Cpu)?;

        let mut out = vec![0.0f32; n * c_out * out_hw];
        let mut col = vec![0.0f32; taps * out_hw];
        for ni in 0..n {
            im2col(
                &x,
                ni * c_in * h * w,
                c_in,
                h,
                w,
                kh,
                kw,
                stride,
                padding,
                out_h,
                out_w,
                &mut col,
            );
            let out_i = backend.matmul(&wt, &col, c_out, taps, out_hw);
            out[ni * c_out * out_hw..(ni + 1) * c_out * out_hw].copy_from_slice(&out_i);
        }
        let out_t = Tensor::from_vec(out, &[n, c_out, out_h, out_w])?;

        Ok(
            out_t.record_fn(vec![self.clone(), weight.clone()], move |g| {
                let g_data = g.to_vec();
                let backend = backend_for(Device::Cpu).expect("cpu backend is always registered");

                let mut wt_t = vec![0.0f32; taps * c_out];
                transpose(&wt, &mut wt_t, c_out, taps);

                let mut dx = vec![0.0f32; n * c_in * h * w];
                let mut dw = vec![0.0f32; c_out * taps];
                let mut col = vec![0.0f32; taps * out_hw];
                let mut col_t = vec![0.0f32; out_hw * taps];
                for ni in 0..n {
                    let img_off = ni * c_in * h * w;
                    let g_i = &g_data[ni * c_out * out_hw..(ni + 1) * c_out * out_hw];

                    im2col(
                        &x, img_off, c_in, h, w, kh, kw, stride, padding, out_h, out_w, &mut col,
                    );

                    // dW_mat += dOut_i @ col_i^T
                    transpose(&col, &mut col_t, taps, out_hw);
                    let dw_i = backend.matmul(g_i, &col_t, c_out, out_hw, taps);
                    for (acc, v) in dw.iter_mut().zip(dw_i.iter()) {
                        *acc += v;
                    }

                    // dcol_i = W_mat^T @ dOut_i, then scatter-add into dx.
                    let dcol = backend.matmul(&wt_t, g_i, taps, c_out, out_hw);
                    col2im_add(
                        &dcol, &mut dx, img_off, c_in, h, w, kh, kw, stride, padding, out_h, out_w,
                    );
                }
                vec![
                    Tensor::from_vec(dx, &[n, c_in, h, w]).unwrap(),
                    Tensor::from_vec(dw, &[c_out, c_in, kh, kw]).unwrap(),
                ]
            }),
        )
    }
}
