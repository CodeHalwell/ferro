//! `group_norm`: normalizes each (sample, channel-group) block over its
//! channels and spatial extent. Rank-2 [N, C] or rank-4 NCHW; num_groups must
//! divide C. Same affine/biased-variance convention as layer_norm.
//!
//! Backward per block over m elements, with xhat and dxh = g * w:
//!   dx = (dxh - mean(dxh) - xhat * mean(dxh * xhat)) / std,
//!   dw summed over samples and spatial positions, db summed over everything.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn group_norm(
        &self,
        num_groups: usize,
        weight: &Tensor,
        bias: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        let op = "group_norm";
        let ndim = self.ndim();
        if !(2..=4).contains(&ndim) {
            return Err(Error::Unsupported {
                op,
                msg: format!("expected rank 2 or 4 input, got rank {ndim}"),
            });
        }
        let shape = self.shape().to_vec();
        let (n, c) = (shape[0], shape[1]);
        if num_groups == 0 || c % num_groups != 0 {
            return Err(Error::InvalidShape {
                op,
                msg: format!("num_groups {num_groups} must divide channels {c}"),
            });
        }
        if weight.dtype() != DType::F32
            || bias.dtype() != DType::F32
            || weight.numel() != c
            || bias.numel() != c
        {
            return Err(Error::ShapeMismatch {
                op,
                lhs: vec![c],
                rhs: vec![weight.numel().max(bias.numel())],
            });
        }

        let spatial: usize = if ndim == 4 { shape[2] * shape[3] } else { 1 };
        let gc = c / num_groups;
        let block = gc * spatial;
        let x = self.to_vec();
        let w = weight.to_vec();
        let b = bias.to_vec();

        // Per-channel affine applied first so stats are taken on x * w + b's
        // normalized core: normalize (x - m)/std then scale by w, matching
        // torch which computes stats on raw x grouped by channel.
        let mut yhat = vec![0.0f32; x.len()];
        let mut stds = vec![0.0f32; n * num_groups];
        let mut means = vec![0.0f32; n * num_groups];
        for smp in 0..n {
            for grp in 0..num_groups {
                let blk = smp * num_groups + grp;
                let c0 = grp * gc;
                let start = smp * c * spatial + c0 * spatial;
                let m = pairwise_sum(&x[start..start + block]) / block as f32;
                means[blk] = m;
                let mut v = 0.0f32;
                for xi in &x[start..start + block] {
                    v += (xi - m) * (xi - m);
                }
                let s = (v / block as f32 + eps).sqrt();
                stds[blk] = s;
                for i in 0..block {
                    yhat[start + i] = (x[start + i] - m) / s;
                }
            }
        }
        let y_data: Vec<f32> = yhat
            .iter()
            .enumerate()
            .map(|(i, &h)| h * w[(i / spatial) % c] + b[(i / spatial) % c])
            .collect();
        let out = Tensor::from_vec(y_data, &shape)?;
        if !self.requires_grad() && !weight.requires_grad() && !bias.requires_grad() {
            return Ok(out);
        }

        Ok(
            out.record_fn(vec![self.clone(), weight.clone(), bias.clone()], move |g| {
                let gd = g.to_vec();
                let mut dx = vec![0.0f32; gd.len()];
                let mut dw = vec![0.0f32; c];
                let mut db = vec![0.0f32; c];
                for i in 0..gd.len() {
                    let chn = (i / spatial) % c;
                    db[chn] += gd[i];
                    dw[chn] += gd[i] * yhat[i];
                }
                for smp in 0..n {
                    for grp in 0..num_groups {
                        let blk = smp * num_groups + grp;
                        let c0 = grp * gc;
                        let start = smp * c * spatial + c0 * spatial;
                        let s = stds[blk];
                        let mut sum_dxh = 0.0f32;
                        let mut sum_dxh_h = 0.0f32;
                        for i in 0..block {
                            let chn = c0 + i / spatial;
                            let dxh = gd[start + i] * w[chn];
                            sum_dxh += dxh;
                            sum_dxh_h += dxh * yhat[start + i];
                        }
                        for i in 0..block {
                            let chn = c0 + i / spatial;
                            let dxh = gd[start + i] * w[chn];
                            dx[start + i] = (dxh
                                - sum_dxh / block as f32
                                - yhat[start + i] * sum_dxh_h / block as f32)
                                / s;
                        }
                    }
                }
                vec![
                    Tensor::from_vec(dx, &shape).unwrap(),
                    Tensor::from_vec(dw, &[c]).unwrap(),
                    Tensor::from_vec(db, &[c]).unwrap(),
                ]
            }),
        )
    }
}

fn pairwise_sum(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    if v.len() == 1 {
        return v[0];
    }
    let mid = v.len() / 2;
    pairwise_sum(&v[..mid]) + pairwise_sum(&v[mid..])
}
