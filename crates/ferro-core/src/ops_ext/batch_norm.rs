//! Batch normalization, train mode with running stats (torch semantics).
//! Input rank-2 [N, C] or rank-4 NCHW [N, C, H, W]; per-channel statistics
//! over everything except C. Normalization uses the biased batch variance;
//! running_var is updated with the unbiased one (m / (m - 1)), like torch.
//! Tensors are immutable, so the updated running stats are returned alongside
//! the output rather than written back.
//!
//! Train backward per channel over m elements, with std = sqrt(var + eps),
//! dxh = g * w and xd = x - mean:
//!   dvar = -0.5 * sum(dxh * xd) / std^3
//!   dmean = -sum(dxh) / std + dvar * mean(-2 * xd)
//!   dx = dxh / std + dvar * 2 * xd / m + dmean / m
//!   dw = sum(g * xhat), db = sum(g)

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

pub struct BatchNormOut {
    pub output: Tensor,
    pub running_mean: Tensor,
    pub running_var: Tensor,
}

impl Tensor {
    pub fn batch_norm(
        &self,
        weight: &Tensor,
        bias: &Tensor,
        running_mean: &Tensor,
        running_var: &Tensor,
        eps: f32,
        train: bool,
        momentum: f32,
    ) -> Result<BatchNormOut> {
        let op = "batch_norm";
        let ndim = self.ndim();
        if ndim != 2 && ndim != 4 {
            return Err(Error::Unsupported {
                op,
                msg: format!("expected rank 2 or 4 input, got rank {ndim}"),
            });
        }
        let shape = self.shape().to_vec();
        let c = shape[1];
        let m: usize = shape.iter().product::<usize>() / c;
        if m < 2 {
            return Err(Error::InvalidShape {
                op,
                msg: "batch statistic needs at least 2 elements per channel".into(),
            });
        }
        for (name, t) in [
            ("weight", weight),
            ("bias", bias),
            ("running_mean", running_mean),
            ("running_var", running_var),
        ] {
            if t.dtype() != DType::F32 || t.numel() != c {
                return Err(Error::ShapeMismatch {
                    op,
                    lhs: vec![c],
                    rhs: vec![t.numel()],
                });
            }
        }

        let outer = shape[0];
        // Elements per (sample, channel); 1 when rank-2.
        let spatial: usize = if ndim == 4 { shape[2] * shape[3] } else { 1 };
        let x = self.to_vec();
        let w = weight.to_vec();
        let b = bias.to_vec();

        let mut mean = vec![0.0f32; c];
        let mut var = vec![0.0f32; c];
        for chn in 0..c {
            let mut s = 0.0f32;
            for n in 0..outer {
                let base = (n * c + chn) * spatial;
                s += x[base..base + spatial].iter().sum::<f32>();
            }
            mean[chn] = s / m as f32;
            let mut v = 0.0f32;
            for n in 0..outer {
                let base = (n * c + chn) * spatial;
                for xi in &x[base..base + spatial] {
                    let d = xi - mean[chn];
                    v += d * d;
                }
            }
            var[chn] = v / m as f32;
        }

        let (norm_mean, save_std) = if train {
            (
                mean.clone(),
                var.iter().map(|&v| (v + eps).sqrt()).collect::<Vec<_>>(),
            )
        } else {
            (
                running_mean.to_vec(),
                running_var
                    .to_vec()
                    .iter()
                    .map(|&v| (v + eps).sqrt())
                    .collect::<Vec<_>>(),
            )
        };
        let chan_of = move |i: usize| (i / spatial) % c;

        let y_data: Vec<f32> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| {
                ((xi - norm_mean[chan_of(i)]) / save_std[chan_of(i)]) * w[chan_of(i)]
                    + b[chan_of(i)]
            })
            .collect();

        let (rm_new, rv_new) = if train {
            let unbiased: Vec<f32> = var.iter().map(|&v| v * m as f32 / (m - 1) as f32).collect();
            let rm = running_mean.to_vec();
            let rv = running_var.to_vec();
            let nm: Vec<f32> = rm
                .iter()
                .zip(&mean)
                .map(|(r, &mu)| (1.0 - momentum) * r + momentum * mu)
                .collect();
            let nv: Vec<f32> = rv
                .iter()
                .zip(&unbiased)
                .map(|(r, &v)| (1.0 - momentum) * r + momentum * v)
                .collect();
            (nm, nv)
        } else {
            (running_mean.to_vec(), running_var.to_vec())
        };

        let out = Tensor::from_vec(y_data, &shape)?;
        let rm_t = Tensor::from_vec(rm_new, &[c])?;
        let rv_t = Tensor::from_vec(rv_new, &[c])?;
        let needs_grad = self.requires_grad() || weight.requires_grad() || bias.requires_grad();
        if !needs_grad {
            return Ok(BatchNormOut {
                output: out,
                running_mean: rm_t,
                running_var: rv_t,
            });
        }

        let mf = m as f32;
        if train {
            Ok(BatchNormOut {
                output: out.record_fn(vec![self.clone(), weight.clone(), bias.clone()], move |g| {
                    let gd = g.to_vec();
                    let mut dx = vec![0.0f32; gd.len()];
                    let mut dw = vec![0.0f32; c];
                    let mut db = vec![0.0f32; c];
                    for chn in 0..c {
                        let std = save_std[chn];
                        for n in 0..outer {
                            let base = (n * c + chn) * spatial;
                            for i in 0..spatial {
                                db[chn] += gd[base + i];
                                dw[chn] += gd[base + i] * (x[base + i] - norm_mean[chn]) / std;
                            }
                        }
                        let mut sum_dxh = 0.0f32;
                        let mut sum_dxh_xd = 0.0f32;
                        for (i, &gi) in gd.iter().enumerate() {
                            if chan_of(i) != chn {
                                continue;
                            }
                            let dxh = gi * w[chn];
                            sum_dxh += dxh;
                            sum_dxh_xd += dxh * (x[i] - mean[chn]);
                        }
                        let dvar = -0.5 * sum_dxh_xd / (std * std * std);
                        // sum(xd) is zero by construction of the mean, so
                        // the dvar correction to dmean vanishes.
                        let dmean = -sum_dxh / std;
                        for i in 0..gd.len() {
                            if chan_of(i) != chn {
                                continue;
                            }
                            let xd = x[i] - mean[chn];
                            dx[i] = gd[i] * w[chn] / std + dvar * 2.0 * xd / mf + dmean / mf;
                        }
                    }
                    vec![
                        Tensor::from_vec(dx, &shape).unwrap(),
                        Tensor::from_vec(dw, &[c]).unwrap(),
                        Tensor::from_vec(db, &[c]).unwrap(),
                    ]
                }),
                running_mean: rm_t,
                running_var: rv_t,
            })
        } else {
            // Eval mode normalizes with frozen running stats: the input
            // gradient is the plain inference derivative
            // g * weight / running_std, with no cross-example coupling and no
            // dependence on batch mean/var.
            let wv = w.clone();
            Ok(BatchNormOut {
                output: out.record_fn(vec![self.clone(), weight.clone(), bias.clone()], move |g| {
                    let gd = g.to_vec();
                    let mut dx = vec![0.0f32; gd.len()];
                    let mut db = vec![0.0f32; c];
                    for chn in 0..c {
                        let inv = wv[chn] / save_std[chn];
                        for n in 0..outer {
                            let base = (n * c + chn) * spatial;
                            for i in 0..spatial {
                                dx[base + i] = gd[base + i] * inv;
                                db[chn] += gd[base + i];
                            }
                        }
                    }
                    let mut dw = vec![0.0f32; c];
                    for chn in 0..c {
                        let mu = norm_mean[chn];
                        let std = save_std[chn];
                        for n in 0..outer {
                            let base = (n * c + chn) * spatial;
                            for i in 0..spatial {
                                dw[chn] += gd[base + i] * (x[base + i] - mu) / std;
                            }
                        }
                    }
                    vec![
                        Tensor::from_vec(dx, &shape).unwrap(),
                        Tensor::from_vec(dw, &[c]).unwrap(),
                        Tensor::from_vec(db, &[c]).unwrap(),
                    ]
                }),
                running_mean: rm_t,
                running_var: rv_t,
            })
        }
    }
}
