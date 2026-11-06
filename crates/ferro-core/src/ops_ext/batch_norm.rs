//! Functional batch normalization with returned running statistics.
//! Nonempty training requires at least two elements per channel. Empty batches
//! preserve statistics. Evaluation uses frozen statistics. Zero channels are invalid.
//! Input [N, C, ...], rank at least two; per-channel statistics
//! over everything except C. Normalization uses the biased batch variance;
//! running_var is updated with the unbiased one (m / (m - 1)), like torch.
//! Tensors are immutable, so the updated running stats are returned alongside
//! the output rather than written back.
//!
//! Train backward uses xhat = (x - mean) / sqrt(var + eps):
//!   dx = w / std * (g - mean(g) - xhat * mean(g*xhat))
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
        if self.dtype() != DType::F32 {
            return Err(Error::DtypeMismatch {
                op,
                expected: DType::F32,
                got: self.dtype(),
            });
        }
        let ndim = self.ndim();
        if ndim < 2 {
            return Err(Error::Unsupported {
                op,
                msg: format!("expected input rank at least 2, got rank {ndim}"),
            });
        }
        let shape = self.shape().to_vec();
        let c = shape[1];
        if c == 0 {
            return Err(Error::InvalidShape {
                op,
                msg: "expected at least one channel".into(),
            });
        }
        let m: usize = self.numel() / c;
        if train && m == 1 {
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
            if t.dtype() != DType::F32 {
                return Err(Error::DtypeMismatch {
                    op,
                    expected: DType::F32,
                    got: t.dtype(),
                });
            }
            if t.shape() != [c] {
                return Err(Error::ShapeMismatch {
                    op,
                    lhs: vec![c],
                    rhs: t.shape().to_vec(),
                });
            }
            if name.starts_with("running_") && t.requires_grad() {
                return Err(Error::Unsupported { op, msg: "running statistics are not differentiable".into() });
            }
        }

        if train && eps <= 0.0 {
            return Err(Error::InvalidShape { op, msg: "eps must be positive during training".into() });
        }
        if m == 0 {
            let input_shape = shape.clone();
            let output = Tensor::from_vec(Vec::new(), &shape)?.record_fn(vec![self.clone(), weight.clone(), bias.clone()], move |_| {
                vec![Tensor::from_vec(Vec::new(), &input_shape).unwrap(), Tensor::zeros(&[c]), Tensor::zeros(&[c])]
            });
            let output = if train { output } else { output.save_tensor_versions(&[running_mean, running_var]) };
            return Ok(BatchNormOut { output, running_mean: running_mean.owned_detach_copy(), running_var: running_var.owned_detach_copy() });
        }

        let dev = self.device();
        if self.device_resident_whole() && [weight, bias, running_mean, running_var].iter().all(|t| t.device() == dev && t.device_resident_whole()) {
            if let Ok(out) = resident(self, weight, bias, running_mean, running_var, eps, train, momentum) {
                return Ok(out);
            }
        }
        let outer = shape[0];
        let spatial: usize = shape[2..].iter().product();
        let x = self.to_vec();
        let w = weight.to_vec();
        let b = bias.to_vec();

        let mut mean = vec![0.0f32; c];
        let mut var = vec![0.0f32; c];
        if train {
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
        if !needs_grad && !crate::capture::is_recording() {
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
                        for i in 0..gd.len() {
                            if chan_of(i) != chn {
                                continue;
                            }
                            let normalized = (x[i] - mean[chn]) / std;
                            dx[i] = (gd[i] - db[chn] / mf - normalized * (dw[chn] / mf)) * (w[chn] / std);
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
                }).save_tensor_versions(&[running_mean, running_var]),
                running_mean: rm_t,
                running_var: rv_t,
            })
        }
    }
}

/// Device composition of resident kernels (channel sums via two sum_dim
/// passes, broadcast arithmetic); autograd follows from the ops themselves.
#[allow(clippy::too_many_arguments)]
fn resident(x: &Tensor, weight: &Tensor, bias: &Tensor, running_mean: &Tensor, running_var: &Tensor, eps: f32, train: bool, momentum: f32) -> Result<BatchNormOut> {
    let (shape, dev) = (x.shape().to_vec(), x.device());
    let c = shape[1];
    let m = x.numel() / c;
    let x3 = x.reshape(&[shape[0], c, m / shape[0]])?;
    let scalar = |v: f32| Tensor::full_on(&[], v, dev);
    let channel_mean = |t: &Tensor| -> Result<Tensor> { t.sum_dim(2, true)?.sum_dim(0, true)?.mul(&scalar(1.0 / m as f32)?) };
    let col = |t: &Tensor| t.reshape(&[1, c, 1]);
    let (mean, var) = if train {
        let mean = channel_mean(&x3)?;
        let centered = x3.sub(&mean)?;
        (mean, channel_mean(&centered.mul(&centered)?)?)
    } else {
        (col(running_mean)?, col(running_var)?)
    };
    let xhat = x3.sub(&mean)?.mul(&var.add(&scalar(eps)?)?.pow_scalar(-0.5)?)?;
    let output = xhat.mul(&col(weight)?)?.add(&col(bias)?)?.reshape(&shape)?;
    if !train {
        return Ok(BatchNormOut { output, running_mean: running_mean.owned_detach_copy(), running_var: running_var.owned_detach_copy() });
    }
    let blend = |old: &Tensor, new: &Tensor| -> Result<Tensor> {
        old.mul(&scalar(1.0 - momentum)?)?.add(&new.detach().reshape(&[c])?.mul(&scalar(momentum)?)?)
    };
    let unbiased = var.detach().mul(&scalar(m as f32 / (m - 1) as f32)?)?;
    Ok(BatchNormOut { output, running_mean: blend(&running_mean.detach(), &mean)?.detach(), running_var: blend(&running_var.detach(), &unbiased)?.detach() })
}
