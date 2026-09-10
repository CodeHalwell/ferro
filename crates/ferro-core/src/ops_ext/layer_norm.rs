//! `layer_norm` over the last dimension (affine weight/bias optional):
//! y = (x - mean) / sqrt(var + eps) * w + b per row, biased variance.
//! Backward for one row with xhat = (x - m)/std and dxh = g * w:
//! dx = (dxh - mean(dxh) - xhat * mean(dxh * xhat)) / std,
//! dw = sum over rows of g * xhat, db = sum over rows of g.
//! Placeholder shape-[0] grads stand in for absent affine params; the engine
//! only checks each returned grad against its own input's shape.

use crate::error::{Error, Result};
use crate::reduce::pairwise_sum_strided;
use crate::tensor::{device_leaf, Storage, Tensor};

// Unlike sum_dim's infallible raw seam, propagate a declined device kernel.
fn resident_sum(t: &Tensor, dim: usize) -> Result<Tensor> {
    let shape = t.shape().to_vec();
    let mut keep = shape.clone();
    keep[dim] = 1;
    let backend = crate::dispatch::backend_for(t.device())?;
    let storage = t.0.storage.read();
    let Storage::Device(buf) = &*storage else { unreachable!() };
    let out = device_leaf(backend.sum_dim_dev(buf.as_ref(), &shape, dim)?, &keep, t.device());
    Ok(out.record_fn_forward(vec![t.clone()], crate::autograd::ForwardOp::SumDim(dim, true), move |g| {
        vec![g.reshape(&keep).unwrap().broadcast_to(&shape).unwrap().detach_copy()]
    }))
}

impl Tensor {
    pub fn layer_norm(
        &self,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        let op = "layer_norm";
        let ndim = self.ndim();
        if ndim == 0 {
            return Err(Error::InvalidShape {
                op,
                msg: "cannot normalize a scalar".into(),
            });
        }
        let shape = self.shape().to_vec();
        let d = shape[ndim - 1];
        if d == 0 { return Err(Error::InvalidShape { op, msg: "last dimension must be nonempty".into() }); }
        if self.device_resident_whole() {
            for t in [weight, bias].into_iter().flatten() {
                if t.shape() != [d] { return Err(Error::ShapeMismatch { op, lhs: vec![d], rhs: t.shape().to_vec() }); }
                if t.device() != self.device() { return Err(Error::DeviceMismatch { op, lhs: self.device(), rhs: t.device() }); }
            }
            // Attempt only fallible kernels; a partial backend must take the
            // whole host fallback, not panic or freeze a captured intermediate.
            let resident = (|| -> Result<Tensor> {
                let scale = Tensor::full_on(&[], 1.0 / d as f32, self.device())?;
                let epsilon = Tensor::full_on(&[], eps, self.device())?;
                let mean = resident_sum(self, ndim - 1)?.mul(&scale)?;
                let centered = self.sub(&mean)?;
                let variance = resident_sum(&centered.mul(&centered)?, ndim - 1)?.mul(&scale)?;
                let v = variance.add(&epsilon)?;
                let std = crate::tensor::raw_unary_k(&v, crate::dispatch::UnaryKind::Sqrt)?;
                let saved = std.detach_copy();
                let std = std.record_fn_tagged(vec![v], crate::OpTag::Unary(crate::dispatch::UnaryKind::Sqrt), move |g| {
                    vec![crate::tensor::raw_binary("sqrt_bw", g, &saved, |gg, yy| gg * 0.5 / yy).unwrap()]
                });
                let mut out = centered.div(&std)?;
                if let Some(w) = weight { out = out.mul(w)?; }
                if let Some(b) = bias { out = out.add(b)?; }
                Ok(out)
            })();
            if let Ok(out) = resident { return Ok(out); }
        }
        let rows = self.numel() / d;
        let x = self.to_vec();
        let w = weight.map(|t| t.to_vec());
        let b = bias.map(|t| t.to_vec());
        for (name, v) in [("weight", &w), ("bias", &b)] {
            if let Some(v) = v {
                if v.len() != d {
                    return Err(Error::ShapeMismatch {
                        op,
                        lhs: vec![d],
                        rhs: vec![v.len()],
                    });
                }
            }
        }

        let mut yhat = vec![0.0f32; x.len()];
        let mut stds = vec![0.0f32; rows];
        for r in 0..rows {
            let base = r * d;
            let m = pairwise_sum_strided(&x, base, d, 1) / d as f32;
            let mut v = 0.0f32;
            for i in 0..d {
                let c = x[base + i] - m;
                v += c * c;
            }
            let s = (v / d as f32 + eps).sqrt();
            stds[r] = s;
            for i in 0..d {
                yhat[base + i] = (x[base + i] - m) / s;
            }
        }
        let y: Vec<f32> = yhat
            .iter()
            .enumerate()
            .map(|(i, &h)| {
                h * w.as_ref().map_or(1.0, |w| w[i % d]) + b.as_ref().map_or(0.0, |b| b[i % d])
            })
            .collect();
        let out = Tensor::from_vec(y, &shape)?;
        let needs_grad = self.requires_grad()
            || weight.map_or(false, |t| t.requires_grad())
            || bias.map_or(false, |t| t.requires_grad());
        if !needs_grad && !crate::capture::is_recording() {
            return Ok(out);
        }

        let none = Tensor::zeros(&[0]);
        Ok(out.record_fn_forward(
            vec![
                self.clone(),
                weight.cloned().unwrap_or_else(|| none.clone()),
                bias.cloned().unwrap_or(none),
            ],
            crate::autograd::ForwardOp::LayerNorm { weight: weight.is_some(), bias: bias.is_some(), eps },
            move |g| {
                let gd = g.to_vec();
                let mut dx = vec![0.0f32; gd.len()];
                let mut dw = vec![0.0f32; d];
                let mut db = vec![0.0f32; d];
                let has_w = w.is_some();
                let has_b = b.is_some();
                for r in 0..rows {
                    let base = r * d;
                    let s = stds[r];
                    let mut sum_dxh = 0.0f32;
                    let mut sum_dxh_h = 0.0f32;
                    for i in 0..d {
                        let dxh = gd[base + i] * w.as_ref().map_or(1.0, |w| w[i]);
                        sum_dxh += dxh;
                        sum_dxh_h += dxh * yhat[base + i];
                    }
                    for i in 0..d {
                        let dxh = gd[base + i] * w.as_ref().map_or(1.0, |w| w[i]);
                        dx[base + i] =
                            (dxh - sum_dxh / d as f32 - yhat[base + i] * sum_dxh_h / d as f32) / s;
                        if has_w {
                            dw[i] += gd[base + i] * yhat[base + i];
                        }
                        if has_b {
                            db[i] += gd[base + i];
                        }
                    }
                }
                let gw = if has_w {
                    Tensor::from_vec(dw, &[d]).unwrap()
                } else {
                    Tensor::zeros(&[0])
                };
                let gb = if has_b {
                    Tensor::from_vec(db, &[d]).unwrap()
                } else {
                    Tensor::zeros(&[0])
                };
                vec![Tensor::from_vec(dx, &shape).unwrap(), gw, gb]
            },
        ))
    }
}
