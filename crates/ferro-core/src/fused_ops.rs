//! Fused composite ops. Each is one forward pass over the data with a single
//! self-contained backward closure recorded via `record_fn`, so intermediates
//! (bias-added pre-activations, normalized rows) never materialize as separate
//! autograd nodes.

use crate::error::{Error, Result};
use crate::reduce::pairwise_sum_strided;
use crate::tensor::Tensor;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    Identity,
    Relu,
    Gelu,
    Silu,
}

const GELU_C: f32 = 0.797_884_6; // sqrt(2/pi)
const GELU_A: f32 = 0.044715;

fn act_fwd(kind: Act, z: f32) -> f32 {
    match kind {
        Act::Identity => z,
        Act::Relu => z.max(0.0),
        Act::Gelu => 0.5 * z * (1.0 + (GELU_C * (z + GELU_A * z * z * z)).tanh()),
        Act::Silu => z / (1.0 + (-z).exp()),
    }
}

/// d(act)/dz evaluated at z, for the fused backward.
fn act_bwd(kind: Act, z: f32) -> f32 {
    match kind {
        Act::Identity => 1.0,
        Act::Relu => {
            if z > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        Act::Gelu => {
            let t = (GELU_C * (z + GELU_A * z * z * z)).tanh();
            0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * GELU_C * (1.0 + 3.0 * GELU_A * z * z)
        }
        Act::Silu => {
            let s = (-z).exp();
            let sig = 1.0 / (1.0 + s);
            sig * (1.0 + z * s * sig)
        }
    }
}

impl Tensor {
    /// Fused bias-add + activation over a 2-D+ input whose last dim matches
    /// `bias`. Computes y = act(x + b) in one pass; gradients flow to both
    /// operands through the fused chain rule with db summed over leading dims.
    pub fn bias_add_activation(&self, bias: &Tensor, act: Act) -> Result<Tensor> {
        let op = "bias_add_activation";
        if self.ndim() < 2 {
            return Err(Error::InvalidShape {
                op,
                msg: "input needs rank >= 2".into(),
            });
        }
        let shape = self.shape().to_vec();
        let d = shape[self.ndim() - 1];
        if bias.numel() != d || bias.ndim() > 1 {
            return Err(Error::ShapeMismatch {
                op,
                lhs: vec![d],
                rhs: bias.shape().to_vec(),
            });
        }
        let xv = self.to_vec();
        let bv = bias.to_vec();
        let zv: Vec<f32> = xv.iter().enumerate().map(|(i, &x)| x + bv[i % d]).collect();
        let y: Vec<f32> = zv.iter().map(|&z| act_fwd(act, z)).collect();
        let out = Tensor::from_vec(y, &shape)?;
        if !self.requires_grad() && !bias.requires_grad() {
            return Ok(out);
        }
        Ok(out.record_fn(vec![self.clone(), bias.clone()], move |g| {
            let gd = g.to_vec();
            let dx: Vec<f32> = zv
                .iter()
                .zip(&gd)
                .map(|(&z, &gg)| gg * act_bwd(act, z))
                .collect();
            let mut db = vec![0f32; d];
            for i in 0..dx.len() {
                db[i % d] += dx[i];
            }
            vec![
                Tensor::from_vec(dx, &shape).unwrap(),
                Tensor::from_vec(db, &[d]).unwrap(),
            ]
        }))
    }

    /// Fused residual + layernorm over the last dim:
    /// y = layer_norm(x + residual) * weight + bias.
    /// One pass computes the sum, per-row mean/std, and normalized values;
    /// the backward splits the LN row-gradient between x and residual and
    /// reduces affine grads, mirroring ops_ext::layer_norm's formulas.
    pub fn residual_layernorm(
        &self,
        residual: &Tensor,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        let op = "residual_layernorm";
        if self.shape() != residual.shape() {
            return Err(Error::ShapeMismatch {
                op,
                lhs: self.shape().to_vec(),
                rhs: residual.shape().to_vec(),
            });
        }
        if self.ndim() == 0 {
            return Err(Error::InvalidShape {
                op,
                msg: "cannot normalize a scalar".into(),
            });
        }
        let shape = self.shape().to_vec();
        let d = shape[shape.len() - 1];
        let rows = self.numel() / d;
        let w = weight.map(|t| t.to_vec());
        let b = bias.map(|t| t.to_vec());
        for (_, v) in [("weight", &w), ("bias", &b)] {
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
        let xv = self.to_vec();
        let rv = residual.to_vec();
        let sv: Vec<f32> = xv.iter().zip(&rv).map(|(&x, &r)| x + r).collect();

        let mut yhat = vec![0f32; sv.len()];
        let mut stds = vec![0f32; rows];
        for r in 0..rows {
            let base = r * d;
            let m = pairwise_sum_strided(&sv, base, d, 1) / d as f32;
            let mut v = 0f32;
            for i in 0..d {
                let c = sv[base + i] - m;
                v += c * c;
            }
            let s = (v / d as f32 + eps).sqrt();
            stds[r] = s;
            for i in 0..d {
                yhat[base + i] = (sv[base + i] - m) / s;
            }
        }
        let y: Vec<f32> = yhat
            .iter()
            .enumerate()
            .map(|(i, &h)| {
                h * w.as_ref().map_or(1.0, |w| w[i % d]) + b.as_ref().map_or(0.0, |bb| bb[i % d])
            })
            .collect();
        let out = Tensor::from_vec(y, &shape)?;
        let needs_grad = self.requires_grad()
            || residual.requires_grad()
            || weight.map_or(false, |t| t.requires_grad())
            || bias.map_or(false, |t| t.requires_grad());
        if !needs_grad {
            return Ok(out);
        }

        let none = Tensor::zeros(&[0]);
        Ok(out.record_fn(
            vec![
                self.clone(),
                residual.clone(),
                weight.cloned().unwrap_or_else(|| none.clone()),
                bias.cloned().unwrap_or(none),
            ],
            move |g| {
                let gd = g.to_vec();
                let mut dx = vec![0f32; gd.len()];
                let mut dw = vec![0f32; d];
                let mut dbv = vec![0f32; d];
                let has_w = w.is_some();
                let has_b = b.is_some();
                for r in 0..rows {
                    let base = r * d;
                    let s = stds[r];
                    let mut sum_dxh = 0f32;
                    let mut sum_dxh_h = 0f32;
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
                            dbv[i] += gd[base + i];
                        }
                    }
                }
                let gw = if has_w {
                    Tensor::from_vec(dw, &[d]).unwrap()
                } else {
                    Tensor::zeros(&[0])
                };
                let gb = if has_b {
                    Tensor::from_vec(dbv, &[d]).unwrap()
                } else {
                    Tensor::zeros(&[0])
                };
                // The sum x + r sends the same gradient to both operands.
                let gx = Tensor::from_vec(dx.clone(), &shape).unwrap();
                let gr = Tensor::from_vec(dx, &shape).unwrap();
                vec![gx, gr, gw, gb]
            },
        ))
    }
}
