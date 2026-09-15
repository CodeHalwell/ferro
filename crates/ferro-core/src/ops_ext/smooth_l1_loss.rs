//! smooth_l1_loss(target, beta): d = input - target; per-element loss is
//! 0.5*d^2/beta if |d| < beta else |d| - 0.5*beta (torch semantics), reduced
//! by mean to a scalar. Backward (pre-reduction): dL/dinput = d/beta if
//! |d| < beta else sign(d); dL/dtarget = -dL/dinput. Composing the elementwise
//! op with the existing `mean()` folds in the 1/numel scaling and broadcast.

use crate::error::Result;
use crate::dispatch::UnaryKind;
use crate::tensor::{raw_binary, raw_unary_k, unbroadcast, Tensor};

impl Tensor {
    pub fn smooth_l1_loss(&self, target: &Tensor, beta: f32) -> Result<Tensor> {
        if !beta.is_finite() || beta < 0.0 {
            return Err(crate::Error::Unsupported {
                op: "smooth_l1_loss",
                msg: "beta must be finite and nonnegative".into(),
            });
        }
        let out = raw_binary("smooth_l1_loss", self, target, |a, b| {
            let d = a - b;
            let ad = d.abs();
            if ad < beta {
                0.5 * d * d / beta
            } else {
                ad - 0.5 * beta
            }
        })?
        .to_device(self.device())?;
        // Place detached derivatives before recording: transfers detach history.
        let dx = raw_binary("smooth_l1_loss_dx", self, target, |a, b| {
            let d = a - b;
            if d.abs() < beta {
                d / beta
            } else if d > 0.0 {
                1.0
            } else if d < 0.0 {
                -1.0
            } else {
                0.0
            }
        })?;
        let dt = raw_unary_k(&dx, UnaryKind::Neg)?.to_device(self.device())?;
        let dx = dx.to_device(self.device())?;
        let (sx, sy) = (self.shape().to_vec(), target.shape().to_vec());
        let elem = out.record_fn(vec![self.clone(), target.clone()], move |g| {
            let ga = g.mul(&dx).unwrap();
            let gb = g.mul(&dt).unwrap();
            vec![unbroadcast(&ga, &sx), unbroadcast(&gb, &sy)]
        });
        Ok(elem.mean())
    }
}
