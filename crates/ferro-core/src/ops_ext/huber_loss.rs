//! huber_loss: mean-reduced Huber loss. Per element, with d = input - target:
//! 0.5*d^2 if |d| <= delta, else delta*(|d| - 0.5*delta). Backward
//! (pre-reduction): d/dd huber(d) = d if |d| <= delta, else delta*sign(d);
//! the target gradient is its negation. Chaining the existing `mean()` folds
//! in the 1/numel scaling and its own backward, rather than reconstructing
//! the reduced scalar (and its autograd edge) by hand.

use crate::tensor::{raw_binary, unbroadcast, Tensor};
use crate::Result;

impl Tensor {
    pub fn huber_loss(&self, target: &Tensor, delta: f32) -> Result<Tensor> {
        if !delta.is_finite() || delta <= 0.0 {
            return Err(crate::Error::Unsupported {
                op: "huber_loss",
                msg: "delta must be finite and positive".into(),
            });
        }
        let out = raw_binary("huber_loss", self, target, move |a, b| {
            let d = a - b;
            let ad = d.abs();
            if ad <= delta {
                0.5 * d * d
            } else {
                delta * (ad - 0.5 * delta)
            }
        })?
        .to_device(self.device())?;
        // Place detached derivatives before recording: transfers detach history.
        let dx = raw_binary("huber_loss_dx", self, target, |a, b| {
            let d = a - b;
            if d.abs() <= delta {
                d
            } else {
                delta * d.signum()
            }
        })?;
        let dt = Tensor::from_vec(dx.to_vec().iter().map(|d| -d).collect(), dx.shape())?
            .to_device(self.device())?;
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
