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
        let (x, y) = (self.detach_copy(), target.detach_copy());
        let (sx, sy) = (self.shape().to_vec(), target.shape().to_vec());
        let elem = out.record_fn(vec![self.clone(), target.clone()], move |g| {
            let dpart = raw_binary("huber_loss_dd", &x, &y, move |a, b| {
                let d = a - b;
                if d.abs() <= delta {
                    d
                } else {
                    delta * d.signum()
                }
            })
            .unwrap();
            let ga = raw_binary("huber_loss_bwa", g, &dpart, |gg, p| gg * p).unwrap();
            let gb = raw_binary("huber_loss_bwb", g, &dpart, |gg, p| -gg * p).unwrap();
            vec![unbroadcast(&ga, &sx), unbroadcast(&gb, &sy)]
        });
        Ok(elem.mean())
    }
}
