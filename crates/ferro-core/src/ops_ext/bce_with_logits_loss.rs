//! bce_with_logits_loss: numerically stable binary cross-entropy on logits x
//! against target t, mean(max(x,0) - x*t + ln(1 + exp(-|x|))). The last term
//! is -|x| <= 0 always, so exp(-|x|) never overflows.
//!
//! The value is piecewise via relu/abs for stability, but the true gradient
//! is smooth everywhere including x == 0: dy/dx = sigmoid(x) - t,
//! dy/dt = -x. Composing relu(x) - x*t + softplus(-|x|) and letting autograd
//! differentiate through it would instead give relu'(0) + abs'(0)-derived
//! terms, both 0 in this crate's convention, silently dropping the
//! sigmoid(0) = 0.5 contribution at x == 0. So the gradient is computed
//! directly here instead of composed.

use crate::error::Result;
use crate::tensor::{raw_binary, unbroadcast, Tensor};

impl Tensor {
    pub fn bce_with_logits_loss(&self, target: &Tensor) -> Result<Tensor> {
        let out = raw_binary("bce_with_logits_loss", self, target, |x, t| {
            x.max(0.0) - x * t + (-x.abs()).exp().ln_1p()
        })?
        .to_device(self.device())?;
        let (x, t) = (self.detach_copy(), target.detach_copy());
        let (sx, st) = (self.shape().to_vec(), target.shape().to_vec());
        let elementwise = out.record_fn(vec![self.clone(), target.clone()], move |g| {
            let dx = raw_binary("bce_with_logits_loss_dx", &x, &t, |xx, tt| {
                1.0 / (1.0 + (-xx).exp()) - tt
            })
            .unwrap();
            let gx = raw_binary("bce_with_logits_loss_gx", g, &dx, |gg, p| gg * p).unwrap();
            let gt = raw_binary("bce_with_logits_loss_gt", g, &x, |gg, xx| -gg * xx).unwrap();
            vec![unbroadcast(&gx, &sx), unbroadcast(&gt, &st)]
        });
        Ok(elementwise.mean())
    }
}
