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

use crate::device::Device;
use crate::dispatch::UnaryKind;
use crate::error::Result;
use crate::tensor::{raw_binary, raw_unary_k, unbroadcast, Tensor};

impl Tensor {
    pub fn bce_with_logits_loss(&self, target: &Tensor) -> Result<Tensor> {
        let out = raw_binary("bce_with_logits_loss", self, target, |x, t| {
            x.max(0.0) - x * t + (-x.abs()).exp().ln_1p()
        })?
        .to_device(self.device())?;
        // Place detached derivatives before recording: transfers detach history.
        let dx = raw_binary("bce_with_logits_loss_dx", self, target, |x, t| {
            1.0 / (1.0 + (-x).exp()) - t
        })?
        .to_device(self.device())?;
        let dt = raw_unary_k(&self.to_device(Device::Cpu)?, UnaryKind::Neg)?
            .to_device(self.device())?;
        let (sx, st) = (self.shape().to_vec(), target.shape().to_vec());
        let elementwise = out.record_fn(vec![self.clone(), target.clone()], move |g| {
            let gx = g.mul(&dx).unwrap();
            let gt = g.mul(&dt).unwrap();
            vec![unbroadcast(&gx, &sx), unbroadcast(&gt, &st)]
        });
        Ok(elementwise.mean())
    }
}
