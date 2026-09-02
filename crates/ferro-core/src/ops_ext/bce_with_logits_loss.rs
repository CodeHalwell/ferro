//! bce_with_logits_loss: numerically stable binary cross-entropy on logits x
//! against target t, mean(max(x,0) - x*t + ln(1 + exp(-|x|))). The last term
//! is exactly softplus(-|x|), and -|x| <= 0 always, so the sum never
//! overflows. Composed from existing ops, so autograd flows through the
//! composition and no backward closure is needed.

use crate::error::Result;
use crate::tensor::Tensor;

impl Tensor {
    pub fn bce_with_logits_loss(&self, target: &Tensor) -> Result<Tensor> {
        let xt = self.mul(target)?;
        let stable_term = self.abs().neg().softplus()?;
        Ok(self.relu().sub(&xt)?.add(&stable_term)?.mean())
    }
}
