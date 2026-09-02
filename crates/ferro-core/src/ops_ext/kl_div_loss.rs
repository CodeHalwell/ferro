//! kl_div_loss: torch's F.kl_div. self is a log-probability input, target a
//! probability; per-element target*(ln(target) - input), mean-reduced.
//! Composed from existing ops (xlogy gives target*ln(target) with the 0-at-
//! target==0 convention), so autograd flows through the composition and no
//! backward closure is needed.

use crate::error::Result;
use crate::tensor::Tensor;

impl Tensor {
    pub fn kl_div_loss(&self, target: &Tensor) -> Result<Tensor> {
        let term = target.mul(self)?;
        Ok(target.xlogy(target)?.sub(&term)?.mean())
    }
}
