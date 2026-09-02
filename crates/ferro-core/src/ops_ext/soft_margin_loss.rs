//! soft_margin_loss: mean(ln(1 + exp(-target*input))), i.e. mean(softplus(-target*input)).
//! Composed from existing ops, so autograd flows through the composition and
//! no backward closure is needed.

use crate::error::Result;
use crate::tensor::Tensor;

impl Tensor {
    pub fn soft_margin_loss(&self, target: &Tensor) -> Result<Tensor> {
        Ok(self.mul(target)?.neg().softplus()?.mean())
    }
}
