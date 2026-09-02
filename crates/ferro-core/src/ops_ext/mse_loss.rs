//! mse_loss: mean((input - target)^2), mean-reduced to a scalar. Composed
//! from existing ops, so autograd flows through the composition and no
//! backward closure is needed.

use crate::error::Result;
use crate::tensor::Tensor;

impl Tensor {
    pub fn mse_loss(&self, target: &Tensor) -> Result<Tensor> {
        Ok(self.sub(target)?.square()?.mean())
    }
}
