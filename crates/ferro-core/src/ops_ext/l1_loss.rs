//! l1_loss: mean(|input - target|). Composed from existing ops, so autograd
//! flows through the composition and no backward closure is needed.
use crate::error::Result;
use crate::tensor::Tensor;

impl Tensor {
    pub fn l1_loss(&self, target: &Tensor) -> Result<Tensor> {
        Ok(self.sub(target)?.abs().mean())
    }
}
