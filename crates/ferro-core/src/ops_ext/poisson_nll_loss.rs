//! poisson_nll_loss: torch's F.poisson_nll_loss with the default log_input=true,
//! full=false. self is a log-rate (log_input), target the observed count;
//! mean(exp(self) - target*self). Composed from existing ops, so autograd
//! flows through the composition and no backward closure is needed.

use crate::error::Result;
use crate::tensor::Tensor;

impl Tensor {
    pub fn poisson_nll_loss(&self, target: &Tensor) -> Result<Tensor> {
        let term = target.mul(self)?;
        Ok(self.exp().sub(&term)?.mean())
    }
}
