//! `log_sigmoid` operator. Forward: y = ln(sigmoid(x)) = -softplus(-x), which
//! reuses softplus's stable form so large negative x does not underflow to
//! -inf. Backward: d/dx ln(sigmoid(x)) = sigmoid(-x) = 1 - sigmoid(x).
//! Composed from existing ops, so autograd flows through the composition and
//! no backward closure is needed.

use crate::error::Result;
use crate::tensor::Tensor;

impl Tensor {
    pub fn log_sigmoid(&self) -> Result<Tensor> {
        Ok(self.neg().softplus()?.neg())
    }
}
