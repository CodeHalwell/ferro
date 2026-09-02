//! addcdiv: out = self + value * (t1 / t2). Composed from existing div/mul/
//! add and a scalar tensor, so autograd flows through the composition and no
//! backward closure is needed.

use crate::error::Result;
use crate::tensor::Tensor;

impl Tensor {
    pub fn addcdiv(&self, t1: &Tensor, t2: &Tensor, value: f32) -> Result<Tensor> {
        self.add(&t1.div(t2)?.mul(&Tensor::scalar(value))?)
    }
}
