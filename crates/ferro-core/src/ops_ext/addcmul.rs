//! addcmul: out = self + value * (t1 * t2). Composed from existing ops, so
//! autograd flows through the composition and no backward closure is needed.

use crate::error::Result;
use crate::tensor::Tensor;

impl Tensor {
    pub fn addcmul(&self, t1: &Tensor, t2: &Tensor, value: f32) -> Result<Tensor> {
        self.add(&t1.mul(t2)?.mul(&Tensor::scalar(value))?)
    }
}
