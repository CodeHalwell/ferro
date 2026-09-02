//! addcdiv: out = self + value * (t1 / t2). Composed from existing div/mul/
//! add and a scalar tensor, so autograd flows through the composition and no
//! backward closure is needed. The scalar is built on self's device, not
//! `Tensor::scalar` (always cpu), so this stays composable with device-
//! resident inputs.

use crate::error::Result;
use crate::tensor::Tensor;

impl Tensor {
    pub fn addcdiv(&self, t1: &Tensor, t2: &Tensor, value: f32) -> Result<Tensor> {
        let scalar = Tensor::full_on(&[], value, self.device())?;
        self.add(&t1.div(t2)?.mul(&scalar)?)
    }
}
