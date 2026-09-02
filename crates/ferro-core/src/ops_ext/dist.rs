//! `dist`: p-norm of (self - other) reduced to a scalar, i.e.
//! (sum(|self - other|^p))^(1/p). Composed from existing ops, so autograd
//! flows through the composition and no backward closure is needed.
use crate::error::Result;
use crate::tensor::Tensor;

impl Tensor {
    pub fn dist(&self, other: &Tensor, p: f32) -> Result<Tensor> {
        Ok(self.sub(other)?.abs().powf(p).sum().powf(1.0 / p))
    }
}
