//! softmin: softmax(-x, dim). Composed from existing ops, so autograd flows
//! through the composition and no backward closure is needed.
use crate::error::Result;
use crate::tensor::Tensor;

impl Tensor {
    pub fn softmin(&self, dim: usize) -> Result<Tensor> {
        self.neg().softmax(dim)
    }
}
