//! outer(other): [n] x [m] -> [n, m] with y[i, j] = a[i] * b[j], via a
//! broadcast multiply of the [n, 1] and [1, m] views. Composed from existing
//! ops, so autograd flows through the composition and no backward closure is
//! needed.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn outer(&self, other: &Tensor) -> Result<Tensor> {
        for t in [self, other] {
            if t.ndim() != 1 {
                return Err(Error::InvalidShape { op: "outer", msg: format!("expected 1-d inputs, got rank {}", t.ndim()) });
            }
        }
        self.reshape(&[self.numel(), 1])?.mul(&other.reshape(&[1, other.numel()])?)
    }
}
