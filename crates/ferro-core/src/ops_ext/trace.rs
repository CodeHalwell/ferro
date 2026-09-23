//! trace: sum of the main diagonal of a 2-d tensor, min(n, m) entries for a
//! non-square input. Masks the diagonal with a 0/1 tensor built on the
//! input's device and sums, so autograd flows through the composition and
//! the gradient is g on the diagonal and 0 elsewhere.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn trace(&self) -> Result<Tensor> {
        if self.ndim() != 2 {
            return Err(Error::InvalidShape { op: "trace", msg: format!("expected a 2-d input, got rank {}", self.ndim()) });
        }
        let (n, m) = (self.shape()[0], self.shape()[1]);
        let mask: Vec<f32> = (0..n * m).map(|i| if i / m == i % m { 1.0 } else { 0.0 }).collect();
        let mask = Tensor::from_vec(mask, &[n, m])?.to_device(self.device())?;
        Ok(self.mul(&mask)?.sum())
    }
}
