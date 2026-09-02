//! `std_dim`: standard deviation over one dim, sqrt(sum((x - mean(x, dim,
//! keepdim=true))^2, dim) / (n - correction)). Composed from existing
//! mean_dim/sub/square/sum_dim/sqrt, so autograd flows through the
//! composition and no backward closure is needed.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn std_dim(&self, dim: usize, correction: usize, keepdim: bool) -> Result<Tensor> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape {
                op: "std_dim",
                msg: format!("dim {dim} out of range for rank {ndim}"),
            });
        }
        let n = self.shape()[dim];
        if correction >= n {
            return Err(Error::InvalidShape {
                op: "std_dim",
                msg: format!("correction {correction} >= dim size {n}"),
            });
        }
        let mean = self.mean_dim(dim, true)?;
        let centered = self.sub(&mean)?;
        let summed = centered.square()?.sum_dim(dim, keepdim)?;
        let scale = Tensor::full_on(&[], 1.0 / (n - correction) as f32, summed.device())?;
        Ok(summed.mul(&scale)?.sqrt())
    }
}
