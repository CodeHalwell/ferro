//! `var_dim`: variance over one dim with a Bessel correction, computed as
//! mean((x - mean(x, dim, keepdim=true))^2, dim) * n / (n - correction).
//! Composed from existing mean_dim/sub/square/mul, so autograd flows through
//! the composition and no backward closure is needed.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn var_dim(&self, dim: usize, correction: usize, keepdim: bool) -> Result<Tensor> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape {
                op: "var_dim",
                msg: format!("dim {dim} out of range for rank {ndim}"),
            });
        }
        let n = self.shape()[dim];
        if correction >= n {
            return Err(Error::InvalidShape {
                op: "var_dim",
                msg: format!("correction {correction} >= size {n} of dim {dim}"),
            });
        }
        let mean = self.mean_dim(dim, true)?;
        let sq = self.sub(&mean)?.square()?;
        let biased = sq.mean_dim(dim, keepdim)?;
        let scale = n as f32 / (n - correction) as f32;
        let scale = Tensor::full_on(&[], scale, biased.device())?;
        biased.mul(&scale)
    }
}
