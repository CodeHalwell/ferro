//! pairwise_distance: p-norm of (a - b + eps) along the last dim, i.e.
//! (sum(|a - b + eps|^p, dim=-1))^(1/p). Composed from existing ops, so
//! autograd flows through the composition and no backward closure is needed.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn pairwise_distance(&self, other: &Tensor, p: f32, eps: f32) -> Result<Tensor> {
        let ndim = self.ndim();
        if ndim == 0 {
            return Err(Error::InvalidShape {
                op: "pairwise_distance",
                msg: "input must have at least 1 dim".to_string(),
            });
        }
        let eps_t = Tensor::scalar(eps).to_device(self.device())?;
        let diff = self.sub(other)?.add(&eps_t)?.abs().powf(p);
        Ok(diff.sum_dim(ndim - 1, false)?.powf(1.0 / p))
    }
}
