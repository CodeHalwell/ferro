//! cosine_similarity: sum(a*b, dim) / (norm(a, dim) * norm(b, dim)), each
//! norm floored at eps to avoid division by zero (matches torch's clamp_min
//! on the norms). Composed from existing ops, so autograd flows through the
//! composition and no backward closure is needed.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn cosine_similarity(&self, other: &Tensor, dim: usize, eps: f32) -> Result<Tensor> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape {
                op: "cosine_similarity",
                msg: format!("dim {dim} out of range for rank {ndim}"),
            });
        }
        let dot = self.mul(other)?.sum_dim(dim, false)?;
        let norm_a = self.square()?.sum_dim(dim, false)?.sqrt().clamp(eps, f32::INFINITY);
        let norm_b = other.square()?.sum_dim(dim, false)?.sqrt().clamp(eps, f32::INFINITY);
        dot.div(&norm_a.mul(&norm_b)?)
    }
}
