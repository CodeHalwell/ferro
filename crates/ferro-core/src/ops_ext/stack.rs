//! `stack`: join equal-shaped tensors along a new dim inserted at `dim`.
//! Composed as unsqueeze + cat, so dtype support, device behavior and the
//! gradient (cat's split-back through unsqueeze's reshape) are theirs.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn stack(tensors: &[Tensor], dim: usize) -> Result<Tensor> {
        let first = tensors.first().ok_or_else(|| Error::InvalidShape { op: "stack", msg: "expected a non-empty list of tensors".into() })?;
        if dim > first.ndim() {
            return Err(Error::InvalidShape { op: "stack", msg: format!("dim {dim} out of range for rank {}", first.ndim() + 1) });
        }
        if let Some(t) = tensors.iter().find(|t| t.shape() != first.shape()) {
            return Err(Error::ShapeMismatch { op: "stack", lhs: first.shape().to_vec(), rhs: t.shape().to_vec() });
        }
        let parts = tensors.iter().map(|t| t.unsqueeze(dim)).collect::<Result<Vec<_>>>()?;
        Tensor::cat(&parts, dim)
    }
}
