//! normalize: x / max(||x||_2 over dim, eps), keepdim so the norm broadcasts
//! back against x. Composed from existing ops, so autograd flows through the
//! composition and no backward closure is needed.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn normalize(&self, dim: usize, eps: f32) -> Result<Tensor> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape {
                op: "normalize",
                msg: format!("dim {dim} out of range for rank {ndim}"),
            });
        }
        let norm = self.square()?.sum_dim(dim, true)?.sqrt().clamp(eps, f32::MAX);
        self.div(&norm)
    }
}
