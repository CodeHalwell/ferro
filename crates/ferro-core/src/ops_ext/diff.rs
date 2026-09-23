//! diff(dim): first forward difference along `dim`, y[i] = x[i + 1] - x[i],
//! so the output is one shorter along that dim. Composed from index_select
//! and sub, so autograd flows through the composition and no backward
//! closure is needed. Requires at least two entries along `dim`.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn diff(&self, dim: usize) -> Result<Tensor> {
        let op = "diff";
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape { op, msg: format!("dim {dim} out of range for rank {ndim}") });
        }
        let n = self.shape()[dim];
        if n < 2 {
            return Err(Error::InvalidShape { op, msg: format!("need at least 2 entries along dim {dim}, got {n}") });
        }
        let hi: Vec<usize> = (1..n).collect();
        let lo: Vec<usize> = (0..n - 1).collect();
        self.index_select(dim, &hi)?.sub(&self.index_select(dim, &lo)?)
    }
}
