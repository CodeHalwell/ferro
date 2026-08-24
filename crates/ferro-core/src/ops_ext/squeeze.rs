//! `squeeze`/`unsqueeze` operators: insert or remove a size-1 dim, composed
//! as a single `reshape` so gradients flow through the existing reshape
//! backward.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn unsqueeze(&self, dim: usize) -> Result<Tensor> {
        let ndim = self.ndim();
        if dim > ndim {
            return Err(Error::InvalidShape {
                op: "unsqueeze",
                msg: format!("dim {dim} out of range for rank {ndim}"),
            });
        }
        let mut shape = self.shape().to_vec();
        shape.insert(dim, 1);
        self.reshape(&shape)
    }

    pub fn squeeze(&self, dim: usize) -> Result<Tensor> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape {
                op: "squeeze",
                msg: format!("dim {dim} out of range for rank {ndim}"),
            });
        }
        let mut shape = self.shape().to_vec();
        if shape[dim] != 1 {
            return Err(Error::InvalidShape {
                op: "squeeze",
                msg: format!("dim {dim} has size {} != 1", shape[dim]),
            });
        }
        shape.remove(dim);
        self.reshape(&shape)
    }
}
