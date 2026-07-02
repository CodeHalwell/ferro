//! Sum reduction over one dimension. Forward reuses the `raw_sum_dim` kernel;
//! backward expands the reduced grad back to the input shape: reshape to the
//! keepdim shape (size-1 at `dim`), broadcast to the input shape, materialize.

use crate::error::{Error, Result};
use crate::tensor::{raw_sum_dim, Tensor};

impl Tensor {
    pub fn sum_dim(&self, dim: usize, keepdim: bool) -> Result<Tensor> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape { op: "sum_dim", msg: format!("dim {dim} out of range for rank {ndim}") });
        }
        let out = raw_sum_dim(self, dim, keepdim);
        let in_shape = self.shape().to_vec();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            let mut keep_shape = in_shape.clone();
            keep_shape[dim] = 1;
            let g = g.reshape(&keep_shape).unwrap();
            vec![g.broadcast_to(&in_shape).unwrap().detach_copy()]
        }))
    }
}
