//! Insert or remove a singleton dimension without materializing strided data.
//! Shape, stride and offset remain views of the original storage.

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
        let mut stride = self.0.stride.clone();
        let inserted = if dim == ndim { 1 } else {
            stride[dim].checked_mul(shape[dim]).ok_or_else(|| Error::InvalidShape {
                op: "unsqueeze", msg: "inserted stride overflow".into(),
            })?
        };
        shape.insert(dim, 1);
        stride.insert(dim, inserted);
        singleton_view(self, shape, stride, dim, true)
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
        let mut stride = self.0.stride.clone();
        stride.remove(dim);
        singleton_view(self, shape, stride, dim, false)
    }
}

fn singleton_view(input: &Tensor, shape: Vec<usize>, stride: Vec<usize>, dim: usize, inserted: bool) -> Result<Tensor> {
    let view = Tensor::from_parts(input.0.storage.clone(), shape.clone(), stride, input.0.offset, input.device(), false, None);
    let view = input.capture_layout(view)?;
    let forward = if inserted { crate::autograd::ForwardOp::Unsqueeze(dim) } else { crate::autograd::ForwardOp::Squeeze(dim) };
    Ok(view.record_fn_forward(vec![input.clone()], forward, move |g| {
        vec![if inserted { g.squeeze(dim).unwrap() } else { g.unsqueeze(dim).unwrap() }]
    }))
}
