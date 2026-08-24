//! `mean_dim` operator: mean reduction over one dim, composed as
//! `sum_dim(dim, keepdim) * (1/n)` so gradients flow through the existing
//! `sum_dim` and `mul` backwards.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn mean_dim(&self, dim: usize, keepdim: bool) -> Result<Tensor> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape {
                op: "mean_dim",
                msg: format!("dim {dim} out of range for rank {ndim}"),
            });
        }
        // No max(1) guard: like mean(), an empty reduced dim yields NaN
        // (0 * inf) to match torch rather than a silent 0.
        let n = self.shape()[dim] as f32;
        // sum_dim on a device VIEW falls back to a cpu tensor (only whole
        // device buffers reduce in place), so the scale scalar must live on the
        // sum's device, not self's, or a device-view mean hits DeviceMismatch.
        let s = self.sum_dim(dim, keepdim)?;
        let scale = Tensor::full_on(&[], 1.0 / n, s.device())?;
        s.mul(&scale)
    }
}
