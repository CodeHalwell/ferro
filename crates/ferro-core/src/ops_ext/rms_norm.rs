//! `rms_norm` over the last dimension (affine weight optional):
//! y = x / sqrt(mean(x^2) + eps) * w per row, no mean subtraction.
//! Composed from existing ops (square, mean_dim, add, rsqrt, mul), so
//! autograd flows through the composition and no backward closure is
//! needed. eps is built on the input's device with `full_on` so the
//! composition stays device-resident when its pieces are.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn rms_norm(&self, weight: Option<&Tensor>, eps: f32) -> Result<Tensor> {
        let op = "rms_norm";
        let ndim = self.ndim();
        if ndim == 0 {
            return Err(Error::InvalidShape { op, msg: "cannot normalize a scalar".into() });
        }
        let d = self.shape()[ndim - 1];
        if d == 0 {
            return Err(Error::InvalidShape { op, msg: "last dimension must be nonempty".into() });
        }
        if let Some(w) = weight {
            if w.shape() != [d] {
                return Err(Error::ShapeMismatch { op, lhs: vec![d], rhs: w.shape().to_vec() });
            }
        }
        let eps = Tensor::full_on(&[], eps, self.device())?;
        let inv_rms = self.square()?.mean_dim(ndim - 1, true)?.add(&eps)?.rsqrt()?;
        let y = self.mul(&inv_rms)?;
        match weight {
            Some(w) => y.mul(w),
            None => Ok(y),
        }
    }
}
