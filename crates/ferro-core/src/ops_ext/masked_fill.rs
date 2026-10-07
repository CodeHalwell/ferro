//! `masked_fill(mask, value)`: replace elements where `mask != 0` with
//! `value`. The mask (any dtype, no gradient) broadcasts to `self`'s shape and
//! may not enlarge it. Host compute, so device inputs return cpu tensors;
//! backward zeroes the gradient at filled positions.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::shape::broadcast_shapes;
use crate::tensor::Tensor;

impl Tensor {
    pub fn masked_fill(&self, mask: &Tensor, value: f32) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(Error::DtypeMismatch { op: "masked_fill", expected: DType::F32, got: self.dtype() });
        }
        if mask.device() != self.device() {
            return Err(Error::DeviceMismatch { op: "masked_fill", lhs: self.device(), rhs: mask.device() });
        }
        let shape = self.shape().to_vec();
        if broadcast_shapes("masked_fill", &shape, mask.shape())? != shape {
            return Err(Error::ShapeMismatch { op: "masked_fill", lhs: shape, rhs: mask.shape().to_vec() });
        }
        let keep: Vec<bool> = mask.broadcast_to(&shape)?.to_vec_f64().into_iter().map(|m| m == 0.0).collect();
        let y = self.to_vec().into_iter().zip(&keep).map(|(x, &k)| if k { x } else { value }).collect();
        let out = Tensor::from_vec(y, &shape)?;
        Ok(out.record_fn(vec![self.clone()], move |g| {
            let gx = g.to_vec().into_iter().zip(&keep).map(|(v, &k)| if k { v } else { 0.0 }).collect();
            vec![Tensor::from_vec(gx, &shape).unwrap()]
        }))
    }
}
