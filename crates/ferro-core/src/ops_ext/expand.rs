//! `expand`: broadcast to a larger shape as a zero-copy view (stride 0 on
//! expanded and prepended dims). Only size-1 dims may grow, as in torch.
//! Backward sums the gradient over the broadcast dims via `unbroadcast`, which
//! stays on the gradient's device.

use crate::error::{Error, Result};
use crate::tensor::{unbroadcast, Tensor};

impl Tensor {
    pub fn expand(&self, shape: &[usize]) -> Result<Tensor> {
        let cur = self.shape();
        let ok = shape.len().checked_sub(cur.len()).is_some_and(|p| cur.iter().zip(&shape[p..]).all(|(&c, &s)| c == s || c == 1));
        if !ok {
            return Err(Error::ShapeMismatch { op: "expand", lhs: cur.to_vec(), rhs: shape.to_vec() });
        }
        let out = self.capture_layout(self.broadcast_to(shape)?)?;
        let in_shape = cur.to_vec();
        Ok(out.record_fn(vec![self.clone()], move |g| vec![unbroadcast(g, &in_shape)]))
    }
}
