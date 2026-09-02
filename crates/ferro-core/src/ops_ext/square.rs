//! `square` operator. Forward: y = x * x. Backward: d/dx (x*x) = 2x, so
//! dx = g * 2x. No device kernel exists for this op, so both forward and
//! backward go through the host `raw_binary` path, applied to the tensor
//! with itself.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn square(&self) -> Result<Tensor> {
        let out = raw_binary("square", self, self, |v, _| v * v)?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("square_bw", g, &x, |gg, xx| gg * 2.0 * xx).unwrap()]
        }))
    }
}
