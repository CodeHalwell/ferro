//! `floor` operator. Forward: y = floor(x). Backward: zero everywhere (torch
//! semantics - floor is piecewise-constant a.e., so dx = 0). No device kernel
//! exists for this op, so both forward and backward go through the host
//! `raw_binary` path.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn floor(&self) -> Result<Tensor> {
        let out = raw_binary("floor", self, self, |v, _| v.floor())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("floor_bw", g, &x, |_, _| 0.0).unwrap()]
        }))
    }
}
