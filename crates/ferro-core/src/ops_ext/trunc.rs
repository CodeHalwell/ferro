//! `trunc` operator. Forward: y = x truncated toward zero. Backward: trunc is
//! piecewise constant, so dx = 0 everywhere. No device kernel exists for this
//! op, so both forward and backward go through the host `raw_binary` path.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn trunc(&self) -> Result<Tensor> {
        let out = raw_binary("trunc", self, self, |v, _| v.trunc())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("trunc_bw", g, &x, |_, _| 0.0).unwrap()]
        }))
    }
}
