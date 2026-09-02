//! Tangent. y = tan(x). Backward: d/dx tan(x) = 1 + tan(x)^2 (= sec^2 x),
//! so dx = g * (1 + y^2). No device kernel exists for this op, so both
//! forward and backward go through the host `raw_binary` path.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn tan(&self) -> Result<Tensor> {
        let out = raw_binary("tan", self, self, |v, _| v.tan())?;
        let y = out.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("tan_bw", g, &y, |gg, yy| gg * (1.0 + yy * yy)).unwrap()]
        }))
    }
}
