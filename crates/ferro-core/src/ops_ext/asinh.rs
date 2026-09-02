//! Inverse hyperbolic sine. y = asinh(x). Backward: d/dx asinh(x) =
//! 1/sqrt(x^2+1), so dx = g / sqrt(x^2+1). No device kernel exists for this
//! op, so both forward and backward go through the host `raw_binary` path.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn asinh(&self) -> Result<Tensor> {
        let out = raw_binary("asinh", self, self, |v, _| v.asinh())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("asinh_bw", g, &x, |gg, xx| gg / (xx * xx + 1.0).sqrt()).unwrap()]
        }))
    }
}
