//! `rsqrt` operator. Forward: y = 1 / sqrt(x), domain x > 0. Backward:
//! d/dx rsqrt(x) = -0.5 * x^(-3/2) = -0.5 * y / x, so dx = g * -0.5 * y / x.
//! No device kernel exists for this op, so both forward and backward go
//! through the host `raw_binary` path, applied to the tensor with itself.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn rsqrt(&self) -> Result<Tensor> {
        let out = raw_binary("rsqrt", self, self, |v, _| 1.0 / v.sqrt())?;
        let x = self.detach_copy();
        let y = out.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            let dydx = raw_binary("rsqrt_dydx", &y, &x, |yy, xx| -0.5 * yy / xx).unwrap();
            vec![raw_binary("rsqrt_bw", g, &dydx, |gg, d| gg * d).unwrap()]
        }))
    }
}
