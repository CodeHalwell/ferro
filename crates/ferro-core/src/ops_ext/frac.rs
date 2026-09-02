//! frac: y = x - trunc(x) (torch semantics: keeps the sign of x). Backward:
//! the derivative is 1 everywhere away from integers, so dx = g. No device
//! kernel exists for this op, so both forward and backward go through the
//! host `raw_binary` path.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn frac(&self) -> Result<Tensor> {
        let out = raw_binary("frac", self, self, |v, _| v - v.trunc())?.to_device(self.device())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("frac_bw", g, &x, |gg, _| gg).unwrap()]
        }))
    }
}
