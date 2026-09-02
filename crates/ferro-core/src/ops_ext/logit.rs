//! Logit (inverse sigmoid): y = ln(x / (1 - x)), for x in (0, 1). Backward:
//! d/dx logit(x) = 1/(x*(1-x)), so dx = g / (x*(1-x)). No device kernel
//! exists for this op, so both forward and backward go through the host
//! `raw_binary` path, applied to the tensor with itself.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn logit(&self) -> Result<Tensor> {
        let out = raw_binary("logit", self, self, |v, _| (v / (1.0 - v)).ln())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("logit_bw", g, &x, |gg, xx| gg / (xx * (1.0 - xx))).unwrap()]
        }))
    }
}
