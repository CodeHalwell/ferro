//! `cosh` operator. Forward: y = cosh(x). Backward: d/dx cosh(x) = sinh(x),
//! so dx = g * sinh(x). No device kernel exists for this op, so both
//! forward and backward go through the host `raw_binary` path.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn cosh(&self) -> Result<Tensor> {
        let out = raw_binary("cosh", self, self, |v, _| v.cosh())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("cosh_bw", g, &x, |gg, xx| gg * xx.sinh()).unwrap()]
        }))
    }
}
