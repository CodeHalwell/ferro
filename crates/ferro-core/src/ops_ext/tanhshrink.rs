//! `tanhshrink` operator. Forward: y = x - tanh(x). Backward: dy/dx = tanh(x)^2,
//! so dx = g * tanh(x)^2. No device kernel exists for this op, so both forward
//! and backward go through the host `raw_binary` path, applied to the tensor
//! with itself.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn tanhshrink(&self) -> Result<Tensor> {
        let out = raw_binary("tanhshrink", self, self, |v, _| v - v.tanh())?.to_device(self.device())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("tanhshrink_bw", g, &x, |gg, xx| gg * xx.tanh() * xx.tanh()).unwrap()]
        }))
    }
}
