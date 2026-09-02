//! `atanh` operator. Forward: y = atanh(x), for x in (-1, 1). Backward:
//! d/dx atanh(x) = 1/(1-x^2), so dx = g / (1 - x^2). No device kernel exists
//! for this op, so both forward and backward go through the host `raw_binary`
//! path, applied to the tensor with itself.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn atanh(&self) -> Result<Tensor> {
        let out = raw_binary("atanh", self, self, |v, _| v.atanh())?.to_device(self.device())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("atanh_bw", g, &x, |gg, xx| gg / (1.0 - xx * xx)).unwrap()]
        }))
    }
}
