//! Sine. y = sin(x). Backward: d/dx sin(x) = cos(x), so dx = g * cos(x).

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn sin(&self) -> Result<Tensor> {
        let out = raw_binary("sin", self, self, |v, _| v.sin())?.to_device(self.device())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("sin_bw", g, &x, |gg, xx| gg * xx.cos()).unwrap()]
        }))
    }
}
