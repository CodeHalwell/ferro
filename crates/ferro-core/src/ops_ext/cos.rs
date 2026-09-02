//! Cosine. y = cos(x). Backward: d/dx cos(x) = -sin(x), so dx = -g * sin(x).

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn cos(&self) -> Result<Tensor> {
        let out = raw_binary("cos", self, self, |v, _| v.cos())?.to_device(self.device())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("cos_bw", g, &x, |gg, xx| -gg * xx.sin()).unwrap()]
        }))
    }
}
