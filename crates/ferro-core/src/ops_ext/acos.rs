//! acos: y = acos(x). Backward: dx = g * -1 / sqrt(1 - x^2).
use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn acos(&self) -> Result<Tensor> {
        let out = raw_binary("acos", self, self, |v, _| v.acos())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("acos_bw", g, &x, |gg, xx| gg * -1.0 / (1.0 - xx * xx).sqrt()).unwrap()]
        }))
    }
}
