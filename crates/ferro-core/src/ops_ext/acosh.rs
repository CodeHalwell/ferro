//! Inverse hyperbolic cosine: y = acosh(x) = ln(x + sqrt(x^2 - 1)), domain
//! x > 1. Backward: d/dx acosh(x) = 1 / sqrt(x^2 - 1), so dx = g / sqrt(x^2 - 1).

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn acosh(&self) -> Result<Tensor> {
        let out = raw_binary("acosh", self, self, |v, _| (v + (v * v - 1.0).sqrt()).ln())?.to_device(self.device())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("acosh_bw", g, &x, |gg, xx| gg / (xx * xx - 1.0).sqrt()).unwrap()]
        }))
    }
}
