//! Reciprocal: y = 1 / x. Backward: d/dx (1/x) = -1/x^2, so dx = g * (-1/x^2),
//! computed here as -g / (x*x) to reuse the input snapshot directly.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn reciprocal(&self) -> Result<Tensor> {
        let out = raw_binary("reciprocal", self, self, |v, _| 1.0 / v)?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("reciprocal_bw", g, &x, |gg, xx| -gg / (xx * xx)).unwrap()]
        }))
    }
}
