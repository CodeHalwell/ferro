//! Base-10 logarithm: y = log10(x). Backward: d/dx log10(x) = 1/(x*ln(10)),
//! so dx = g / (x * ln 10). Domain x > 0.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn log10(&self) -> Result<Tensor> {
        let out = raw_binary("log10", self, self, |v, _| v.log10())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("log10_bw", g, &x, |gg, xx| gg / (xx * std::f32::consts::LN_10)).unwrap()]
        }))
    }
}
