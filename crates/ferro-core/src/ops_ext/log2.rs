//! log2: y = log2(x), x > 0. Backward: d/dx log2(x) = 1 / (x * ln 2),
//! so dx = g / (x * ln 2).

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn log2(&self) -> Result<Tensor> {
        let out = raw_binary("log2", self, self, |v, _| v.log2())?.to_device(self.device())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("log2_bw", g, &x, |gg, xx| gg / (xx * std::f32::consts::LN_2)).unwrap()]
        }))
    }
}
