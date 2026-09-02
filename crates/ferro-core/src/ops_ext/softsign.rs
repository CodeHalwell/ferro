//! `softsign` operator: y = x / (1 + |x|). Backward: dy/dx = 1 / (1 + |x|)^2,
//! so dx = g / (1 + |x|)^2. No device kernel exists for this op, so both
//! forward and backward go through the host `raw_binary` path.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn softsign(&self) -> Result<Tensor> {
        let out = raw_binary("softsign", self, self, |v, _| v / (1.0 + v.abs()))?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("softsign_bw", g, &x, |gg, xx| {
                let d = 1.0 + xx.abs();
                gg / (d * d)
            })
            .unwrap()]
        }))
    }
}
