//! exp2: y = 2^x, computed via f32::exp2. Backward: d/dx 2^x = 2^x * ln 2,
//! so dx = g * xx.exp2() * ln 2. No device kernel exists for this op, so
//! both forward and backward go through the host `raw_binary` path.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn exp2(&self) -> Result<Tensor> {
        let out = raw_binary("exp2", self, self, |v, _| v.exp2())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("exp2_bw", g, &x, |gg, xx| gg * xx.exp2() * std::f32::consts::LN_2).unwrap()]
        }))
    }
}
