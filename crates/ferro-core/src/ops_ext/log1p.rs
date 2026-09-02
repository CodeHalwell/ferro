//! `log1p` operator. Forward: y = ln(1 + x), for x > -1, computed with
//! `f32::ln_1p` for accuracy near 0. Backward: d/dx ln(1 + x) = 1 / (1 + x),
//! so dx = g / (1 + x). No device kernel exists for this op, so both forward
//! and backward go through the host `raw_binary` path.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn log1p(&self) -> Result<Tensor> {
        let out = raw_binary("log1p", self, self, |v, _| v.ln_1p())?.to_device(self.device())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("log1p_bw", g, &x, |gg, xx| gg / (1.0 + xx)).unwrap()]
        }))
    }
}
