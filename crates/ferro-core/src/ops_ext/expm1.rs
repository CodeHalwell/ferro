//! `expm1` operator. Forward: y = exp(x) - 1, computed via f32::exp_m1 for
//! accuracy near x = 0. Backward: d/dx expm1(x) = exp(x), so dx = g * exp(x).
//! No device kernel exists for this op, so both forward and backward go
//! through the host `raw_binary` path.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn expm1(&self) -> Result<Tensor> {
        let out = raw_binary("expm1", self, self, |v, _| v.exp_m1())?.to_device(self.device())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("expm1_bw", g, &x, |gg, xx| gg * xx.exp()).unwrap()]
        }))
    }
}
