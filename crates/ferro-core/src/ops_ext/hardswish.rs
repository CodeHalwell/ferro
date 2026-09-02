//! Hardswish (torch semantics): y = 0 for x <= -3, x for x >= 3, else
//! x * (x + 3) / 6. Backward: dx = 0 for x < -3, g for x > 3, else
//! g * (2x + 3) / 6. No device kernel exists for this op, so both forward
//! and backward go through the host `raw_binary` path.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn hardswish(&self) -> Result<Tensor> {
        let out = raw_binary("hardswish", self, self, |v, _| {
            if v <= -3.0 {
                0.0
            } else if v >= 3.0 {
                v
            } else {
                v * (v + 3.0) / 6.0
            }
        })?.to_device(self.device())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("hardswish_bw", g, &x, |gg, xx| {
                if xx < -3.0 {
                    0.0
                } else if xx > 3.0 {
                    gg
                } else {
                    gg * (2.0 * xx + 3.0) / 6.0
                }
            })
            .unwrap()]
        }))
    }
}
