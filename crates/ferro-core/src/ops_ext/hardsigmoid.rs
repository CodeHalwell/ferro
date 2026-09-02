//! `hardsigmoid` operator (torch semantics): y = 0 for x <= -3, 1 for x >= 3,
//! else x/6 + 1/2. Backward: dy/dx = 1/6 inside (-3, 3), else 0. No device
//! kernel exists for this op, so both forward and backward go through the
//! host `raw_binary` path.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn hardsigmoid(&self) -> Result<Tensor> {
        let out = raw_binary("hardsigmoid", self, self, |v, _| {
            if v <= -3.0 {
                0.0
            } else if v >= 3.0 {
                1.0
            } else {
                v / 6.0 + 0.5
            }
        })?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("hardsigmoid_bw", g, &x, |gg, xx| {
                if xx > -3.0 && xx < 3.0 {
                    gg / 6.0
                } else {
                    0.0
                }
            })
            .unwrap()]
        }))
    }
}
