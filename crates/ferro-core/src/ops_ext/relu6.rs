//! `relu6` operator: y = min(max(x, 0), 6). Backward: dy/dx = 1 strictly
//! inside (0, 6), else 0. No device kernel exists for this op, so both
//! forward and backward go through the host `raw_binary` path.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn relu6(&self) -> Result<Tensor> {
        let out = raw_binary("relu6", self, self, |v, _| v.max(0.0).min(6.0))?.to_device(self.device())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("relu6_bw", g, &x, |gg, xx| {
                if xx > 0.0 && xx < 6.0 {
                    gg
                } else {
                    0.0
                }
            })
            .unwrap()]
        }))
    }
}
