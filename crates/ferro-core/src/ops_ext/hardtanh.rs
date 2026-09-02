//! `hardtanh` operator with configurable bounds: y = clamp(x, min_val, max_val).
//! Backward: dy/dx = 1 strictly inside (min_val, max_val), else 0. No device
//! kernel exists for this op, so both forward and backward go through the
//! host `raw_binary` path.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn hardtanh(&self, min_val: f32, max_val: f32) -> Result<Tensor> {
        let out = raw_binary("hardtanh", self, self, move |v, _| v.max(min_val).min(max_val))?.to_device(self.device())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("hardtanh_bw", g, &x, move |gg, xx| {
                if xx > min_val && xx < max_val {
                    gg
                } else {
                    0.0
                }
            })
            .unwrap()]
        }))
    }
}
