//! `threshold` operator: y = x for x > threshold, else value. Backward:
//! dy/dx = 1 for x > threshold, else 0. No device kernel exists for this op,
//! so both forward and backward go through the host `raw_binary` path.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn threshold(&self, threshold: f32, value: f32) -> Result<Tensor> {
        let out = raw_binary("threshold", self, self, move |v, _| {
            if v > threshold {
                v
            } else {
                value
            }
        })?.to_device(self.device())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("threshold_bw", g, &x, move |gg, xx| {
                if xx > threshold {
                    gg
                } else {
                    0.0
                }
            })
            .unwrap()]
        }))
    }
}
