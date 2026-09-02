//! `leaky_relu` operator with a `negative_slope` parameter.
//! Forward: y = x for x >= 0, else negative_slope * x.
//! Backward: dx = g * 1 for x > 0, else g * negative_slope. No device kernel
//! exists for this op, so both forward and backward go through the host
//! `raw_binary` path.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn leaky_relu(&self, negative_slope: f32) -> Result<Tensor> {
        let out = raw_binary("leaky_relu", self, self, move |v, _| {
            if v >= 0.0 {
                v
            } else {
                negative_slope * v
            }
        })?.to_device(self.device())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("leaky_relu_bw", g, &x, move |gg, xx| {
                if xx > 0.0 {
                    gg
                } else {
                    gg * negative_slope
                }
            })
            .unwrap()]
        }))
    }
}
