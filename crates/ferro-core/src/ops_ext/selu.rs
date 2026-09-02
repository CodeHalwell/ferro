//! `selu` operator: y = scale * x for x > 0, else scale * alpha * (exp(x) - 1).
//! Backward: dy/dx = scale for x > 0, else scale * alpha * exp(x). No device
//! kernel exists for this op, so both forward and backward go through the
//! host `raw_binary` path.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

// SELU fixed points (Klambauer et al., 2017).
const ALPHA: f32 = 1.6732632423543772848170429916717;
const SCALE: f32 = 1.0507009873554804934193349852946;

impl Tensor {
    pub fn selu(&self) -> Result<Tensor> {
        let out = raw_binary("selu", self, self, |v, _| {
            if v > 0.0 {
                SCALE * v
            } else {
                SCALE * ALPHA * v.exp_m1()
            }
        })?.to_device(self.device())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("selu_bw", g, &x, |gg, xx| {
                if xx > 0.0 {
                    gg * SCALE
                } else {
                    gg * SCALE * ALPHA * xx.exp()
                }
            })
            .unwrap()]
        }))
    }
}
