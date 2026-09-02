//! `elu` operator: y = x for x > 0, else alpha * (exp(x) - 1). Backward:
//! dy/dx = 1 for x > 0, else alpha * exp(x). No device kernel exists for
//! this op, so both forward and backward go through the host `raw_binary`
//! path.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn elu(&self, alpha: f32) -> Result<Tensor> {
        let out = raw_binary("elu", self, self, move |v, _| {
            if v > 0.0 {
                v
            } else {
                alpha * v.exp_m1()
            }
        })?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("elu_bw", g, &x, move |gg, xx| {
                if xx > 0.0 {
                    gg
                } else {
                    gg * alpha * xx.exp()
                }
            })
            .unwrap()]
        }))
    }
}
