//! `celu` operator: y = max(0, x) + min(0, alpha * (exp(x/alpha) - 1)).
//! Backward: dy/dx = 1 for x > 0, else exp(x/alpha). No device kernel exists
//! for this op, so both forward and backward go through the host `raw_binary`
//! path.

use crate::error::{Error, Result};
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn celu(&self, alpha: f32) -> Result<Tensor> {
        if alpha == 0.0 {
            return Err(Error::InvalidShape {
                op: "celu",
                msg: "alpha must be nonzero".to_string(),
            });
        }
        let out = raw_binary("celu", self, self, move |v, _| {
            if v > 0.0 {
                v
            } else {
                alpha * (v / alpha).exp_m1()
            }
        })?.to_device(self.device())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("celu_bw", g, &x, move |gg, xx| {
                if xx > 0.0 {
                    gg
                } else {
                    gg * (xx / alpha).exp()
                }
            })
            .unwrap()]
        }))
    }
}
