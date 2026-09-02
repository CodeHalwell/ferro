//! `hardshrink` operator: y = x for |x| > lambd, else 0. Backward: dx = g
//! for |x| > lambd, else 0. No device kernel exists for this op, so both
//! forward and backward go through the host `raw_binary` path.

use crate::error::{Error, Result};
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn hardshrink(&self, lambd: f32) -> Result<Tensor> {
        if lambd < 0.0 {
            return Err(Error::InvalidShape {
                op: "hardshrink",
                msg: format!("lambd must be nonnegative, got {lambd}"),
            });
        }
        let out = raw_binary("hardshrink", self, self, move |v, _| {
            if v.abs() > lambd {
                v
            } else {
                0.0
            }
        })?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("hardshrink_bw", g, &x, move |gg, xx| {
                if xx.abs() > lambd {
                    gg
                } else {
                    0.0
                }
            })
            .unwrap()]
        }))
    }
}
