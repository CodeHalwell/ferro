//! `softshrink` operator: y = x - lambd for x > lambd, x + lambd for
//! x < -lambd, else 0. Backward: dy/dx = 1 for |x| > lambd, else 0. No
//! device kernel exists for this op, so both forward and backward go through
//! the host `raw_binary` path.

use crate::error::{Error, Result};
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn softshrink(&self, lambd: f32) -> Result<Tensor> {
        if lambd < 0.0 {
            return Err(Error::InvalidShape {
                op: "softshrink",
                msg: format!("lambd must be non-negative, got {lambd}"),
            });
        }
        let out = raw_binary("softshrink", self, self, move |v, _| {
            if v > lambd {
                v - lambd
            } else if v < -lambd {
                v + lambd
            } else {
                0.0
            }
        })?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("softshrink_bw", g, &x, move |gg, xx| {
                if xx > lambd || xx < -lambd {
                    gg
                } else {
                    0.0
                }
            })
            .unwrap()]
        }))
    }
}
