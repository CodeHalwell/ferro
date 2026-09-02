//! `softplus` operator. Forward: y = ln(1 + exp(x)), computed with the
//! numerically stable form used by torch: for x above a threshold (20.0)
//! exp(x) would overflow while contributing nothing to the result, so return
//! x directly; otherwise (1 + x.exp()).ln(). Backward: dy/dx = sigmoid(x),
//! so dx = g * sigmoid(x), stable for all x via 1 / (1 + (-x).exp()). No
//! device kernel exists for this op, so both forward and backward go through
//! the host `raw_binary` path.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

const THRESHOLD: f32 = 20.0;

impl Tensor {
    pub fn softplus(&self) -> Result<Tensor> {
        let out = raw_binary("softplus", self, self, |v, _| {
            if v > THRESHOLD {
                v
            } else {
                (1.0 + v.exp()).ln()
            }
        })?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("softplus_bw", g, &x, |gg, xx| gg * (1.0 / (1.0 + (-xx).exp()))).unwrap()]
        }))
    }
}
