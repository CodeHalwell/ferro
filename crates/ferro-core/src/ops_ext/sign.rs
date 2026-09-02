//! `sign` operator. Forward: y = -1 for x < 0, 0 for x == 0, 1 for x > 0.
//! Backward: the true derivative is 0 almost everywhere (torch semantics),
//! so dx = 0 regardless of g. No device kernel exists for this op, so both
//! forward and backward go through the host `raw_binary` path.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn sign(&self) -> Result<Tensor> {
        let out = raw_binary("sign", self, self, |v, _| {
            if v < 0.0 {
                -1.0
            } else if v > 0.0 {
                1.0
            } else {
                0.0
            }
        })?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("sign_bw", g, &x, |_, _| 0.0).unwrap()]
        }))
    }
}
