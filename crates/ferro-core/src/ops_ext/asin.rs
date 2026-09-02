//! Arcsine. Forward: y = asin(x), for x in (-1, 1). Backward:
//! d/dx asin(x) = 1 / sqrt(1 - x^2), so dx = g / sqrt(1 - x^2).
//! No device kernel exists for this op; forward runs on the host via
//! `raw_binary` applied to the tensor with itself, the accepted pattern
//! for new elementwise ops without a `UnaryKind` variant.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn asin(&self) -> Result<Tensor> {
        let out = raw_binary("asin", self, self, |v, _| v.asin())?.to_device(self.device())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("asin_bw", g, &x, |gg, xx| gg / (1.0 - xx * xx).sqrt()).unwrap()]
        }))
    }
}
