//! `erfc` operator. Forward: y = 1 - erf(x), the complementary error
//! function, reusing the shared `erf_f32` approximation. Backward:
//! d/dx erfc(x) = -2/sqrt(pi) * exp(-x^2), so dx = g * -2/sqrt(pi) * exp(-x^2).
//! No device kernel exists for this op, so both forward and backward go
//! through the host `raw_binary` path.

use crate::dispatch::erf_f32;
use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn erfc(&self) -> Result<Tensor> {
        let out = raw_binary("erfc", self, self, |v, _| 1.0 - erf_f32(v))?.to_device(self.device())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("erfc_bw", g, &x, |gg, xx| {
                gg * -std::f32::consts::FRAC_2_SQRT_PI * (-xx * xx).exp()
            })
            .unwrap()]
        }))
    }
}
