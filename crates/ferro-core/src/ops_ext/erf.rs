//! `erf` operator. Forward: y = erf(x), the Gauss error function, reusing
//! the shared `erf_f32` approximation so host backends agree bitwise.
//! Backward: d/dx erf(x) = 2/sqrt(pi) * exp(-x^2), so dx = g * 2/sqrt(pi) *
//! exp(-x^2). No device kernel exists for this op, so both forward and
//! backward go through the host `raw_binary` path.

use crate::dispatch::erf_f32;
use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn erf(&self) -> Result<Tensor> {
        let out = raw_binary("erf", self, self, |v, _| erf_f32(v))?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("erf_bw", g, &x, |gg, xx| {
                gg * std::f32::consts::FRAC_2_SQRT_PI * (-xx * xx).exp()
            })
            .unwrap()]
        }))
    }
}
