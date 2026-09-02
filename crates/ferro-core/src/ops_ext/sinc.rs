//! `sinc` operator (normalized, as torch): y = sin(pi*x)/(pi*x), with y = 1 at
//! x == 0. Backward: dx = g * f'(x), where
//! f'(x) = (cos(pi*x)*pi*x - sin(pi*x)) / (pi*x^2), and f'(0) = 0. No device
//! kernel exists for this op, so both forward and backward go through the
//! host `raw_binary` path.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn sinc(&self) -> Result<Tensor> {
        let out = raw_binary("sinc", self, self, |v, _| {
            if v == 0.0 {
                1.0
            } else {
                let px = std::f32::consts::PI * v;
                px.sin() / px
            }
        })?.to_device(self.device())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("sinc_bw", g, &x, |gg, xx| {
                if xx == 0.0 {
                    0.0
                } else {
                    let px = std::f32::consts::PI * xx;
                    gg * (px.cos() * px - px.sin()) / (std::f32::consts::PI * xx * xx)
                }
            })
            .unwrap()]
        }))
    }
}
