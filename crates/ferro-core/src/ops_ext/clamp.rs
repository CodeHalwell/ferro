//! `clamp` operator. Forward computes `x.max(min).min(max)` elementwise, which
//! never panics and matches torch: min > max yields max everywhere. Backward
//! passes the gradient through only where the input is strictly inside the
//! range, else 0 (zero everywhere when min >= max).

use crate::tensor::{raw_binary, raw_unary, Tensor};

impl Tensor {
    pub fn clamp(&self, min: f32, max: f32) -> Tensor {
        let out = raw_unary(self, |x| x.max(min).min(max));
        let x = self.clone();
        out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("clamp_bw", g, &x, |gg, xx| {
                if xx > min && xx < max { gg } else { 0.0 }
            })
            .unwrap()]
        })
    }
}
