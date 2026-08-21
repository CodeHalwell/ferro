//! `clamp` operator. Forward computes `x.max(min).min(max)` elementwise, which
//! never panics and matches torch: min > max yields max everywhere. Backward
//! passes the gradient through where min <= x <= max (inclusive bounds, like
//! torch's clamp derivative), else 0 (zero everywhere when min > max).

use crate::dispatch::UnaryKind;
use crate::tensor::{raw_binary, raw_unary_k, Tensor};

impl Tensor {
    pub fn clamp(&self, min: f32, max: f32) -> Tensor {
        let kind = UnaryKind::Clamp { min, max };
        let out = raw_unary_k(self, kind).expect("cpu backend is always registered");
        let x = self.clone();
        out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("clamp_bw", g, &x, |gg, xx| {
                if xx >= min && xx <= max { gg } else { 0.0 }
            })
            .unwrap()]
        })
    }
}
