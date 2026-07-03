//! `abs` operator. Backward: dx = g * sign(x), with sign(0) = 0
//! (avoid f32::signum, which returns +1 at 0).

use crate::dispatch::UnaryKind;
use crate::tensor::{raw_binary, raw_unary_k, Tensor};

impl Tensor {
    pub fn abs(&self) -> Tensor {
        let out = raw_unary_k(self, UnaryKind::Abs).expect("cpu backend is always registered");
        let x = self.clone();
        out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("abs_bw", g, &x, |gg, xx| {
                if xx > 0.0 {
                    gg
                } else if xx < 0.0 {
                    -gg
                } else {
                    0.0
                }
            })
            .unwrap()]
        })
    }
}
