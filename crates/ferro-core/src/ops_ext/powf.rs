//! `powf` operator: tensor raised to a scalar power. Forward: x^p.
//! Backward: d/dx x^p = p * x^(p-1), so dx = g * p * x^(p-1).

use crate::tensor::{raw_binary, raw_unary, Tensor};

impl Tensor {
    pub fn powf(&self, p: f32) -> Tensor {
        let out = raw_unary(self, move |x| x.powf(p));
        let x = self.clone();
        out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("powf_bw", g, &x, move |gg, xx| gg * p * xx.powf(p - 1.0)).unwrap()]
        })
    }
}
