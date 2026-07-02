//! Hyperbolic tangent. Backward: d/dx tanh(x) = 1 - tanh(x)^2, so with
//! y = tanh(x) the gradient is dx = g * (1 - y^2).

use crate::tensor::{raw_binary, raw_unary, Tensor};

impl Tensor {
    pub fn tanh(&self) -> Tensor {
        let out = raw_unary(self, |x| x.tanh());
        if !self.requires_grad() {
            return out;
        }
        let y = out.detach_copy();
        out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("tanh_bw", g, &y, |gg, yy| gg * (1.0 - yy * yy)).unwrap()]
        })
    }
}
