//! Square root. Inputs assumed positive. Backward: d/dx sqrt(x) = 1/(2*sqrt(x)),
//! so with y = sqrt(x), dx = g / (2*y).

use crate::tensor::{raw_binary, raw_unary, Tensor};

impl Tensor {
    pub fn sqrt(&self) -> Tensor {
        let out = raw_unary(self, |x| x.sqrt());
        let y = out.detach_copy();
        out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("sqrt_bw", g, &y, |gg, yy| gg * 0.5 / yy).unwrap()]
        })
    }
}
