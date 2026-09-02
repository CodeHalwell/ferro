//! mish: y = x * tanh(softplus(x)), softplus(x) = ln(1 + exp(x)) (stable: x
//! for x > 20.0). Backward: with sp = softplus(x), t = tanh(sp),
//! s = sigmoid(x), dy/dx = t + x * s * (1 - t*t).

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

fn softplus(v: f32) -> f32 {
    if v > 20.0 {
        v
    } else {
        v.exp().ln_1p()
    }
}

impl Tensor {
    pub fn mish(&self) -> Result<Tensor> {
        let out = raw_binary("mish", self, self, |v, _| v * softplus(v).tanh())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("mish_bw", g, &x, |gg, xx| {
                let t = softplus(xx).tanh();
                let s = 1.0 / (1.0 + (-xx).exp());
                gg * (t + xx * s * (1.0 - t * t))
            })
            .unwrap()]
        }))
    }
}
