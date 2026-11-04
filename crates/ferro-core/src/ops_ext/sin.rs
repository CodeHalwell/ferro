//! Sine. y = sin(x). Backward: d/dx sin(x) = cos(x), so dx = g * cos(x).

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn sin(&self) -> Result<Tensor> {
        if self.dtype() == crate::DType::F64 {
            return crate::typed_math::unary_grad_tensor(self, "sin", f64::sin, false, |g, x| g.mul(&x.cos()?));
        }
        let out = raw_binary("sin", self, self, |v, _| v.sin())?.to_device(self.device())?;
        let x = self.detach_copy();
        Ok(out.record_fn_forward(vec![self.clone()], crate::autograd::ForwardOp::Sin, move |g| {
            vec![raw_binary("sin_bw", g, &x, |gg, xx| gg * xx.cos()).unwrap()]
        }))
    }
}
