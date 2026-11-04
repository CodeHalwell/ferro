//! Cosine. y = cos(x). Backward: d/dx cos(x) = -sin(x), so dx = -g * sin(x).

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn cos(&self) -> Result<Tensor> {
        if self.dtype() == crate::DType::F64 {
            return crate::typed_math::unary_grad_tensor(self, "cos", f64::cos, false, |g, x| g.mul(&x.sin()?.neg()));
        }
        let out = raw_binary("cos", self, self, |v, _| v.cos())?.to_device(self.device())?;
        let x = self.detach_copy();
        Ok(out.record_fn_forward(vec![self.clone()], crate::autograd::ForwardOp::Cos, move |g| {
            vec![raw_binary("cos_bw", g, &x, |gg, xx| -gg * xx.sin()).unwrap()]
        }))
    }
}
