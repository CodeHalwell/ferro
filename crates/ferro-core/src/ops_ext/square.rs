//! `square` operator. Forward: y = x * x. Backward: d/dx (x*x) = 2x, so
//! dx = g * 2x. No device kernel exists for this op, so both forward and
//! backward go through the host `raw_binary` path, applied to the tensor
//! with itself.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn square(&self) -> Result<Tensor> {
        if self.dtype() == crate::DType::F64 && self.is_structural_zero() { return self.powf64(2.0); }
        if self.dtype() == crate::DType::F64 {
            return crate::typed_math::unary_grad_tensor(self, "square", |x| x * x, false, |g, x| g.mul(&x.powf64(1.0)?.mul(&crate::typed_math::full_f64(&[], 2.0)?)?));
        }
        if self.dtype() == crate::DType::I64 { return self.mul(self); }
        let out = raw_binary("square", self, self, |v, _| v * v)?.to_device(self.device())?;
        let x = self.detach_copy();
        Ok(out.record_fn_forward(vec![self.clone()], crate::autograd::ForwardOp::Square, move |g| {
            vec![raw_binary("square_bw", g, &x, |gg, xx| gg * (2.0 * xx)).unwrap()]
        }))
    }
}
