//! `cosh` operator. Forward: y = cosh(x). Backward: d/dx cosh(x) = sinh(x),
//! so dx = g * sinh(x). No device kernel exists for this op, so both
//! forward and backward go through the host `raw_binary` path.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn cosh(&self) -> Result<Tensor> {
        if self.dtype() == crate::DType::F64 {
            return crate::typed_math::unary_grad_tensor(self, "cosh", f64::cosh, false, |g, x| g.mul(&x.sinh()?));
        }
        let out = raw_binary("cosh", self, self, |v, _| v.cosh())?.to_device(self.device())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("cosh_bw", g, &x, |gg, xx| gg * xx.sinh()).unwrap()]
        }))
    }
}
