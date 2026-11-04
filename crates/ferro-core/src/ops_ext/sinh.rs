//! sinh: y = sinh(x) = (e^x - e^-x)/2. Backward: d/dx sinh(x) = cosh(x),
//! so dx = g * cosh(x).

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn sinh(&self) -> Result<Tensor> {
        if self.dtype() == crate::DType::F64 {
            return crate::typed_math::unary_grad_tensor(self, "sinh", f64::sinh, false, |g, x| g.mul(&x.cosh()?));
        }
        let out = raw_binary("sinh", self, self, |v, _| v.sinh())?.to_device(self.device())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("sinh_bw", g, &x, |gg, xx| gg * xx.cosh()).unwrap()]
        }))
    }
}
