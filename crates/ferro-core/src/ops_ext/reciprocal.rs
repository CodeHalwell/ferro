//! reciprocal with the saved-output adjoint (-g) * (saved_y * saved_y).
//! Device tensors retain the explicit host-kernel fallback.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn reciprocal(&self) -> Result<Tensor> {
        if self.dtype() == crate::DType::F64 {
            return crate::typed_math::unary_grad_tensor(self, "reciprocal", f64::recip, true, |g, y| g.neg().mul(&y.mul(y)?));
        }
        let out = if self.device() == crate::Device::Cpu {
            crate::typed_math::pointwise_binary_real("reciprocal", self, self, |x,_| x.recip(), |_,_| unreachable!("float32 branch"))?
        } else { raw_binary("reciprocal", self, self, |x,_| x.recip())?.to_device(self.device())? };
        let y = out.detach_copy_layout()?;
        Ok(out.record_fn_saved(vec![self.clone()], &[], true, move |g| {
            vec![if g.device() == crate::Device::Cpu { g.neg().mul(&y.mul(&y).unwrap()).unwrap() }
            else { raw_binary("reciprocal_bw", g, &y, |gg,yy| (-gg) * (yy * yy)).unwrap() }]
        }))
    }
}
