//! tan with the saved-output adjoint g * (1 + saved_y * saved_y).
//! Device tensors retain the explicit host-kernel fallback.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn tan(&self) -> Result<Tensor> {
        if self.dtype() == crate::DType::F64 {
            return crate::typed_math::unary_grad_tensor(self, "tan", f64::tan, true, |g, y| g.mul(&y.mul(y)?.add(&crate::typed_math::full_f64(&[], 1.0)?)?));
        }
        let out = if self.device() == crate::Device::Cpu {
            crate::typed_math::pointwise_binary_real("tan", self, self, |x,_| x.tan(), |_,_| unreachable!("float32 branch"))?
        } else { raw_binary("tan", self, self, |x,_| x.tan())?.to_device(self.device())? };
        let y = out.detach_copy_layout()?;
        Ok(out.record_fn_saved(vec![self.clone()], &[], true, move |g| {
            vec![if g.device() == crate::Device::Cpu { g.mul(&y.mul(&y).unwrap().add(&Tensor::scalar(1.0)).unwrap()).unwrap() }
            else { raw_binary("tan_bw", g, &y, |gg,yy| gg * (1.0 + yy * yy)).unwrap() }]
        }))
    }
}
