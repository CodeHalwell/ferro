//! exp2 with the saved-output adjoint (g * saved_y) * ln(2).
//! Device tensors retain the explicit host-kernel fallback.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn exp2(&self) -> Result<Tensor> {
        if self.dtype() == crate::DType::F64 {
            return crate::typed_math::unary_grad_tensor(self, "exp2", f64::exp2, true, |g, y| g.mul(y)?.mul(&crate::typed_math::full_f64(&[], std::f64::consts::LN_2)?));
        }
        let out = if self.device() == crate::Device::Cpu {
            crate::typed_math::pointwise_binary_real("exp2", self, self, |x,_| x.exp2(), |_,_| unreachable!("float32 branch"))?
        } else { raw_binary("exp2", self, self, |x,_| x.exp2())?.to_device(self.device())? };
        let y = out.detach_copy_layout()?;
        Ok(out.record_fn_saved(vec![self.clone()], &[], true, move |g| {
            vec![if g.device() == crate::Device::Cpu { g.mul(&y).unwrap().mul(&Tensor::scalar(std::f32::consts::LN_2)).unwrap() }
            else { raw_binary("exp2_bw", g, &y, |gg,yy| (gg * yy) * std::f32::consts::LN_2).unwrap() }]
        }))
    }
}
