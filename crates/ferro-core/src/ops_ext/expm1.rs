//! expm1 with the saved-output adjoint g * (saved_y + 1).
//! Device tensors retain the explicit host-kernel fallback.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn expm1(&self) -> Result<Tensor> {
        if self.dtype() == crate::DType::F64 {
            return crate::typed_math::unary_grad_tensor(self, "expm1", f64::exp_m1, true, |g, y| g.mul(&y.add(&crate::typed_math::full_f64(&[], 1.0)?)?));
        }
        let out = if self.device() == crate::Device::Cpu {
            crate::typed_math::pointwise_binary_real("expm1", self, self, |x,_| x.exp_m1(), |_,_| unreachable!("float32 branch"))?
        } else { raw_binary("expm1", self, self, |x,_| x.exp_m1())?.to_device(self.device())? };
        let y = out.detach_copy_layout()?;
        Ok(out.record_fn_saved(vec![self.clone()], &[], true, move |g| {
            vec![if g.device() == crate::Device::Cpu { g.mul(&y.add(&Tensor::scalar(1.0)).unwrap()).unwrap() }
            else { raw_binary("expm1_bw", g, &y, |gg,yy| gg * (yy + 1.0)).unwrap() }]
        }))
    }
}
